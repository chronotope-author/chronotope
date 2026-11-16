#include "kmbox.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <thread>

#include <fcntl.h>
#include <glob.h>
#include <termios.h>
#include <unistd.h>

namespace chronotope {

struct Kmbox {
    int fd = -1;
    bool fake = false;
    std::string port, error;
    std::vector<std::string> lines;
    double settle_s = 0.01;
};

namespace {
int64_t now_ns() { timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec; }
void sleep_s(double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); }

const char* ERR_MARKERS[] = {"Traceback", "Error", "NameError", "SyntaxError"};

std::map<std::string, int> keycodes() {
    std::map<std::string, int> m;
    for (int i = 0; i < 26; ++i) m[std::string(1, (char)('a' + i))] = 4 + i;
    for (int d = 1; d <= 9; ++d) m[std::to_string(d)] = 29 + d;
    m["0"] = 39; m["enter"] = 40; m["esc"] = 41; m["backspace"] = 42; m["tab"] = 43; m["space"] = 44; m["minus"] = 45; m["-"] = 45;
    m["="] = 46; m["["] = 47; m["]"] = 48; m[";"] = 51; m["'"] = 52; m[","] = 54; m["."] = 55; m["/"] = 56; m["f11"] = 68;
    m["right"] = 79; m["left"] = 80; m["down"] = 81; m["up"] = 82; m["ctrl"] = 224; m["shift"] = 225; m["alt"] = 226;
    return m;
}

std::string read_avail(Kmbox* k) {
    std::string out;
    char buf[512];
    for (;;) {
        ssize_t n = read(k->fd, buf, sizeof buf);
        if (n <= 0) break;
        out.append(buf, (size_t)n);
        if ((size_t)n < sizeof buf) break;
    }
    return out;
}

// write one REPL line; return the monotonic stamp taken immediately BEFORE the write; -1 if the device rejected it
int64_t send_raw(Kmbox* k, const std::string& line, double settle_s) {
    k->lines.push_back(line);
    if (k->fake) return now_ns();
    std::string data = line + "\r\n";
    int64_t t0 = now_ns();
    ssize_t w = write(k->fd, data.data(), data.size());
    if (w != (ssize_t)data.size()) { k->error = std::string("serial write: ") + std::strerror(errno); return -1; }
    tcdrain(k->fd);
    sleep_s(settle_s);
    std::string reply = read_avail(k);
    for (const char* m : ERR_MARKERS)
        if (reply.find(m) != std::string::npos) { k->error = "device rejected '" + line + "': " + reply; return -1; }
    return t0;
}
}  // namespace

std::string kmbox_find_port() {
    for (const char* pat : {"/dev/serial/by-id/*CP2102N*", "/dev/serial/by-id/usb-1a86_*"}) {
        glob_t g{};
        if (glob(pat, 0, nullptr, &g) == 0 && g.gl_pathc > 0) { std::string p = g.gl_pathv[0]; globfree(&g); return p; }
        globfree(&g);
    }
    return "";
}

Kmbox* kmbox_open(const char* port_or_null, int baud, char* err, size_t errlen) {
    std::string port = port_or_null ? port_or_null : kmbox_find_port();
    if (port.empty()) { if (err && errlen) std::snprintf(err, errlen, "no KMBox serial port found (pass --hid serial:/dev/ttyUSBn)"); return nullptr; }
    auto* k = new Kmbox();
    k->port = port;
    k->fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (k->fd < 0) { if (err && errlen) std::snprintf(err, errlen, "%s: %s", port.c_str(), std::strerror(errno)); delete k; return nullptr; }
    termios tio{};
    tcgetattr(k->fd, &tio);
    cfmakeraw(&tio);
    speed_t sp = (baud == 9600) ? B9600 : (baud == 57600) ? B57600 : B115200;
    cfsetispeed(&tio, sp); cfsetospeed(&tio, sp);
    tio.c_cflag |= CLOCAL | CREAD;
    tcsetattr(k->fd, TCSANOW, &tio);
    tcflush(k->fd, TCIOFLUSH);
    // interrupt the REPL (raw 0x03, NOT a HID Ctrl+C); a continuation-mode device cannot masquerade as a working one
    char ctrl_c = 3;
    (void)!write(k->fd, &ctrl_c, 1);
    sleep_s(0.3);
    (void)!write(k->fd, "\r\n", 2);
    sleep_s(0.3);
    std::string reply = read_avail(k);
    while (!reply.empty() && (reply.back() == '\r' || reply.back() == '\n' || reply.back() == ' ')) reply.pop_back();
    if (reply.size() >= 3 && reply.compare(reply.size() - 3, 3, "...") == 0) {
        if (err && errlen) std::snprintf(err, errlen, "REPL still in continuation mode after interrupt");
        kmbox_close(k); return nullptr;
    }
    if (send_raw(k, "import km", 0.05) < 0 ||
        send_raw(k, "import time", 0.05) < 0 ||
        send_raw(k, "exec(\"def kmc(b,ms):\\n b(1)\\n time.sleep_ms(ms)\\n b(0)\")", 0.05) < 0 ||
        send_raw(k, "exec(\"def kmk(c,ms):\\n km.down(c)\\n time.sleep_ms(ms)\\n km.up(c)\")", 0.05) < 0) {
        if (err && errlen) std::snprintf(err, errlen, "%s", k->error.c_str());
        kmbox_close(k); return nullptr;
    }
    return k;
}

Kmbox* kmbox_open_fake() { auto* k = new Kmbox(); k->fake = true; k->port = "fake"; return k; }
bool kmbox_is_fake(const Kmbox* k) { return k && k->fake; }

int64_t kmbox_click(Kmbox* k, const char* button, int hold_ms) {
    if (!k) return -1;
    return send_raw(k, std::string("kmc(km.") + (button ? button : "left") + "," + std::to_string(hold_ms) + ")", k->settle_s);
}

int64_t kmbox_move(Kmbox* k, int dx, int dy, int step) {
    if (!k) return -1;
    int64_t t0 = -1;
    while (dx != 0 || dy != 0) {
        int sx = std::max(-step, std::min(step, dx)), sy = std::max(-step, std::min(step, dy));
        int64_t t = send_raw(k, "km.move(" + std::to_string(sx) + "," + std::to_string(sy) + ")", k->settle_s);
        if (t < 0) return -1;
        if (t0 < 0) t0 = t;
        dx -= sx; dy -= sy;
    }
    return t0 < 0 ? now_ns() : t0;
}

int64_t kmbox_key(Kmbox* k, const char* name, int hold_ms) {
    static const std::map<std::string, int> codes = keycodes();
    auto it = codes.find(name ? name : "");
    if (!k || it == codes.end()) { if (k) k->error = "unknown key"; return -1; }
    return send_raw(k, "kmk(" + std::to_string(it->second) + "," + std::to_string(hold_ms) + ")", k->settle_s);
}

bool kmbox_version(Kmbox* k, std::string* out) {
    if (!k) return false;
    if (k->fake) { *out = "fake"; return true; }
    if (send_raw(k, "print(km.version())", 0.2) < 0) return false;
    *out = read_avail(k);
    return true;
}

const std::vector<std::string>& kmbox_lines(const Kmbox* k) { static const std::vector<std::string> none; return k ? k->lines : none; }
const std::string& kmbox_error(const Kmbox* k) { static const std::string none; return k ? k->error : none; }

void kmbox_close(Kmbox* k) {
    if (!k) return;
    if (k->fd >= 0) { (void)!write(k->fd, "km.zero(0)\r\n", 12); tcdrain(k->fd); close(k->fd); }
    delete k;
}

void Cursor::home() {
    kmbox_move(hid, -(w + 200), -(h + 200), 120);
    x = y = 0; homed = true;
}

int64_t Cursor::move_to(int tx, int ty) {
    if (!homed) home();
    tx = std::max(0, std::min(w - 1, tx)); ty = std::max(0, std::min(h - 1, ty));
    int64_t t = kmbox_move(hid, tx - x, ty - y, 120);
    x = tx; y = ty;
    return t;
}

}  // namespace chronotope
