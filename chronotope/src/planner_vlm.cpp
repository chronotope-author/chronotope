#include "planner_vlm.h"
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sstream>

#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <zlib.h>

namespace chronotope {

namespace {
int64_t now_ns() { timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec; }

// chronotope.controller.SYSTEM_PROMPTS 1-6, verbatim ({w}x{h} = the upscaled image size for 1-3)
const char* PROMPTS[7] = {
    "",
    "You control a graphical interface by clicking. You are shown one screenshot of a {w}x{h}-pixel screen (origin top-left, x to the right, y down). A bright white square target appears at an unknown location. If the target is visible, answer with action \"click\" and the integer pixel coordinates of its center. If no target is visible, answer with action \"wait\". Answer with the JSON object only.",
    "You control a desktop by clicking. You are shown one screenshot of a {w}x{h}-pixel screen (origin top-left, x to the right, y down). At some point a notification card with a blue button appears at an unknown position and expires shortly after. If the card is visible, answer with action \"click\" and the integer pixel coordinates of the center of its blue button. If no card is visible, answer with action \"wait\". Answer with the JSON object only.",
    "You control a game by clicking. You are shown one screenshot of a {w}x{h}-pixel screen (origin top-left, x to the right, y down). A red balloon appears at an unknown position and may drift; it disappears after a short time. If the balloon is visible, answer with action \"click\" and the integer pixel coordinates of the center of the red balloon. If no balloon is visible, answer with action \"wait\". Answer with the JSON object only.",
    "You control a desktop by clicking. You are shown one screenshot. At some point a notification card with a blue button appears at an unknown position and expires shortly after. If the card is visible, answer with action \"click\" and the coordinates of the center of its blue button on a 0-1000 grid (x: 0 = left edge, 1000 = right edge; y: 0 = top edge, 1000 = bottom edge). If no card is visible, answer with action \"wait\". Answer with the JSON object only.",
    "You control a game by clicking. You are shown one screenshot. A red balloon appears at an unknown position and may drift; it disappears after a short time. If the balloon is visible, answer with action \"click\" and the coordinates of the center of the red balloon on a 0-1000 grid (x: 0 = left edge, 1000 = right edge; y: 0 = top edge, 1000 = bottom edge). If no balloon is visible, answer with action \"wait\". Answer with the JSON object only.",
    "You control a graphical interface by clicking. You are shown one screenshot. A bright white square target appears at an unknown location. If the target is visible, answer with action \"click\" and the coordinates of its center on a 0-1000 grid (x: 0 = left edge, 1000 = right edge; y: 0 = top edge, 1000 = bottom edge). If no target is visible, answer with action \"wait\". Answer with the JSON object only.",
};
const char* USER_TEXT = "Screenshot attached. Click the target's center or wait.";
const char* ACTION_SCHEMA = "{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"click\",\"wait\"]},"
                            "\"x\":{\"type\":\"integer\"},\"y\":{\"type\":\"integer\"}},\"required\":[\"action\"],\"additionalProperties\":false}";

std::string replace_all(std::string s, const std::string& a, const std::string& b) {
    for (size_t p = s.find(a); p != std::string::npos; p = s.find(a, p + b.size())) s.replace(p, a.size(), b);
    return s;
}

std::string json_quote(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t"; else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c;
    }
    return o + "\"";
}

// the string literal that follows `"key":` (first occurrence after `from`), unescaped; empty if absent
bool json_string_after(const std::string& s, const std::string& key, size_t from, std::string* out, size_t* end = nullptr) {
    size_t k = s.find("\"" + key + "\"", from);
    if (k == std::string::npos) return false;
    size_t c = s.find(':', k);
    if (c == std::string::npos) return false;
    size_t q = s.find_first_not_of(" \t\r\n", c + 1);
    if (q == std::string::npos || s[q] != '"') return false;
    std::string o;
    for (size_t i = q + 1; i < s.size(); ++i) {
        char ch = s[i];
        if (ch == '"') { *out = o; if (end) *end = i + 1; return true; }
        if (ch == '\\' && i + 1 < s.size()) {
            char n = s[++i];
            if (n == 'n') o += '\n'; else if (n == 't') o += '\t'; else if (n == 'r') o += '\r'; else if (n == 'b') o += '\b'; else if (n == 'f') o += '\f';
            else if (n == 'u' && i + 4 < s.size()) { unsigned cp = std::strtoul(s.substr(i + 1, 4).c_str(), nullptr, 16); if (cp < 0x80) o += (char)cp; else o += '?'; i += 4; }
            else o += n;
        } else o += ch;
    }
    return false;
}

bool json_int_after(const std::string& s, const std::string& key, long* out) {
    size_t k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return false;
    size_t c = s.find(':', k);
    if (c == std::string::npos) return false;
    size_t q = s.find_first_not_of(" \t\r\n", c + 1);
    if (q == std::string::npos || !(std::isdigit((unsigned char)s[q]) || s[q] == '-')) return false;
    *out = std::strtol(s.c_str() + q, nullptr, 10);
    return true;
}

void put_u32(std::vector<uint8_t>& v, uint32_t x) { v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x); }

void png_chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
    put_u32(out, (uint32_t)data.size());
    size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    uint32_t crc = (uint32_t)crc32(0L, out.data() + start, (uInt)(out.size() - start));
    put_u32(out, crc);
}
}  // namespace

std::vector<uint8_t> VlmPlanner::encode_png(const std::vector<uint8_t>& rgb, int w, int h, int scale) {
    int W = w * scale, H = h * scale;
    std::vector<uint8_t> raw((size_t)(1 + 3 * W) * H);
    for (int y = 0; y < H; ++y) {
        uint8_t* row = raw.data() + (size_t)y * (1 + 3 * W);
        row[0] = 0;
        const uint8_t* src = rgb.data() + (size_t)(y / scale) * w * 3;
        for (int x = 0; x < W; ++x) { const uint8_t* p = src + (size_t)(x / scale) * 3; row[1 + 3 * x] = p[0]; row[2 + 3 * x] = p[1]; row[3 + 3 * x] = p[2]; }
    }
    uLongf clen = compressBound((uLong)raw.size());
    std::vector<uint8_t> comp(clen);
    if (compress2(comp.data(), &clen, raw.data(), (uLong)raw.size(), 6) != Z_OK) return {};
    comp.resize(clen);
    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<uint8_t> ihdr;
    put_u32(ihdr, (uint32_t)W); put_u32(ihdr, (uint32_t)H);
    ihdr.push_back(8); ihdr.push_back(2); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    png_chunk(out, "IHDR", ihdr);
    png_chunk(out, "IDAT", comp);
    png_chunk(out, "IEND", {});
    return out;
}

std::string VlmPlanner::base64(const std::vector<uint8_t>& b) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    o.reserve((b.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < b.size(); i += 3) {
        uint32_t v = (b[i] << 16) | (b[i + 1] << 8) | b[i + 2];
        o += T[(v >> 18) & 63]; o += T[(v >> 12) & 63]; o += T[(v >> 6) & 63]; o += T[v & 63];
    }
    if (i + 1 == b.size()) { uint32_t v = b[i] << 16; o += T[(v >> 18) & 63]; o += T[(v >> 12) & 63]; o += "=="; }
    else if (i + 2 == b.size()) { uint32_t v = (b[i] << 16) | (b[i + 1] << 8); o += T[(v >> 18) & 63]; o += T[(v >> 12) & 63]; o += T[(v >> 6) & 63]; o += '='; }
    return o;
}

bool http_post_json(const std::string& url, const std::string& body, double timeout_s, std::string* response, int* status, std::string* error) {
    std::string u = url;
    if (u.rfind("http://", 0) != 0) { *error = "only http:// URLs are supported"; return false; }
    u = u.substr(7);
    size_t slash = u.find('/');
    std::string hostport = u.substr(0, slash), path = slash == std::string::npos ? "/" : u.substr(slash);
    std::string host = hostport, port = "80";
    size_t colon = hostport.find(':');
    if (colon != std::string::npos) { host = hostport.substr(0, colon); port = hostport.substr(colon + 1); }
    addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) { *error = "getaddrinfo " + host; return false; }
    int fd = -1;
    for (addrinfo* p = res; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        close(fd); fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) { *error = "connect " + hostport + ": " + std::strerror(errno); return false; }
    std::string req = "POST " + path + " HTTP/1.1\r\nHost: " + hostport + "\r\nContent-Type: application/json\r\nAccept: application/json\r\n"
                      "Connection: close\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    size_t sent = 0;
    while (sent < req.size()) {
        ssize_t n = send(fd, req.data() + sent, req.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) { *error = std::string("send: ") + std::strerror(errno); close(fd); return false; }
        sent += (size_t)n;
    }
    std::string all;
    char buf[65536];
    int64_t deadline = now_ns() + (int64_t)(timeout_s * 1e9);
    for (;;) {
        int64_t rem_ms = (deadline - now_ns()) / 1000000;
        if (rem_ms <= 0) { *error = "timeout"; close(fd); return false; }
        pollfd pf{fd, POLLIN, 0};
        int r = poll(&pf, 1, (int)std::min<int64_t>(rem_ms, 60000));
        if (r == 0) continue;
        if (r < 0) { if (errno == EINTR) continue; *error = "poll"; close(fd); return false; }
        ssize_t n = recv(fd, buf, sizeof buf, 0);
        if (n < 0) { if (errno == EINTR) continue; *error = std::string("recv: ") + std::strerror(errno); close(fd); return false; }
        if (n == 0) break;
        all.append(buf, (size_t)n);
    }
    close(fd);
    size_t hdr_end = all.find("\r\n\r\n");
    if (hdr_end == std::string::npos) { *error = "malformed HTTP response"; return false; }
    std::string head = all.substr(0, hdr_end), bodyp = all.substr(hdr_end + 4);
    *status = std::atoi(head.c_str() + 9);
    std::string lower = head;
    for (auto& ch : lower) ch = (char)std::tolower((unsigned char)ch);
    if (lower.find("transfer-encoding: chunked") != std::string::npos) {   // de-chunk
        std::string out; size_t p = 0;
        while (p < bodyp.size()) {
            size_t e = bodyp.find("\r\n", p);
            if (e == std::string::npos) break;
            size_t len = std::strtoul(bodyp.substr(p, e - p).c_str(), nullptr, 16);
            if (len == 0) break;
            out.append(bodyp, e + 2, len);
            p = e + 2 + len + 2;
        }
        bodyp = out;
    }
    *response = bodyp;
    return true;
}

std::string vlm_system_prompt(int prompt_version, int w, int h, int image_scale) {
    int v = prompt_version;
    if (v < 1 || v > 6) v = 5;
    return replace_all(replace_all(PROMPTS[v], "{w}", std::to_string(w * image_scale)), "{h}", std::to_string(h * image_scale));
}

bool vlm_parse_action(const std::string& content, VlmAnswer* a, int prompt_version, int w, int h, int image_scale) {
    std::string action;
    if (!json_string_after(content, "action", 0, &action) || action != "click") { a->click = false; return false; }
    long x, y;
    if (!json_int_after(content, "x", &x) || !json_int_after(content, "y", &y)) {
        // K4a (2026-08-27): the Edge-LLM checkpoints answer {"action": "click", "coordinate": [x, y]} (often inside a list);
        // accept the pair, as the Python normaliser does (chronotope/python/chronotope_core/edgellm.py _normalize_action).
        size_t c = content.find("\"coordinate\"");
        if (c == std::string::npos) { a->click = false; return false; }
        size_t lb = content.find('[', c); if (lb == std::string::npos) { a->click = false; return false; }
        char* endp = nullptr; const char* p0 = content.c_str() + lb + 1;
        double fx = std::strtod(p0, &endp); if (endp == p0) { a->click = false; return false; }
        while (*endp == ' ' || *endp == ',') ++endp; const char* p1 = endp;
        double fy = std::strtod(p1, &endp); if (endp == p1) { a->click = false; return false; }
        x = (long)(fx + 0.5); y = (long)(fy + 0.5);
    }
    a->image_x = (int)x; a->image_y = (int)y;
    if (prompt_version >= 4) { a->x = (int)(x * w / 1000); a->y = (int)(y * h / 1000); }   // norm1000 (VLMDirectController._to_canvas)
    else { a->x = (int)(x / image_scale); a->y = (int)(y / image_scale); }                  // image px
    a->click = true;
    return true;
}

VlmPlanner::VlmPlanner(VlmConfig cfg) : cfg_(std::move(cfg)) {
    system_ = vlm_system_prompt(cfg_.prompt_version, cfg_.w, cfg_.h, cfg_.image_scale);
}

VlmPlanner::~VlmPlanner() { if (th_.joinable()) th_.join(); }

VlmAnswer VlmPlanner::invoke(const std::vector<uint8_t>& rgb) {
    VlmAnswer a;
    a.t_start_ns = now_ns();
    std::vector<uint8_t> png = encode_png(rgb, cfg_.w, cfg_.h, cfg_.image_scale);
    std::ostringstream body;
    body << "{\"model\":" << json_quote(cfg_.model) << ",\"max_tokens\":" << cfg_.max_tokens << ",\"temperature\":" << cfg_.temperature
         << ",\"messages\":[{\"role\":\"system\",\"content\":" << json_quote(system_) << "},{\"role\":\"user\",\"content\":["
         << "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64," << base64(png) << "\"}},"
         << "{\"type\":\"text\",\"text\":" << json_quote(USER_TEXT) << "}]}],"
         << "\"chat_template_kwargs\":{\"enable_thinking\":" << (cfg_.enable_thinking ? "true" : "false") << "},"
         << "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"gui_action\",\"strict\":true,\"schema\":" << ACTION_SCHEMA << "}}}";
    std::string resp; int status = 0;
    if (!http_post_json(cfg_.base_url + "/chat/completions", body.str(), cfg_.timeout_s, &resp, &status, &a.error)) { a.t_end_ns = now_ns(); return a; }
    a.t_end_ns = now_ns();
    if (status != 200) { a.error = "HTTP " + std::to_string(status) + ": " + resp.substr(0, 300); return a; }
    json_string_after(resp, "id", 0, &a.request_id);
    json_string_after(resp, "model", 0, &a.model_reported);
    long v;
    if (json_int_after(resp, "prompt_tokens", &v)) a.input_tokens = (int)v;
    if (json_int_after(resp, "completion_tokens", &v)) a.output_tokens = (int)v;
    size_t msg = resp.find("\"message\"");
    std::string content;
    if (msg == std::string::npos || !json_string_after(resp, "content", msg, &content)) { a.error = "no message content in the response"; return a; }
    a.raw = content.substr(0, 500);
    a.ok = true;
    vlm_parse_action(content, &a, cfg_.prompt_version, cfg_.w, cfg_.h, cfg_.image_scale);
    return a;
}

void VlmPlanner::ask(const std::vector<uint8_t>& rgb) {
    if (busy_.load()) return;
    if (th_.joinable()) th_.join();
    busy_ = true; done_ = false;
    std::vector<uint8_t> copy = rgb;
    th_ = std::thread([this, copy]() { result_ = invoke(copy); done_ = true; busy_ = false; });
}

VlmAnswer VlmPlanner::take() {
    if (th_.joinable()) th_.join();
    done_ = false;
    return result_;
}

}  // namespace chronotope
