// kmbox.h — Action -> physical keyboard/mouse input; make it boring (design §9). The REPL protocol of
// chronotope.phy.kmbox (KMBox B on a CP2102N serial bridge): interrupt on open, `import km`, two helper
// definitions, then `km.move(dx,dy)` (chunked), `kmc(km.left,ms)`, `kmk(code,ms)`. Every write is stamped
// on CLOCK_MONOTONIC BEFORE the bytes leave (the decision-side stamp; never reported as A), and the echo
// is checked for the device's error markers. A fake box records the same lines in memory.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chronotope {

struct Kmbox;

Kmbox* kmbox_open(const char* port_or_null, int baud, char* err, size_t errlen);   // nullptr = autodetect by /dev/serial/by-id
Kmbox* kmbox_open_fake();
bool kmbox_is_fake(const Kmbox* k);
int64_t kmbox_click(Kmbox* k, const char* button, int hold_ms);   // returns the serial-write stamp (ns), -1 on rejection
int64_t kmbox_move(Kmbox* k, int dx, int dy, int step);            // relative, chunked to |d| <= step per report
int64_t kmbox_key(Kmbox* k, const char* name, int hold_ms);
bool kmbox_version(Kmbox* k, std::string* out);
const std::vector<std::string>& kmbox_lines(const Kmbox* k);       // every line written (fake and real)
const std::string& kmbox_error(const Kmbox* k);
void kmbox_close(Kmbox* k);
std::string kmbox_find_port();

// Dead-reckoned cursor on the target screen: home() rams the cursor into the top-left corner (relative
// moves clamp at the edge), after which relative moves track exactly without pointer acceleration.
struct Cursor {
    Kmbox* hid = nullptr;
    int w = 1920, h = 1080, x = 0, y = 0;
    bool homed = false;
    void home();
    int64_t move_to(int tx, int ty);
};

}  // namespace chronotope
