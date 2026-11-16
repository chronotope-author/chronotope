// synthetic.h — a moving target rendered on the host, for headless runs and tests. The same rules as
// chronotope's moving_target manifest with the balloon skin: constant velocity after the event,
// displacement floored to pixels, a red disc with a highlight and a string, sky and ground.
#pragma once
#include <cstdint>
#include <vector>

namespace chronotope {

struct SynthBox {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool present = false;
    int cx() const { return (x0 + x1) / 2; }
    int cy() const { return (y0 + y1) / 2; }
    bool contains(int x, int y) const { return present && x >= x0 && x < x1 && y >= y0 && y < y1; }
};

struct Synthetic {
    int w = 320, h = 240, size = 24;
    int x0 = 100, y0 = 100;               // top-left at the event
    int64_t vx = 0, vy = 0;               // px/s (integer, as the manifest)
    int64_t t_e_ns = 500000000, window_ns = 16000000000, frame_period_ns = 16666667;
    bool distractor = false;              // a 2x2 grey pixel at (0,0) blinking every frame (irrelevant motion)
    bool plain = false;                   // a plain white square on black (the plain skin) instead of the balloon

    uint8_t target[3] = {220, 38, 38};    // the balloon's exact colour (the detector's key)
    bool visible(int64_t task_ns) const { return task_ns >= t_e_ns && task_ns < t_e_ns + window_ns; }
    SynthBox box_at(int64_t task_ns) const;
    void render(int64_t task_ns, std::vector<uint8_t>& rgb) const;
    void place_for_seed(uint64_t seed);   // deterministic start position inside the canvas, clear of the edges
};

}  // namespace chronotope
