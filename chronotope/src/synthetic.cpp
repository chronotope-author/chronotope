#include "synthetic.h"
#include <algorithm>
#include <cmath>

namespace chronotope {

namespace {
int64_t floordiv(int64_t a, int64_t b) { int64_t q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }

void fill(std::vector<uint8_t>& buf, int w, int h, int x0, int y0, int x1, int y1, const uint8_t c[3]) {
    x0 = std::max(0, x0); y0 = std::max(0, y0); x1 = std::min(w, x1); y1 = std::min(h, y1);
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) { size_t o = ((size_t)y * w + x) * 3; buf[o] = c[0]; buf[o + 1] = c[1]; buf[o + 2] = c[2]; }
}

void disc(std::vector<uint8_t>& buf, int w, int h, double cx, double cy, double r, const uint8_t c[3]) {
    for (int yy = std::max(0, (int)(cy - r)); yy < std::min(h, (int)(cy + r) + 1); ++yy)
        for (int xx = std::max(0, (int)(cx - r)); xx < std::min(w, (int)(cx + r) + 1); ++xx)
            if ((xx + 0.5 - cx) * (xx + 0.5 - cx) + (yy + 0.5 - cy) * (yy + 0.5 - cy) <= r * r) {
                size_t o = ((size_t)yy * w + xx) * 3; buf[o] = c[0]; buf[o + 1] = c[1]; buf[o + 2] = c[2];
            }
}
}  // namespace

SynthBox Synthetic::box_at(int64_t task_ns) const {
    SynthBox b;
    if (!visible(task_ns)) return b;
    int64_t dt = task_ns - t_e_ns;
    int dx = (int)floordiv(vx * dt, 1000000000LL), dy = (int)floordiv(vy * dt, 1000000000LL);
    b.x0 = x0 + dx; b.y0 = y0 + dy; b.x1 = b.x0 + size; b.y1 = b.y0 + size; b.present = true;
    return b;
}

void Synthetic::render(int64_t task_ns, std::vector<uint8_t>& rgb) const {
    rgb.assign((size_t)w * h * 3, 0);
    if (!plain) {
        const uint8_t sky[3] = {186, 222, 246}, ground[3] = {134, 190, 120}, cloud[3] = {248, 250, 252};
        fill(rgb, w, h, 0, 0, w, h, sky);
        disc(rgb, w, h, w * 0.2, h * 0.22, h * 0.09, cloud); disc(rgb, w, h, w * 0.3, h * 0.2, h * 0.07, cloud);
        disc(rgb, w, h, w * 0.75, h * 0.3, h * 0.08, cloud); disc(rgb, w, h, w * 0.84, h * 0.33, h * 0.06, cloud);
        fill(rgb, w, h, 0, h - h / 10, w, h, ground);
    }
    if (distractor && (task_ns / frame_period_ns) % 2 == 0) { const uint8_t g[3] = {128, 128, 128}; fill(rgb, w, h, 0, 0, 2, 2, g); }
    SynthBox b = box_at(task_ns);
    if (!b.present) return;
    if (plain) { fill(rgb, w, h, b.x0, b.y0, b.x1, b.y1, target); return; }
    double r = size / 2.0, cx = b.x0 + r, cy = b.y0 + r;
    const uint8_t str[3] = {90, 90, 90}, hi[3] = {252, 165, 165};
    fill(rgb, w, h, (int)cx, b.y1, (int)cx + 1, b.y1 + size, str);
    disc(rgb, w, h, cx, cy, r, target);
    disc(rgb, w, h, cx - r * 0.35, cy - r * 0.35, r * 0.22, hi);
}

void Synthetic::place_for_seed(uint64_t seed) {
    uint64_t s = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    int margin = size + 8;
    int span_x = std::max(1, w - 2 * margin - size), span_y = std::max(1, h - 2 * margin - size - h / 10);
    x0 = margin + (int)((s >> 17) % (uint64_t)span_x);
    y0 = margin + (int)((s >> 41) % (uint64_t)span_y);
}

}  // namespace chronotope
