// capture.h — /dev/video0 -> FrameView, nothing else (design §4). V4L2 MMAP streaming with the kernel's
// CLOCK_MONOTONIC buffer timestamp and the dequeue time on the same clock (chronotope.phy.capture's
// contract item 1). capture_latest() drains the queue and returns the newest frame: freshness over
// completeness (the design's "812, not 810 811 812").
#pragma once
#include <cstddef>
#include <cstdint>

namespace chronotope {

struct FrameView {
    const uint8_t* data = nullptr;
    size_t len = 0;
    uint32_t width = 0, height = 0;
    uint32_t seq = 0;
    int64_t ts_kernel_ns = 0;     // driver buffer timestamp (CLOCK_MONOTONIC when ts_monotonic)
    int64_t t_dq_ns = 0;          // CLOCK_MONOTONIC right after dequeue
    bool ts_monotonic = false;
    uint32_t pixfmt = 0;          // CHRONOTOPE_PIX_*
    int index = -1;               // V4L2 buffer index (for release)
};

struct Capture;

// fourcc: "NV12" (default), "RGB3" (RGB24) or "GREY". MJPG is not decoded here (the Python instrument does that).
Capture* capture_open(const char* device, uint32_t width, uint32_t height, const char* fourcc, double fps, int n_buffers,
                      char* err, size_t errlen);
bool capture_next(Capture* c, FrameView* out, double timeout_s);
bool capture_latest(Capture* c, FrameView* out, double timeout_s);
void capture_release(Capture* c, const FrameView& f);
void capture_close(Capture* c);
uint64_t capture_dropped(const Capture* c);      // frames skipped by capture_latest
uint64_t capture_count(const Capture* c);
const char* capture_error(const Capture* c);

}  // namespace chronotope
