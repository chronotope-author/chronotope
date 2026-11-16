#include "capture.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

#include "chronotope_api.h"

namespace chronotope {

struct Capture {
    int fd = -1;
    std::string device, error;
    uint32_t width = 0, height = 0, fourcc = 0, pixfmt = 0;
    std::vector<void*> bufs;
    std::vector<size_t> lens;
    bool streaming = false;
    uint64_t dropped = 0, count = 0;
};

namespace {
int64_t now_ns() { timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec; }

int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

bool dequeue_one(Capture* c, v4l2_buffer* buf) {
    std::memset(buf, 0, sizeof(*buf));
    buf->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf->memory = V4L2_MEMORY_MMAP;
    return xioctl(c->fd, VIDIOC_DQBUF, buf) == 0;
}

void fill_view(Capture* c, const v4l2_buffer& buf, int64_t t_dq, FrameView* out) {
    out->data = static_cast<const uint8_t*>(c->bufs[buf.index]);
    out->len = buf.bytesused;
    out->width = c->width; out->height = c->height;
    out->seq = buf.sequence;
    out->ts_kernel_ns = (int64_t)buf.timestamp.tv_sec * 1000000000LL + (int64_t)buf.timestamp.tv_usec * 1000LL;
    out->t_dq_ns = t_dq;
    out->ts_monotonic = (buf.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
    out->pixfmt = c->pixfmt;
    out->index = (int)buf.index;
}

bool requeue(Capture* c, int index) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = (uint32_t)index;
    return xioctl(c->fd, VIDIOC_QBUF, &b) == 0;
}
}  // namespace

Capture* capture_open(const char* device, uint32_t width, uint32_t height, const char* fourcc, double fps, int n_buffers,
                      char* err, size_t errlen) {
    auto fail = [&](const std::string& m) { if (err && errlen) std::snprintf(err, errlen, "%s: %s (%s)", device, m.c_str(), std::strerror(errno)); return (Capture*)nullptr; };
    std::string fc = fourcc ? fourcc : "NV12";
    uint32_t code; uint32_t pix;
    if (fc == "NV12") { code = V4L2_PIX_FMT_NV12; pix = CHRONOTOPE_PIX_NV12; }
    else if (fc == "RGB3") { code = V4L2_PIX_FMT_RGB24; pix = CHRONOTOPE_PIX_RGB24; }
    else if (fc == "GREY") { code = V4L2_PIX_FMT_GREY; pix = CHRONOTOPE_PIX_GREY8; }
    else { if (err && errlen) std::snprintf(err, errlen, "unsupported fourcc %s (NV12, RGB3, GREY)", fc.c_str()); return nullptr; }
    auto* c = new Capture();
    c->device = device;
    c->fd = open(device, O_RDWR | O_NONBLOCK);
    if (c->fd < 0) { delete c; return fail("open"); }
    v4l2_format f{};
    f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    f.fmt.pix.width = width; f.fmt.pix.height = height; f.fmt.pix.pixelformat = code; f.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(c->fd, VIDIOC_S_FMT, &f) < 0) { close(c->fd); delete c; return fail("VIDIOC_S_FMT"); }
    if (f.fmt.pix.pixelformat != code) { close(c->fd); delete c; if (err && errlen) std::snprintf(err, errlen, "%s: driver substituted another pixel format for %s", device, fc.c_str()); return nullptr; }
    c->width = f.fmt.pix.width; c->height = f.fmt.pix.height; c->fourcc = code; c->pixfmt = pix;
    if (fps > 0) {
        v4l2_streamparm p{};
        p.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        p.parm.capture.timeperframe.numerator = 1; p.parm.capture.timeperframe.denominator = (uint32_t)(fps + 0.5);
        xioctl(c->fd, VIDIOC_S_PARM, &p);   // best effort: the driver may keep its own rate
    }
    v4l2_requestbuffers req{};
    req.count = (uint32_t)(n_buffers > 0 ? n_buffers : 4); req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(c->fd, VIDIOC_REQBUFS, &req) < 0) { close(c->fd); delete c; return fail("VIDIOC_REQBUFS"); }
    for (uint32_t i = 0; i < req.count; ++i) {
        v4l2_buffer b{};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = i;
        if (xioctl(c->fd, VIDIOC_QUERYBUF, &b) < 0) { capture_close(c); return fail("VIDIOC_QUERYBUF"); }
        void* p = mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, c->fd, b.m.offset);
        if (p == MAP_FAILED) { capture_close(c); return fail("mmap"); }
        c->bufs.push_back(p); c->lens.push_back(b.length);
        if (xioctl(c->fd, VIDIOC_QBUF, &b) < 0) { capture_close(c); return fail("VIDIOC_QBUF"); }
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(c->fd, VIDIOC_STREAMON, &type) < 0) { capture_close(c); return fail("VIDIOC_STREAMON"); }
    c->streaming = true;
    return c;
}

bool capture_next(Capture* c, FrameView* out, double timeout_s) {
    if (!c || c->fd < 0) return false;
    for (;;) {
        fd_set fds; FD_ZERO(&fds); FD_SET(c->fd, &fds);
        timeval tv{}; tv.tv_sec = (long)timeout_s; tv.tv_usec = (long)((timeout_s - (long)timeout_s) * 1e6);
        int r = select(c->fd + 1, &fds, nullptr, nullptr, &tv);
        if (r == 0) { c->error = "no frame within the timeout: HDMI source active?"; return false; }
        if (r < 0) { if (errno == EINTR) continue; c->error = std::strerror(errno); return false; }
        v4l2_buffer buf{};
        if (!dequeue_one(c, &buf)) { if (errno == EAGAIN) continue; c->error = std::strerror(errno); return false; }
        int64_t t = now_ns();
        fill_view(c, buf, t, out);
        c->count++;
        return true;
    }
}

bool capture_latest(Capture* c, FrameView* out, double timeout_s) {
    if (!capture_next(c, out, timeout_s)) return false;
    // drain: anything already queued behind this frame is older than what the device will hand us next
    for (;;) {
        v4l2_buffer buf{};
        if (!dequeue_one(c, &buf)) break;   // EAGAIN: nothing more
        int64_t t = now_ns();
        requeue(c, out->index);
        c->dropped++;
        fill_view(c, buf, t, out);
        c->count++;
    }
    return true;
}

void capture_release(Capture* c, const FrameView& f) {
    if (c && f.index >= 0) requeue(c, f.index);
}

void capture_close(Capture* c) {
    if (!c) return;
    if (c->streaming) { v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE; xioctl(c->fd, VIDIOC_STREAMOFF, &type); }
    for (size_t i = 0; i < c->bufs.size(); ++i) munmap(c->bufs[i], c->lens[i]);
    if (c->fd >= 0) close(c->fd);
    delete c;
}

uint64_t capture_dropped(const Capture* c) { return c ? c->dropped : 0; }
uint64_t capture_count(const Capture* c) { return c ? c->count : 0; }
const char* capture_error(const Capture* c) { return c ? c->error.c_str() : "null"; }

}  // namespace chronotope
