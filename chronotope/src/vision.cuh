// vision.cuh — pixels -> representation. Nothing here decides what to do (design §5).
#pragma once
#include <cuda_runtime.h>
#include "state.cuh"

namespace chronotope::vision {

// payload (RGB24 / GREY8 / NV12) -> rgb8 (w*h*3), grey (float, numpy-identical 0.299R+0.587G+0.114B in
// float32; NV12: the Y plane itself), rgbf (3 float planes, optional). rgb8 may alias payload for RGB24.
void to_planes(const uint8_t* payload, uint32_t pixfmt, int w, int h, uint8_t* rgb8, float* grey, float* rgbf, cudaStream_t s);
// Deterministic 64-bit payload hash: FNV-1a per 4096-byte chunk in parallel, chunks combined in order.
void hash_payload(const uint8_t* payload, size_t len, uint64_t* chunks, uint64_t* out, cudaStream_t s);
int hash_chunks(size_t len);
// sum |grey_cur - grey_prev| * 4096 as an exact integer (acc must be zeroed first).
void change_score(const float* cur, const float* prev, int n, unsigned long long* acc, cudaStream_t s);
// bounding box of all pixels exactly equal to (r, g, b): box4 = {xmin, ymin, xmax, ymax}, xmax < 0 = absent.
void detect_color(const uint8_t* rgb8, int w, int h, uint8_t r, uint8_t g, uint8_t b, int* box4, cudaStream_t s);
// cut the proposal's template from the frame at B (grey or 3 float planes) into traw; records x0,y0,dx,dy.
void cut_template(const float* grey_b, const float* rgbf_b, int w, int h, int radius, int chans, AgentState* st, float* traw, cudaStream_t s);
// mean (float32 as numpy) and norm of the zero-mean template; writes tz = traw - mean.
void template_stats(const float* traw, int n, int side, int chans, TemplateStats* ts, float* tz, cudaStream_t s);
// direct normalized cross-correlation at every valid offset (double), tracker.py's variance guard.
void ncc(const float* grey, const float* rgbf, int w, int h, int side, int chans, const float* tz, const TemplateStats* ts,
         double* scores, cudaStream_t s);
int argmax_blocks(int n);
// first maximum in row-major order (np.argmax semantics), deterministic.
void argmax(const double* scores, int n, ArgmaxPair* partial, ArgmaxPair* out, cudaStream_t s);
// plateau count inside the winner's footprint and the best score outside it.
void peak_stats(const double* scores, int Wv, int Hv, const ArgmaxPair* best, int side, PeakStats* partial, PeakStats* out, cudaStream_t s);
// pending record: does the rgb8 patch around the pending click differ from the stored one? flag must be zeroed.
void patch_diff(const uint8_t* rgb8, int w, int h, const Memory* mem, const uint8_t* patch, int* flag, cudaStream_t s);
// after a commit: store the rgb8 patch around the committed click (no-op unless the commit succeeded).
void cut_patch(const uint8_t* rgb8, int w, int h, const AgentState* st, const Memory* mem, uint8_t* patch, cudaStream_t s);
// %globaltimer, for the device/host clock calibration
void stamp(unsigned long long* out, cudaStream_t s);
// nearest-neighbour upscale of an RGB24 frame by an integer factor (the paper's image_scale), on the device
void upscale_nn(const uint8_t* src, int w, int h, int scale, uint8_t* dst, cudaStream_t s);
void zero_box(int* box4, cudaStream_t s);

}  // namespace chronotope::vision
