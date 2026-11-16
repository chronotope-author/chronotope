// vision.cu — perception kernels. Deterministic by construction: every reduction is either an exact
// integer sum or a total order (score, then smallest index), so a frame always yields the same numbers.
#include "vision.cuh"
#include <algorithm>
#include <climits>

namespace chronotope::vision {

namespace {

constexpr int BLK = 256;

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ float grey_of(uint8_t r, uint8_t g, uint8_t b) {
    // numpy: a[...,0]*0.299 + a[...,1]*0.587 + a[...,2]*0.114 on float32 arrays with weak Python-float scalars:
    // (f32(r)*0.299f + f32(g)*0.587f) + f32(b)*0.114f, each product and sum rounded to float32, no FMA.
    return __fadd_rn(__fadd_rn(__fmul_rn((float)r, 0.299f), __fmul_rn((float)g, 0.587f)), __fmul_rn((float)b, 0.114f));
}

__global__ void k_to_planes(const uint8_t* __restrict__ payload, uint32_t pixfmt, int w, int h,
                            uint8_t* rgb8, float* __restrict__ grey, float* rgbf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w * h;
    if (i >= n) return;
    uint8_t r, g, b;
    if (pixfmt == PIX_RGB24) {
        r = payload[3 * i]; g = payload[3 * i + 1]; b = payload[3 * i + 2];
    } else if (pixfmt == PIX_GREY8) {
        r = g = b = payload[i];
    } else {  // NV12: Y plane then interleaved UV; BT.601 limited range for rgb8, the Y value itself as grey
        int x = i % w, y = i / w;
        float yy = (float)payload[i] - 16.0f;
        int uv = n + (y / 2) * w + (x / 2) * 2;
        float d = (float)payload[uv] - 128.0f, e = (float)payload[uv + 1] - 128.0f;
        float fr = 1.164f * yy + 1.596f * e, fg = 1.164f * yy - 0.392f * d - 0.813f * e, fb = 1.164f * yy + 2.017f * d;
        r = (uint8_t)fminf(fmaxf(fr, 0.0f), 255.0f); g = (uint8_t)fminf(fmaxf(fg, 0.0f), 255.0f); b = (uint8_t)fminf(fmaxf(fb, 0.0f), 255.0f);
        grey[i] = (float)payload[i];
        if (rgb8 != payload) { rgb8[3 * i] = r; rgb8[3 * i + 1] = g; rgb8[3 * i + 2] = b; }
        if (rgbf) { rgbf[i] = (float)r; rgbf[n + i] = (float)g; rgbf[2 * n + i] = (float)b; }
        return;
    }
    if (rgb8 != payload) { rgb8[3 * i] = r; rgb8[3 * i + 1] = g; rgb8[3 * i + 2] = b; }
    grey[i] = grey_of(r, g, b);
    if (rgbf) { rgbf[i] = (float)r; rgbf[n + i] = (float)g; rgbf[2 * n + i] = (float)b; }
}

__device__ __forceinline__ uint64_t fnv1a(const uint8_t* p, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

__global__ void k_hash_chunks(const uint8_t* __restrict__ payload, size_t len, uint64_t* __restrict__ chunks, int nchunks) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nchunks) return;
    size_t off = (size_t)c * HASH_CHUNK;
    size_t n = (off + HASH_CHUNK <= len) ? HASH_CHUNK : (len - off);
    chunks[c] = fnv1a(payload + off, n, 14695981039346656037ULL);
}

__global__ void k_hash_combine(const uint64_t* __restrict__ chunks, int nchunks, uint64_t* out) {
    uint64_t h = 14695981039346656037ULL;
    for (int c = 0; c < nchunks; ++c) {
        uint64_t v = chunks[c];
        for (int k = 0; k < 8; ++k) { h ^= (v >> (8 * k)) & 0xFF; h *= 1099511628211ULL; }
    }
    *out = h;
}

__global__ void k_change(const float* __restrict__ a, const float* __restrict__ b, int n, unsigned long long* acc) {
    __shared__ unsigned long long sh[BLK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long v = 0;
    if (i < n) v = (unsigned long long)llrintf(fabsf(a[i] - b[i]) * 4096.0f);
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(acc, sh[0]);
}

__global__ void k_box_init(int* box4) { box4[0] = INT_MAX; box4[1] = INT_MAX; box4[2] = -1; box4[3] = -1; }

__global__ void k_detect(const uint8_t* __restrict__ rgb8, int w, int h, uint8_t r, uint8_t g, uint8_t b, int* box4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;
    if (rgb8[3 * i] == r && rgb8[3 * i + 1] == g && rgb8[3 * i + 2] == b) {
        int x = i % w, y = i / w;
        atomicMin(&box4[0], x); atomicMin(&box4[1], y); atomicMax(&box4[2], x); atomicMax(&box4[3], y);
    }
}

__global__ void k_cut_template(const float* __restrict__ grey_b, const float* __restrict__ rgbf_b, int w, int h, int radius,
                               int chans, AgentState* st, float* __restrict__ traw) {
    int side = 2 * radius + 1;
    int n = side * side * chans;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (st->proposal.status != PS_PROPOSED) return;   // no proposal this decision: nothing to cut
    int x = st->proposal.action.x, y = st->proposal.action.y;
    // cut_template: clamp the patch into the frame; remember where the click sits in it
    int x0 = min(max(0, x - radius), max(0, w - side));
    int y0 = min(max(0, y - radius), max(0, h - side));
    if (i == 0) {
        st->proposal.tpl_x0 = x0; st->proposal.tpl_y0 = y0;
        st->proposal.tpl_dx = x - x0; st->proposal.tpl_dy = y - y0;
        st->proposal.side = side; st->proposal.chans = chans;
    }
    int c = i / (side * side), rem = i % (side * side), j = rem / side, k = rem % side;
    const float* plane = (chans == 1) ? grey_b : (rgbf_b + (size_t)c * w * h);
    traw[i] = plane[(size_t)(y0 + j) * w + (x0 + k)];
}

__global__ void k_template_stats(const float* __restrict__ traw, int n, int side, int chans, TemplateStats* ts, float* __restrict__ tz) {
    __shared__ double sh[BLK];
    double acc = 0.0;
    for (int i = threadIdx.x; i < n; i += BLK) acc += (double)traw[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    float mean_f = (float)(sh[0] / n);          // float(np.mean(t)) then used as a weak scalar in float32 arithmetic
    __syncthreads();
    double sq = 0.0;
    for (int i = threadIdx.x; i < n; i += BLK) {
        float z = __fsub_rn(traw[i], mean_f);
        tz[i] = z;
        sq += (double)z * (double)z;
    }
    sh[threadIdx.x] = sq;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) {
        ts->mean_f = mean_f; ts->norm = sqrt(sh[0]); ts->n = n; ts->side = side; ts->chans = chans;
        ts->flat = (sh[0] == 0.0) ? 1 : 0;
    }
}

__global__ void k_ncc(const float* __restrict__ grey, const float* __restrict__ rgbf, int w, int h, int side, int chans,
                      const float* __restrict__ tz, const TemplateStats* ts, double* __restrict__ scores, int Wv, int Hv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Wv * Hv) return;
    if (ts->flat) { scores[idx] = -1.0; return; }
    int iy = idx / Wv, ix = idx % Wv;
    double num = 0.0, s = 0.0, ss = 0.0;
    for (int c = 0; c < chans; ++c) {
        const float* plane = (chans == 1) ? grey : (rgbf + (size_t)c * w * h);
        const float* t = tz + (size_t)c * side * side;
        for (int j = 0; j < side; ++j) {
            const float* row = plane + (size_t)(iy + j) * w + ix;
            const float* trow = t + j * side;
            for (int k = 0; k < side; ++k) {
                double g = (double)row[k];
                num += g * (double)trow[k];
                s += g; ss += g * g;
            }
        }
    }
    double n = (double)ts->n, tn = ts->norm;
    double var = ss - s * s / n;
    bool valid = var > 1e-6 * (tn * tn);
    scores[idx] = valid ? num / (sqrt(fmax(var, 0.0)) * tn) : -1.0;
}

__device__ __forceinline__ bool better(double s, int i, double bs, int bi) { return s > bs || (s == bs && i < bi); }

__global__ void k_argmax_partial(const double* __restrict__ scores, int n, ArgmaxPair* partial) {
    __shared__ double ss[BLK];
    __shared__ int si[BLK];
    double bs = -2.0; int bi = INT_MAX;
    for (int i = blockIdx.x * BLK + threadIdx.x; i < n; i += gridDim.x * BLK) {
        double v = scores[i];
        if (better(v, i, bs, bi)) { bs = v; bi = i; }
    }
    ss[threadIdx.x] = bs; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && better(ss[threadIdx.x + s], si[threadIdx.x + s], ss[threadIdx.x], si[threadIdx.x])) {
            ss[threadIdx.x] = ss[threadIdx.x + s]; si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { partial[blockIdx.x].score = ss[0]; partial[blockIdx.x].index = si[0]; }
}

__global__ void k_argmax_final(const ArgmaxPair* partial, int nb, ArgmaxPair* out) {
    __shared__ double ss[BLK];
    __shared__ int si[BLK];
    double bs = -2.0; int bi = INT_MAX;
    for (int i = threadIdx.x; i < nb; i += BLK)
        if (better(partial[i].score, partial[i].index, bs, bi)) { bs = partial[i].score; bi = partial[i].index; }
    ss[threadIdx.x] = bs; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && better(ss[threadIdx.x + s], si[threadIdx.x + s], ss[threadIdx.x], si[threadIdx.x])) {
            ss[threadIdx.x] = ss[threadIdx.x + s]; si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out->score = ss[0]; out->index = si[0]; }
}

__device__ __forceinline__ double clamp1(double v) { return fmin(1.0, fmax(-1.0, v)); }

__global__ void k_peak_partial(const double* __restrict__ scores, int Wv, int Hv, const ArgmaxPair* best, int side, PeakStats* partial) {
    __shared__ int sc[BLK];
    __shared__ double sm[BLK];
    int n = Wv * Hv;
    int iy = best->index / Wv, ix = best->index % Wv;
    double b = clamp1(best->score);
    int r0 = max(0, iy - side + 1), r1 = min(Hv, iy + side), c0 = max(0, ix - side + 1), c1 = min(Wv, ix + side);
    int cnt = 0; double m = -1.0;
    for (int i = blockIdx.x * BLK + threadIdx.x; i < n; i += gridDim.x * BLK) {
        int y = i / Wv, x = i % Wv;
        double v = scores[i];
        if (y >= r0 && y < r1 && x >= c0 && x < c1) { if (v >= b - PLATEAU_TOL) ++cnt; }
        else if (v > m) m = v;
    }
    sc[threadIdx.x] = cnt; sm[threadIdx.x] = m;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sc[threadIdx.x] += sc[threadIdx.x + s]; sm[threadIdx.x] = fmax(sm[threadIdx.x], sm[threadIdx.x + s]); }
        __syncthreads();
    }
    if (threadIdx.x == 0) { partial[blockIdx.x].plateau = sc[0]; partial[blockIdx.x].second = sm[0]; }
}

__global__ void k_peak_final(const PeakStats* partial, int nb, PeakStats* out) {
    __shared__ int sc[BLK];
    __shared__ double sm[BLK];
    int cnt = 0; double m = -1.0;
    for (int i = threadIdx.x; i < nb; i += BLK) { cnt += partial[i].plateau; m = fmax(m, partial[i].second); }
    sc[threadIdx.x] = cnt; sm[threadIdx.x] = m;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sc[threadIdx.x] += sc[threadIdx.x + s]; sm[threadIdx.x] = fmax(sm[threadIdx.x], sm[threadIdx.x + s]); }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out->plateau = sc[0]; out->second = clamp1(sm[0]); }
}

__global__ void k_patch_diff(const uint8_t* __restrict__ rgb8, int w, const Memory* mem, const uint8_t* __restrict__ patch, int* flag) {
    const PendingAction& p = mem->pending;
    if (!p.active) return;
    int pw = p.px1 - p.px0, ph = p.py1 - p.py0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pw * ph * 3) return;
    int pix = i / 3, ch = i % 3;
    int x = p.px0 + pix % pw, y = p.py0 + pix / pw;
    if (rgb8[((size_t)y * w + x) * 3 + ch] != patch[i]) atomicOr(flag, 1);
}

__global__ void k_cut_patch(const uint8_t* __restrict__ rgb8, int w, const AgentState* st, const Memory* mem, uint8_t* __restrict__ patch) {
    if (st->commit.status != CS_COMMITTED || !mem->pending.active) return;
    const PendingAction& p = mem->pending;
    int pw = p.px1 - p.px0, ph = p.py1 - p.py0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pw * ph * 3) return;
    int pix = i / 3, ch = i % 3;
    int x = p.px0 + pix % pw, y = p.py0 + pix / pw;
    patch[i] = rgb8[((size_t)y * w + x) * 3 + ch];
}

__global__ void k_stamp(unsigned long long* out) { *out = gtimer(); }

__global__ void k_upscale_nn(const uint8_t* __restrict__ src, int w, int h, int scale, uint8_t* __restrict__ dst) {
    int W = w * scale, H = h * scale;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    int x = i % W, y = i / W;
    const uint8_t* p = src + ((size_t)(y / scale) * w + (x / scale)) * 3;
    dst[3 * (size_t)i] = p[0]; dst[3 * (size_t)i + 1] = p[1]; dst[3 * (size_t)i + 2] = p[2];
}

inline int blocks(int n) { return (n + BLK - 1) / BLK; }

}  // namespace

void to_planes(const uint8_t* payload, uint32_t pixfmt, int w, int h, uint8_t* rgb8, float* grey, float* rgbf, cudaStream_t s) {
    k_to_planes<<<blocks(w * h), BLK, 0, s>>>(payload, pixfmt, w, h, rgb8, grey, rgbf);
}

int hash_chunks(size_t len) { return (int)((len + HASH_CHUNK - 1) / HASH_CHUNK); }

void hash_payload(const uint8_t* payload, size_t len, uint64_t* chunks, uint64_t* out, cudaStream_t s) {
    int nc = hash_chunks(len);
    k_hash_chunks<<<blocks(nc), BLK, 0, s>>>(payload, len, chunks, nc);
    k_hash_combine<<<1, 1, 0, s>>>(chunks, nc, out);
}

void change_score(const float* cur, const float* prev, int n, unsigned long long* acc, cudaStream_t s) {
    k_change<<<blocks(n), BLK, 0, s>>>(cur, prev, n, acc);
}

void zero_box(int* box4, cudaStream_t s) { k_box_init<<<1, 1, 0, s>>>(box4); }

void detect_color(const uint8_t* rgb8, int w, int h, uint8_t r, uint8_t g, uint8_t b, int* box4, cudaStream_t s) {
    k_box_init<<<1, 1, 0, s>>>(box4);
    k_detect<<<blocks(w * h), BLK, 0, s>>>(rgb8, w, h, r, g, b, box4);
}

void cut_template(const float* grey_b, const float* rgbf_b, int w, int h, int radius, int chans, AgentState* st, float* traw, cudaStream_t s) {
    int side = 2 * radius + 1;
    k_cut_template<<<blocks(side * side * chans), BLK, 0, s>>>(grey_b, rgbf_b, w, h, radius, chans, st, traw);
}

void template_stats(const float* traw, int n, int side, int chans, TemplateStats* ts, float* tz, cudaStream_t s) {
    k_template_stats<<<1, BLK, 0, s>>>(traw, n, side, chans, ts, tz);
}

void ncc(const float* grey, const float* rgbf, int w, int h, int side, int chans, const float* tz, const TemplateStats* ts,
         double* scores, cudaStream_t s) {
    int Wv = w - side + 1, Hv = h - side + 1;
    k_ncc<<<blocks(Wv * Hv), BLK, 0, s>>>(grey, rgbf, w, h, side, chans, tz, ts, scores, Wv, Hv);
}

int argmax_blocks(int n) { return std::min(1024, blocks(n)); }

void argmax(const double* scores, int n, ArgmaxPair* partial, ArgmaxPair* out, cudaStream_t s) {
    int nb = argmax_blocks(n);
    k_argmax_partial<<<nb, BLK, 0, s>>>(scores, n, partial);
    k_argmax_final<<<1, BLK, 0, s>>>(partial, nb, out);
}

void peak_stats(const double* scores, int Wv, int Hv, const ArgmaxPair* best, int side, PeakStats* partial, PeakStats* out, cudaStream_t s) {
    int nb = argmax_blocks(Wv * Hv);
    k_peak_partial<<<nb, BLK, 0, s>>>(scores, Wv, Hv, best, side, partial);
    k_peak_final<<<1, BLK, 0, s>>>(partial, nb, out);
}

void patch_diff(const uint8_t* rgb8, int w, int h, const Memory* mem, const uint8_t* patch, int* flag, cudaStream_t s) {
    (void)h;
    // the patch has at most (2r+1)^2*3 bytes; launch generously for the largest pending radius (the kernel bounds itself)
    k_patch_diff<<<blocks(3 * 129 * 129), BLK, 0, s>>>(rgb8, w, mem, patch, flag);
}

void cut_patch(const uint8_t* rgb8, int w, int h, const AgentState* st, const Memory* mem, uint8_t* patch, cudaStream_t s) {
    (void)h;
    k_cut_patch<<<blocks(3 * 129 * 129), BLK, 0, s>>>(rgb8, w, st, mem, patch);
}

void stamp(unsigned long long* out, cudaStream_t s) { k_stamp<<<1, 1, 0, s>>>(out); }

void upscale_nn(const uint8_t* src, int w, int h, int scale, uint8_t* dst, cudaStream_t s) {
    k_upscale_nn<<<blocks(w * h * scale * scale), BLK, 0, s>>>(src, w, h, scale, dst);
}

}  // namespace chronotope::vision
