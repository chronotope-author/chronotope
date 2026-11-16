#pragma once

// Shared by the CUDA processor and CPU semantic tests. A token is penalized
// once if it occurs in either the entire prompt or generated output.
#ifdef __CUDACC__
#define CHRONOTOPE_HOST_DEVICE __host__ __device__
#else
#define CHRONOTOPE_HOST_DEVICE
#endif

namespace trt_edgellm::rt
{
CHRONOTOPE_HOST_DEVICE inline float repetitionAdjustedLogit(float logit, bool seen, float penalty)
{
    if (!seen || penalty == 1.0f)
    {
        return logit;
    }
    return logit > 0.0f ? logit / penalty : logit * penalty;
}
}

#undef CHRONOTOPE_HOST_DEVICE
