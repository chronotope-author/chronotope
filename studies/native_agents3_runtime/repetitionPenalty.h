#pragma once

#include <cuda_runtime_api.h>

namespace trt_edgellm::rt
{
class Tensor;
struct DecodingInferenceContext;

// Request-local GPU membership mask; no device-to-host logits transfer.
void applyRepetitionPenalty(Tensor& logits, DecodingInferenceContext& context, cudaStream_t stream);
}
