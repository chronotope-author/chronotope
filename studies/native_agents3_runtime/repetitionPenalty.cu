#include "runtime/decoding/repetitionPenalty.h"
#include "runtime/decoding/repetitionPenaltyMath.h"
#include "runtime/state/decodingInferenceContext.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"

#include <cstdint>
#include <memory>

namespace trt_edgellm::rt
{
namespace
{
__global__ void markRepetitionToken(int8_t* seen, int32_t token, int32_t vocabSize)
{
    if (token >= 0 && token < vocabSize)
    {
        seen[token] = 1;
    }
}

__global__ void applyRepetitionPenaltyKernel(float* logits, int8_t const* seen, int32_t vocabSize, float penalty)
{
    int32_t const token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token < vocabSize)
    {
        logits[token] = repetitionAdjustedLogit(logits[token], seen[token] != 0, penalty);
    }
}
}

void applyRepetitionPenalty(Tensor& logits, DecodingInferenceContext& context, cudaStream_t stream)
{
    if (context.repetitionPenalty == 1.0f)
    {
        return;
    }
    auto const shape = logits.getShape();
    check::check(context.activeBatchSize == 1 && shape.getNumDims() == 2 && shape[0] == 1,
        "Repetition penalty requires the validated single-sequence vanilla decoder");
    check::check(logits.getDataType() == nvinfer1::DataType::kFLOAT,
        "Repetition penalty expects FP32 output logits");
    int32_t const vocabSize = static_cast<int32_t>(shape[1]);
    if (!context.repetitionSeenTokens)
    {
        // Includes template/special/media tokens, matching vLLM's full expanded
        // prompt-token mask. Retain host storage until the request stream drains.
        context.repetitionPromptMask.assign(vocabSize, 0);
        for (auto const token : context.rawBatchedInputIds.at(0))
        {
            if (token >= 0 && token < vocabSize)
            {
                context.repetitionPromptMask[token] = 1;
            }
        }
        context.repetitionSeenTokens = std::make_unique<Tensor>(
            std::initializer_list<int64_t>{vocabSize}, DeviceType::kGPU,
            nvinfer1::DataType::kINT8, "RepetitionPenalty::seenTokens");
        CUDA_CHECK(cudaMemcpyAsync(context.repetitionSeenTokens->rawPointer(), context.repetitionPromptMask.data(),
            vocabSize * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
    }
    auto* seen = context.repetitionSeenTokens->dataPointer<int8_t>();
    // Vanilla decoding appends exactly one generated token per sampling round.
    // Before prefill this only marks the already-present final prompt token.
    if (!context.tokenIds.at(0).empty())
    {
        markRepetitionToken<<<1, 1, 0, stream>>>(seen, context.tokenIds[0].back(), vocabSize);
        CUDA_CHECK(cudaGetLastError());
    }
    applyRepetitionPenaltyKernel<<<(vocabSize + 255) / 256, 256, 0, stream>>>(
        logits.dataPointer<float>(), seen, vocabSize, context.repetitionPenalty);
    CUDA_CHECK(cudaGetLastError());
}
}
