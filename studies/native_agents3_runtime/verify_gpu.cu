// Synthetic CUDA processor test. Does not load a model or execute any task.
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "runtime/decoding/repetitionPenalty.h"
#include "runtime/state/decodingInferenceContext.h"

#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    nlohmann::json results = nlohmann::json::array();
    std::vector<float> original(257);
    for (size_t i = 0; i < original.size(); ++i) original[i] = (static_cast<int>(i % 23) - 11) * 0.75f;
    original[0] = 0.0f;
    original[1] = -0.0f;
    original[2] = 8.0f;
    original[3] = -8.0f;
    original[4] = 0.001f;
    original[5] = -0.001f;
    Tensor logits({1, static_cast<int64_t>(original.size())}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    DecodingInferenceContext context;
    for (int request = 0; request < 3; ++request)
    {
        context.initialize(1, 4096, std::nullopt, OptionalInputTensors{}, "", stream);
        context.repetitionPenalty = request == 2 ? 1.0f : 1.05f;
        std::vector<int32_t> prompt = request == 0 ? std::vector<int32_t>{0, 1, 2, 3, 3, 4, 5, 256}
                                                 : std::vector<int32_t>{7, 8, 8, 17};
        context.rawBatchedInputIds = {prompt};
        context.tokenIds = {prompt};
        std::vector<int32_t> outputs;
        std::vector<int32_t> next = {6, 6, 9, 3, 255};
        for (size_t step = 0; step <= next.size(); ++step)
        {
            if (step)
            {
                outputs.push_back(next[step-1]);
                context.tokenIds[0].push_back(next[step-1]);
            }
            CUDA_CHECK(cudaMemcpyAsync(logits.rawPointer(), original.data(), original.size()*sizeof(float),
                cudaMemcpyHostToDevice, stream));
            applyRepetitionPenalty(logits, context, stream);
            std::vector<float> actual(original.size());
            CUDA_CHECK(cudaMemcpyAsync(actual.data(), logits.rawPointer(), actual.size()*sizeof(float),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            results.push_back({{"request", request}, {"step", step}, {"penalty", context.repetitionPenalty},
                {"prompt", prompt}, {"generated", outputs}, {"input_logits", original}, {"actual_logits", actual}});
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::ofstream(argv[1]) << results.dump(2) << '\n';
    std::cout << "Synthetic native CUDA penalty cases=" << results.size() << "; model inference=0; task trials=0\n";
}
