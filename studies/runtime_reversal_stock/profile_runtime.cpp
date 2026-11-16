// Measurement-only client of unmodified NVIDIA TensorRT-Edge-LLM public APIs.
// It replays one stored request and never interprets or applies an action.
#include "common/cudaUtils.h"
#include "common/trtUtils.h"
#include "requestFileParser.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/streaming.h"
#include <cuda_profiler_api.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace trt_edgellm;
using Json = nlohmann::json;

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cerr << "Usage: stock_profile_runtime ENGINE_DIR MULTIMODAL_DIR INPUT_JSON OUTPUT_JSON\n";
        return 2;
    }
    Json audit = {{"diagnostic", "stock_upstream_model_request"}, {"application_episodes", 0},
        {"qualification_trials", 0}, {"model_requests_executed", 0}};
    int callbacks = 0;
    int startStatus = -1;
    int stopStatus = -1;
    bool started = false;
    bool stopped = false;
    try
    {
        auto pluginHandles = loadEdgellmPluginLib();
        if (!pluginHandles)
            throw std::runtime_error("Cannot load the explicitly selected pristine plugin");
        auto parsed = exampleUtils::parseRequestFile(argv[3]);
        if (parsed.second.size() != 1 || parsed.second[0].requests.size() != 1)
            throw std::runtime_error("Exactly one stored request is required");
        auto& request = parsed.second[0];
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        {
            rt::LLMInferenceRuntime runtime(argv[1], argv[2], parsed.first, stream);
            audit["decoding_cuda_graph_captured"] = runtime.captureDecodingCUDAGraph(stream);
            request.onTokenGenerated = [&](rt::TokenCallbackInfo const&) {
                ++callbacks;
                if (callbacks == 5)
                {
                    startStatus = static_cast<int>(cudaProfilerStart());
                    started = startStatus == static_cast<int>(cudaSuccess);
                }
                if (callbacks == 37 && started)
                {
                    stopStatus = static_cast<int>(cudaProfilerStop());
                    stopped = true;
                }
            };
            rt::LLMGenerationResponse response;
            auto const begin = std::chrono::steady_clock::now();
            audit["model_requests_executed"] = 1;
            bool const ok = runtime.handleRequest(request, response, stream);
            audit["handle_request_seconds"] = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin).count();
            if (started && !stopped)
            {
                stopStatus = static_cast<int>(cudaProfilerStop());
                stopped = true;
            }
            audit["ok"] = ok;
            audit["formatted_requests"] = Json::array();
            for (auto const& formatted : request.formattedRequests)
                audit["formatted_requests"].push_back({
                    {"system", formatted.formattedSystemPrompt},
                    {"complete", formatted.formattedCompleteRequest}});
            audit["input_token_ids"] = runtime.getBaseModelInputTokenIds();
            audit["input_token_counts"] = response.inputTokenCounts;
            audit["output_token_ids"] = response.outputIds;
            audit["output_texts"] = response.outputTexts;
            audit["finish_reasons"] = Json::array();
            for (auto const reason : response.finishReasons)
                audit["finish_reasons"].push_back(rt::finishReasonName(reason));
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    catch (std::exception const& error)
    {
        if (started && !stopped)
        {
            stopStatus = static_cast<int>(cudaProfilerStop());
            stopped = true;
        }
        audit["ok"] = false;
        audit["infrastructure_error"] = error.what();
    }
    audit["capture"] = {{"after_output_callback", 5}, {"through_output_callback", 37},
        {"requested_decode_steps", 32}, {"observed_output_callbacks", callbacks},
        {"started", started}, {"stopped", stopped}, {"cuda_start_status", startStatus},
        {"cuda_stop_status", stopStatus}, {"full_generation_preserved", true}};
    std::ofstream output(argv[4]);
    output << audit.dump(2) << '\n';
    return output.good() && audit.value("ok", false) ? 0 : 1;
}
