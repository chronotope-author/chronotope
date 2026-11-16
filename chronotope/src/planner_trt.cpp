#include "planner_trt.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <thread>

#ifdef CHRONOTOPE_WITH_EDGELLM
#include <cuda_runtime.h>
#include "common/tensor.h"
#include "runtime/imageUtils.h"
#include "common/trtUtils.h"      // loadEdgellmPluginLib: the plugin library must be loaded before the engine (K4a, 2026-08-27)
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"
#include "agents3_prompt.h"
namespace rt = trt_edgellm::rt;   // the runtime's namespace
#endif

namespace chronotope {

namespace {
int64_t now_ns() { timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec; }
}

struct TrtPlanner::Impl {
    TrtPlannerConfig cfg;
    std::string system;
    std::thread th;
    std::atomic<bool> busy{false}, done{false};
    VlmAnswer result;
    uint64_t steps = 0;
    double prefill_ms = 0.0, decode_ms = 0.0;
#ifdef CHRONOTOPE_WITH_EDGELLM
    cudaStream_t stream = nullptr;
    std::unique_ptr<void, trt_edgellm::DlDeleter> plugin;              // K4a: keeps libNvInfer_edgellm_plugin.so loaded for the runtime's lifetime
    std::unique_ptr<rt::LLMInferenceRuntime> runtime;
    agents3_prompt::Json prompt_contract;
    agents3_prompt::StaticPrompt prompt;
    std::vector<int32_t> prompt_input_ids;
    std::ofstream prompt_audit;
#endif
};

bool TrtPlanner::compiled() {
#ifdef CHRONOTOPE_WITH_EDGELLM
    return true;
#else
    return false;
#endif
}

TrtPlanner::TrtPlanner(TrtPlannerConfig cfg, void* cuda_stream) : impl_(new Impl()) {
    impl_->cfg = std::move(cfg);
    impl_->system = vlm_system_prompt(impl_->cfg.prompt_version, impl_->cfg.w, impl_->cfg.h, impl_->cfg.image_scale);
#ifdef CHRONOTOPE_WITH_EDGELLM
    impl_->stream = static_cast<cudaStream_t>(cuda_stream);
    if (impl_->cfg.engine_dir.empty() || impl_->cfg.multimodal_engine_dir.empty())
        throw std::runtime_error("TrtPlanner: engine_dir and multimodal_engine_dir are required");
    if (!impl_->cfg.prompt_path.empty()) {
        std::ifstream input(impl_->cfg.prompt_path);
        if (!input) throw std::runtime_error("Cannot read fixed Agent S3 prompt");
        input >> impl_->prompt_contract;
        impl_->prompt = agents3_prompt::prepare(impl_->prompt_contract);
        if (impl_->cfg.w * impl_->cfg.image_scale != 640 || impl_->cfg.h * impl_->cfg.image_scale != 480 ||
            impl_->cfg.max_tokens != 4096 || impl_->cfg.temperature != 0.0 || impl_->cfg.enable_thinking)
            throw std::runtime_error("Agent S3 prompt requires matched geometry and generation settings");
        impl_->system = impl_->prompt_contract.at("planner_system");
        trt_edgellm::tokenizer::Tokenizer tokenizer;
        if (!tokenizer.loadFromHF(impl_->cfg.engine_dir)) throw std::runtime_error("Cannot load stock tokenizer");
        std::string expanded = impl_->prompt.formatted_text, pads;
        for (int i = 0; i < 300; ++i) pads += "<|image_pad|>";
        const auto marker = expanded.find("<|image_pad|>");
        if (marker == std::string::npos || expanded.find("<|image_pad|>", marker + 13) != std::string::npos)
            throw std::runtime_error("Fixed prompt must contain exactly one image");
        expanded.replace(marker, 13, pads);
        impl_->prompt_input_ids = tokenizer.encode(expanded, false);
        if (impl_->prompt_input_ids.size() + 4096 > 16384) throw std::runtime_error("Prompt exceeds engine capacity");
        if (impl_->cfg.prompt_audit_path.empty()) throw std::runtime_error("Fixed prompt requires a request audit path");
        impl_->prompt_audit.open(impl_->cfg.prompt_audit_path, std::ios::app);
        if (!impl_->prompt_audit) throw std::runtime_error("Cannot open fixed prompt audit");
    }
    impl_->plugin = trt_edgellm::loadEdgellmPluginLib();   // K4a: without it the engine's gated_delta_net creator is unknown and deserialization fails
    if (!impl_->plugin) throw std::runtime_error("TrtPlanner: could not load the Edge-LLM plugin library (EDGELLM_PLUGIN_PATH)");
    std::unordered_map<std::string, std::string> no_lora;
    impl_->runtime = std::make_unique<rt::LLMInferenceRuntime>(impl_->cfg.engine_dir, impl_->cfg.multimodal_engine_dir, no_lora, impl_->stream);
    if (impl_->cfg.capture_cuda_graph && !impl_->runtime->captureDecodingCUDAGraph(impl_->stream))
        throw std::runtime_error("TrtPlanner: CUDA graph capture failed");
#else
    (void)cuda_stream;
    throw std::runtime_error("chronotope was built without TensorRT Edge-LLM (configure with -DKGUA_EDGELLM_BUILD=<edge-llm build dir>)");
#endif
}

TrtPlanner::~TrtPlanner() {
    if (impl_ && impl_->th.joinable()) impl_->th.join();
}

const std::string& TrtPlanner::system_prompt() const { return impl_->system; }
bool TrtPlanner::in_flight() const { return impl_->busy.load(); }
bool TrtPlanner::done() const { return impl_->done.load(); }
uint64_t TrtPlanner::host_steps() const { return impl_->steps; }
double TrtPlanner::prefill_ms() const { return impl_->prefill_ms; }
double TrtPlanner::decode_ms() const { return impl_->decode_ms; }

std::string TrtPlanner::runtime_info() const {
    return "tensorrt_edgellm engine_dir=" + impl_->cfg.engine_dir + " multimodal_engine_dir=" + impl_->cfg.multimodal_engine_dir +
           " prompt_version=" + std::to_string(impl_->cfg.prompt_version) + " image_scale=" + std::to_string(impl_->cfg.image_scale) +
           " max_tokens=" + std::to_string(impl_->cfg.max_tokens) + " cuda_graph=" + (impl_->cfg.capture_cuda_graph ? "1" : "0") +
           (impl_->cfg.prompt_path.empty() ? "" : " agents3_prompt=" + impl_->cfg.prompt_path + " history=none reflection=0 retries=0");
}

VlmAnswer TrtPlanner::invoke(DeviceFrame frame) {
    VlmAnswer a;
    a.t_start_ns = now_ns();
#ifdef CHRONOTOPE_WITH_EDGELLM
    const bool fixed_prompt = !impl_->cfg.prompt_path.empty();
    agents3_prompt::Json audit;
    try {
        if (!frame.rgb || frame.w <= 0 || frame.h <= 0) throw std::runtime_error("empty device frame");
        rt::LLMGenerationRequest req;
        req.temperature = (float)impl_->cfg.temperature;
        req.topP = 1.0f;
        req.topK = 1;
        req.maxGenerateLength = impl_->cfg.max_tokens;
        req.enableThinking = impl_->cfg.enable_thinking;
        rt::LLMGenerationRequest::Request r;
        rt::Message sys; sys.role = "system"; sys.contents.push_back({"text", impl_->system});
        rt::Message usr; usr.role = "user";
        usr.contents.push_back({"image", ""});
        usr.contents.push_back({"text", "Screenshot attached. Click the target's center or wait."});
        std::vector<uint8_t> host_frame;
        if (fixed_prompt) {
            audit = {{"schema", "agents3-native-prompt-audit-1"}, {"prompt_sha256", impl_->prompt_contract.at("prompt_sha256")},
                {"request_kind", "planner"}, {"messages", impl_->prompt.messages},
                {"generation_settings", impl_->prompt_contract.at("generation_settings")},
                {"screenshot_width", frame.w}, {"screenshot_height", frame.h},
                {"input_tokens", -1}, {"output_tokens", -1}, {"response", ""}, {"ok", false}, {"t_start_ns", a.t_start_ns},
                {"preflight_input_token_ids", impl_->prompt_input_ids}, {"preflight_input_tokens", impl_->prompt_input_ids.size()},
                {"input_token_ids_available", false},
                {"input_token_ids_unavailable_reason", "Stock runtime getter requires disabled thinker embeddings"},
                {"preflight_input_provenance", "CPU checkpoint tokenization; runtime prompt/count verified"}};
            if (frame.w != 640 || frame.h != 480) throw std::runtime_error("Fixed prompt screenshot geometry changed");
            req.topP = 0.8f; req.applyChatTemplate = false; req.addGenerationPrompt = false;
            sys.contents = {{"text", impl_->prompt.system_text}};
            usr.contents = {{"text", impl_->prompt.user_prefix}, {"image", ""}, {"text", impl_->prompt.user_suffix}};
            // Pristine stock preprocessing accepts host pixels. This compatibility
            // copy is separate from prompt logic and included in decision latency.
            host_frame.resize(static_cast<size_t>(frame.w) * frame.h * 3);
            auto status = cudaMemcpyAsync(host_frame.data(), frame.rgb, host_frame.size(), cudaMemcpyDeviceToHost, impl_->stream);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            status = cudaStreamSynchronize(impl_->stream);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        }
        r.messages = {sys, usr};
        // the frame at B, wrapped without a copy: an external device tensor [1, H, W, 3] uint8; the runner's
        // resize is the identity when the frame already has the model's target size (our upscaled 640x480)
        rt::Tensor t(fixed_prompt ? host_frame.data() : const_cast<uint8_t*>(frame.rgb),
                     rt::Coords{1, (int64_t)frame.h, (int64_t)frame.w, 3},
                     fixed_prompt ? rt::DeviceType::kCPU : rt::DeviceType::kGPU,
                     nvinfer1::DataType::kUINT8, "chronotope.frame_at_B");
        rt::imageUtils::ImageData img(std::move(t));
        img.doResize = true;
        r.imageBuffers.push_back(std::move(img));
        if (!fixed_prompt) r.stopStrings = {"}"};   // Preserve the original legacy JSON stopping rule.
        req.requests.push_back(std::move(r));
        impl_->steps = 0;
        int64_t t_first = 0;
        req.onTokenGenerated = [this, &t_first](rt::TokenCallbackInfo const&) {
            if (impl_->steps == 0) t_first = now_ns();
            ++impl_->steps;
        };
        rt::LLMGenerationResponse resp;
        bool ok = impl_->runtime->handleRequest(req, resp, impl_->stream);
        a.t_end_ns = now_ns();
        impl_->prefill_ms = t_first ? (t_first - a.t_start_ns) / 1e6 : (a.t_end_ns - a.t_start_ns) / 1e6;
        impl_->decode_ms = t_first ? (a.t_end_ns - t_first) / 1e6 : 0.0;
        if (fixed_prompt) {
            if (!resp.inputTokenCounts.empty()) audit["input_tokens"] = resp.inputTokenCounts[0];
            if (!resp.outputIds.empty()) audit["output_tokens"] = resp.outputIds[0].size();
            if (!resp.outputTexts.empty()) audit["response"] = resp.outputTexts[0];
            if (!ok || resp.outputTexts.empty() || resp.outputIds.empty()) throw std::runtime_error("TensorRT request failed or missing token usage");
            if (req.formattedRequests.size() != 1 || req.formattedRequests[0].formattedCompleteRequest != impl_->prompt.formatted_text)
                throw std::runtime_error("Runtime prompt differs from checkpoint rendering");
            audit["formatted_text"] = req.formattedRequests[0].formattedCompleteRequest;
            if (resp.inputTokenCounts.size() != 1 || resp.inputTokenCounts[0] != static_cast<int64_t>(impl_->prompt_input_ids.size()))
                throw std::runtime_error("Runtime token count differs from checkpoint tokenizer preflight");
        }
        if (!ok || resp.outputTexts.empty()) { a.error = "handleRequest failed"; return a; }
        std::string text = resp.outputTexts[0];
        if (!fixed_prompt && text.find('{') != std::string::npos && text.find('}') == std::string::npos) text += "}";
        a.raw = text.substr(0, 500);
        a.output_tokens = resp.outputIds.empty() ? -1 : (int)resp.outputIds[0].size();
        a.input_tokens = resp.inputTokenCounts.empty() ? -1 : resp.inputTokenCounts[0];
        a.model_reported = "tensorrt_edgellm";
        a.request_id = "trt-" + std::to_string(a.t_start_ns);
        a.ok = true;
        if (fixed_prompt) {
            const auto parsed = agents3_prompt::parse_action_once(text, frame.w, frame.h);
            if (!parsed.is_null() && parsed.at("action") == "click") {
                a.click = true; a.image_x = parsed.at("x"); a.image_y = parsed.at("y");
                a.x = a.image_x / impl_->cfg.image_scale; a.y = a.image_y / impl_->cfg.image_scale;
            }
        } else vlm_parse_action(text, &a, impl_->cfg.prompt_version, impl_->cfg.w, impl_->cfg.h, impl_->cfg.image_scale);
    } catch (std::exception const& e) {
        a.t_end_ns = now_ns();
        a.error = e.what();
    }
    if (!audit.is_null()) {
        audit["ok"] = a.ok; audit["t_end_ns"] = now_ns();
        if (!a.error.empty()) audit["error"] = a.error;
        impl_->prompt_audit << audit.dump() << '\n'; impl_->prompt_audit.flush();
        if (!impl_->prompt_audit) { a.ok = false; a.error = "Fixed prompt audit write failed"; }
    }
#else
    (void)frame;
    a.t_end_ns = a.t_start_ns;
    a.error = "not built with Edge-LLM";
#endif
    return a;
}

void TrtPlanner::ask(DeviceFrame frame) {
    if (impl_->busy.load()) return;
    if (impl_->th.joinable()) impl_->th.join();
    impl_->busy = true; impl_->done = false;
    impl_->th = std::thread([this, frame]() { impl_->result = invoke(frame); impl_->done = true; impl_->busy = false; });
}

VlmAnswer TrtPlanner::take() {
    if (impl_->th.joinable()) impl_->th.join();
    impl_->done = false;
    return impl_->result;
}

}  // namespace chronotope
