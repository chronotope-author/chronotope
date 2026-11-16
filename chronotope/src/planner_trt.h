// planner_trt.h — the slow VLM in-process (design §7, planner.cu): NVIDIA TensorRT Edge-LLM's C++ runtime
// on the planner stream, replacing the HTTP hop to vLLM. The frame at B is handed over as a DEVICE buffer
// (the core's pinned frame, upscaled on the planner stream). The opt-in stock-runtime
// Agent S3 prompt path stages that frame on the CPU for the public image API; the
// runtime's own decode loop is host-driven (one stream synchronization per generated token), which the
// planner counts as host_steps for the audit. Built only when chronotope is configured with CHRONOTOPE_EDGELLM_BUILD;
// otherwise available() is false and the constructor throws.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "planner_vlm.h"   // VlmAnswer, the prompt table and the answer parser are shared

namespace chronotope {

struct TrtPlannerConfig {
    std::string engine_dir;             // llm_build output (engine + tokenizer + processed_chat_template.json)
    std::string multimodal_engine_dir;  // visual_build output root
    int prompt_version = 5;
    int image_scale = 2;                // the frame is upscaled by this factor before the model, as the paper's PNG
    int w = 320, h = 240;               // frame size before upscaling
    int max_tokens = 64;
    double temperature = 0.0;
    bool enable_thinking = false;
    bool capture_cuda_graph = true;
    std::string prompt_path;            // opt-in fixed Agent S3 prompt; no Worker/history/reflection
    std::string prompt_audit_path;      // one request/response record per decision
};

struct DeviceFrame {
    const uint8_t* rgb = nullptr;       // RGB24, row-major, on the device, already at (w, h)
    int w = 0, h = 0;
};

class TrtPlanner {
public:
    static bool compiled();             // true when built with Edge-LLM
    TrtPlanner(TrtPlannerConfig cfg, void* cuda_stream);   // throws std::runtime_error on load failure
    ~TrtPlanner();
    void ask(DeviceFrame frame);        // one fixed-input invocation on this device frame
    bool in_flight() const;
    bool done() const;
    VlmAnswer take();
    const std::string& system_prompt() const;
    uint64_t host_steps() const;        // per-token host synchronizations of the last request
    double prefill_ms() const;          // start -> first token
    double decode_ms() const;           // first token -> end
    std::string runtime_info() const;   // engine dirs and settings, for the trace header
    VlmAnswer invoke(DeviceFrame frame);   // blocking (used by ask's thread and by tests)
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace chronotope
