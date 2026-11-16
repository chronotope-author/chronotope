// Opt-in hosted inference. Policy, action parsing and device correction stay legacy.
#pragma once
#include "planner_vlm.h"
#include <nlohmann/json.hpp>

namespace chronotope {
struct ApiConfig {
    std::string provider, model, endpoint, api_key_env, audit_path;
    std::string effort = "low";
    int max_output_tokens = 4096, image_scale = 2, w = 320, h = 240;
    int prompt_version = 4;
    double timeout_s = 600;
};

// Kept separate from networking for CPU-only contract tests.
nlohmann::json api_request(const ApiConfig&, const std::string& system, const std::string& image_url);
VlmAnswer api_response(const ApiConfig&, const nlohmann::json&);

class ApiPlanner {
public:
    explicit ApiPlanner(ApiConfig);
    VlmAnswer invoke(const std::vector<uint8_t>& rgb);
    const nlohmann::json& audit() const { return audit_; }
private:
    ApiConfig cfg_;
    nlohmann::json audit_;
};
} // namespace chronotope
