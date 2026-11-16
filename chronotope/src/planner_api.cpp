#include "planner_api.h"
#include <curl/curl.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace chronotope {
using Json = nlohmann::json;
namespace {
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
size_t append(char* ptr, size_t size, size_t count, void* out) {
    static_cast<std::string*>(out)->append(ptr, size * count); return size * count;
}
constexpr const char* user_text = "Screenshot attached. Click the target's center or wait.";
void validate(const ApiConfig& c) {
    const bool openai = c.provider == "openai";
    if ((!openai && c.provider != "anthropic") ||
        c.model != (openai ? "gpt-6.1-sol" : "claude-fable-5-1"))
        throw std::invalid_argument("Unsupported hosted provider/model pair");
    if (c.endpoint != (openai ? "https://api.openai.com/v1/responses" : "https://api.anthropic.com/v1/messages"))
        throw std::invalid_argument("Hosted evaluation requires the official HTTPS endpoint");
    if (c.api_key_env != (openai ? "OPENAI_API_KEY" : "ANTHROPIC_API_KEY"))
        throw std::invalid_argument("Unexpected credential environment variable");
    if (c.effort != "low" || c.max_output_tokens != 4096 || c.prompt_version != 4 ||
        c.w != 320 || c.h != 240 || c.image_scale != 2 || c.timeout_s != 600)
        throw std::invalid_argument("Hosted Notification contract differs from the frozen configuration");
}
} // namespace

Json api_request(const ApiConfig& c, const std::string& system, const std::string& image_url) {
    validate(c);
    if (c.provider == "openai") {
        return {{"model", c.model}, {"store", false}, {"service_tier", "default"},
                {"reasoning", {{"effort", c.effort}}}, {"max_output_tokens", c.max_output_tokens},
                {"input", Json::array({
                    {{"role", "system"}, {"content", Json::array({{{"type", "input_text"}, {"text", system}}})}},
                    {{"role", "user"}, {"content", Json::array({
                        {{"type", "input_image"}, {"image_url", image_url}, {"detail", "high"}},
                        {{"type", "input_text"}, {"text", user_text}}})}}})}};
    }
    const std::string prefix = "data:image/png;base64,";
    if (image_url.rfind(prefix, 0)) throw std::invalid_argument("Expected PNG data URL");
    return {{"model", c.model}, {"max_tokens", c.max_output_tokens}, {"service_tier", "standard_only"},
            {"thinking", {{"type", "adaptive"}}}, {"output_config", {{"effort", c.effort}}},
            {"stop_sequences", Json::array({"}"})}, {"system", system},
            {"messages", Json::array({{{"role", "user"}, {"content", Json::array({
                {{"type", "image"}, {"source", {{"type", "base64"}, {"media_type", "image/png"}, {"data", image_url.substr(prefix.size())}}}},
                {{"type", "text"}, {"text", user_text}}})}}})}};
}

VlmAnswer api_response(const ApiConfig& c, const Json& j) {
    VlmAnswer a;
    a.request_id = j.at("id").get<std::string>();
    a.model_reported = j.at("model").get<std::string>();
    if (a.model_reported != c.model) throw std::runtime_error("Provider returned a different model identifier; explicit review required");
    const auto& usage = j.at("usage");
    a.input_tokens = usage.at("input_tokens").get<int>();
    a.output_tokens = usage.at("output_tokens").get<int>();
    if (a.input_tokens < 0 || a.output_tokens < 0) throw std::runtime_error("Invalid API token usage");
    std::string content;
    if (c.provider == "openai") {
        if (j.value("status", "") != "completed" && j.value("status", "") != "incomplete")
            throw std::runtime_error("Provider response did not complete");
        for (const auto& item : j.at("output"))
            if (item.value("type", "") == "message")
                for (const auto& part : item.at("content"))
                    if (part.value("type", "") == "output_text") content += part.at("text").get<std::string>();
    } else {
        for (const auto& part : j.at("content"))
            if (part.value("type", "") == "text") content += part.at("text").get<std::string>();
        // The legacy stop string consumes the closing brace.
        if (j.value("stop_reason", "") == "stop_sequence" && j.value("stop_sequence", "") == "}") content += "}";
    }
    a.raw = content; // Full text retained, including refusal/truncation evidence.
    a.ok = true; // A model's empty, refused or malformed answer is a scored task failure.
    vlm_parse_action(content, &a, c.prompt_version, c.w, c.h, c.image_scale);
    return a;
}

ApiPlanner::ApiPlanner(ApiConfig c) : cfg_(std::move(c)) {
    validate(cfg_);
    if (cfg_.audit_path.empty()) throw std::invalid_argument("Hosted inference requires an audit path");
    const char* key = std::getenv(cfg_.api_key_env.c_str());
    if (!key || !*key) throw std::runtime_error("Missing " + cfg_.api_key_env);
    static const auto initialized = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (initialized != CURLE_OK) throw std::runtime_error("curl_global_init failed");
}

VlmAnswer ApiPlanner::invoke(const std::vector<uint8_t>& rgb) {
    VlmAnswer answer;
    const auto start = now_ns();
    audit_ = {{"schema", "hosted-api-call-1"}, {"role", "planner"}, {"provider", cfg_.provider},
              {"model", cfg_.model}, {"start_ns", start}, {"transport_retries", 0},
              {"cpu_image_staging", true}, {"pixel_bytes", rgb.size()}};
    try {
        const auto png = VlmPlanner::encode_png(rgb, cfg_.w, cfg_.h, cfg_.image_scale);
        const auto url = "data:image/png;base64," + VlmPlanner::base64(png);
        auto request = api_request(cfg_, vlm_system_prompt(cfg_.prompt_version, cfg_.w, cfg_.h, cfg_.image_scale), url);
        const auto body = request.dump();
        // Retain exact model inputs (synthetic benchmark screenshots), never headers/credentials.
        audit_["request"] = request;
        audit_["request_start_ns"] = now_ns();
        CURL* curl = curl_easy_init();
        if (!curl) throw std::runtime_error("curl_easy_init failed");
        curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        const std::string key = std::getenv(cfg_.api_key_env.c_str());
        headers = curl_slist_append(headers, ((cfg_.provider == "openai" ? "Authorization: Bearer " : "x-api-key: ") + key).c_str());
        if (cfg_.provider == "anthropic") headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");
        std::string response;
        curl_easy_setopt(curl, CURLOPT_URL, cfg_.endpoint.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(body.size()));
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, append);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(cfg_.timeout_s * 1000));
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "https");
        // Certificate verification remains enabled; no redirects or silent fallback.
        const auto result = curl_easy_perform(curl);
        long status = 0; curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
        curl_slist_free_all(headers); curl_easy_cleanup(curl);
        audit_["http_status"] = status;
        audit_["response_end_ns"] = now_ns();
        if (result != CURLE_OK) throw std::runtime_error(std::string("HTTPS transport: ") + curl_easy_strerror(result));
        if (status != 200) throw std::runtime_error("Provider HTTP " + std::to_string(status));
        const auto parsed = Json::parse(response);
        audit_["response"] = parsed;
        answer = api_response(cfg_, parsed);
    } catch (const std::exception& e) {
        answer.error = e.what(); audit_["error"] = answer.error;
    }
    answer.t_start_ns = start; answer.t_end_ns = now_ns();
    audit_["end_ns"] = answer.t_end_ns;
    std::ofstream out(cfg_.audit_path, std::ios::app);
    out << audit_.dump() << '\n'; out.flush();
    if (!out) { answer.ok = false; answer.error = "Cannot write hosted API audit"; }
    return answer;
}
} // namespace chronotope
