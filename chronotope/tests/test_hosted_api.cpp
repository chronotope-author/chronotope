// CPU-only wire contract tests. No GPU, model calls, or experimental episodes.
#include "planner_api.h"
#include <iostream>
#include <stdexcept>

using namespace chronotope;
using Json = nlohmann::json;
void check(bool condition) { if (!condition) throw std::runtime_error("Hosted API contract failed"); }
int main() {
    for (const auto& provider : {"openai", "anthropic"}) {
        ApiConfig c; c.provider = provider;
        const bool openai = c.provider == "openai";
        c.model = openai ? "gpt-6.1-sol" : "claude-fable-5-1";
        c.endpoint = openai ? "https://api.openai.com/v1/responses" : "https://api.anthropic.com/v1/messages";
        c.api_key_env = openai ? "OPENAI_API_KEY" : "ANTHROPIC_API_KEY";
        auto req = api_request(c, "compact prompt", "data:image/png;base64,AA==");
        check(!req.contains("temperature") && !req.contains("tools") && !req.contains("chat_template_kwargs"));
        check(req[openai ? "max_output_tokens" : "max_tokens"] == 4096);
        Json response = {{"id", "request-test"}, {"model", c.model}, {"usage", {{"input_tokens", 42}, {"output_tokens", 12}}}};
        if (openai) {
            response["status"] = "completed";
            response["output"] = Json::array({{{"type", "reasoning"}}, {{"type", "message"},
                {"content", Json::array({{{"type", "output_text"}, {"text", "{\"action\":\"click\",\"x\":500,\"y\":500}"}}})}}});
        } else {
            response["content"] = Json::array({{{"type", "thinking"}, {"thinking", "ignored"}},
                {{"type", "text"}, {"text", "{\"action\":\"click\",\"x\":500,\"y\":500"}}});
            response["stop_reason"] = "stop_sequence"; response["stop_sequence"] = "}";
        }
        auto a = api_response(c, response);
        check(a.ok && a.click && a.x == 160 && a.y == 120 && a.input_tokens == 42);
        if (openai) response["output"] = Json::array({{{"type", "message"}, {"content", Json::array({{{"type", "refusal"}, {"refusal", "no"}}})}}});
        else { response["content"] = Json::array(); response["stop_reason"] = "max_tokens"; }
        a = api_response(c, response); check(a.ok && !a.click);
        response["model"] = "wrong-model";
        bool rejected = false;
        try { api_response(c, response); } catch (const std::exception&) { rejected = true; }
        check(rejected);
    }
    std::cout << "Hosted API contracts passed (zero model calls).\n";
}
