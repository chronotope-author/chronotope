// planner_vlm.h — the slow VLM (design §7): the same request the paper's LocalOpenAIAdapter sends to the
// local OpenAI-compatible server (system prompt of the declared version, the frame as a PNG data URL,
// structured JSON answer {"action": "click"|"wait", "x", "y"}), issued from a host thread so that the
// core keeps admitting frames while the model deliberates. Its answer enters the core as the one typed
// pre-commit state through chronotope_propose(). No third-party dependency: HTTP/1.1 over a POSIX socket, PNG
// through zlib.
#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace chronotope {

struct VlmConfig {
    std::string base_url = "http://127.0.0.1:8000/v1";
    std::string model;
    int max_tokens = 64;
    double temperature = 0.0;
    bool enable_thinking = false;
    int prompt_version = 5;       // 1-3 pixels, 4-6 the 0-1000 grid (chronotope.controller.SYSTEM_PROMPTS)
    int image_scale = 2;
    int w = 320, h = 240;
    double timeout_s = 600.0;
};

struct VlmAnswer {
    bool ok = false;              // the request completed and parsed
    bool click = false;           // false = wait / unparseable
    int x = 0, y = 0;             // canvas pixels (already mapped from the prompt's coordinate space)
    int image_x = 0, image_y = 0; // as answered
    std::string raw, error, request_id, model_reported;
    int64_t t_start_ns = 0, t_end_ns = 0;
    int input_tokens = -1, output_tokens = -1;
};

class VlmPlanner {
public:
    explicit VlmPlanner(VlmConfig cfg);
    ~VlmPlanner();
    void ask(const std::vector<uint8_t>& rgb);      // one fixed-input invocation on this frame (w*h*3)
    bool in_flight() const { return busy_.load(); }
    bool done() const { return done_.load(); }
    VlmAnswer take();                                // after done(): the answer; resets the planner
    const std::string& system_prompt() const { return system_; }
    static std::vector<uint8_t> encode_png(const std::vector<uint8_t>& rgb, int w, int h, int scale);
    static std::string base64(const std::vector<uint8_t>& bytes);
    // one blocking call (used by ask's thread and by tests)
    VlmAnswer invoke(const std::vector<uint8_t>& rgb);
private:
    VlmConfig cfg_;
    std::string system_;
    std::thread th_;
    std::atomic<bool> busy_{false}, done_{false};
    VlmAnswer result_;
};

// the system prompt of a declared version (chronotope.controller.SYSTEM_PROMPTS 1-6), and the answer parser +
// coordinate mapping shared by the HTTP and the in-process planners
std::string vlm_system_prompt(int prompt_version, int w, int h, int image_scale);
bool vlm_parse_action(const std::string& content, VlmAnswer* a, int prompt_version, int w, int h, int image_scale);
// minimal HTTP/1.1 POST (Content-Length and chunked responses); returns false on transport error
bool http_post_json(const std::string& url, const std::string& body, double timeout_s, std::string* response, int* status, std::string* error);

}  // namespace chronotope
