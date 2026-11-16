// Native, opt-in port of the stripped Agent S3 Worker policy. This owns CPU text
// history and calls a supplied in-process model function; it never runs Python.
// The frozen JSON contract supplies the upstream prompt bytes and settings.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace chronotope::agents3_policy {
using Json = nlohmann::json;

inline bool python_space(unsigned c) {
    return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 ||
           c == 0x202f || c == 0x205f || c == 0x3000;
}
inline std::string strip(std::string s, bool left = true, bool right = true) {
    // Python/Jinja strip is Unicode-aware. Preserve all interior bytes exactly.
    std::vector<std::pair<size_t, unsigned>> chars;
    for (size_t i = 0; i < s.size();) {
        size_t start = i; unsigned c = static_cast<unsigned char>(s[i++]);
        int tail = c < 0x80 ? 0 : (c & 0xe0) == 0xc0 ? 1 : (c & 0xf0) == 0xe0 ? 2 : (c & 0xf8) == 0xf0 ? 3 : 0;
        if (tail) { c &= (1u << (6 - tail)) - 1; for (int k = 0; k < tail && i < s.size(); ++k) c = (c << 6) | (static_cast<unsigned char>(s[i++]) & 0x3f); }
        chars.emplace_back(start, c);
    }
    size_t a = 0, b = chars.size();
    if (left) while (a < b && python_space(chars[a].second)) ++a;
    if (right) while (b > a && python_space(chars[b-1].second)) --b;
    size_t begin = a < chars.size() ? chars[a].first : s.size();
    size_t end = b < chars.size() ? chars[b].first : s.size();
    return s.substr(begin, end - begin);
}
inline Json message(const std::string& role, const std::string& text, bool image = false) {
    Json result = {{"role", role}, {"content", Json::array({{{"type", "text"}, {"text", text}}})}};
    if (image) result["content"].push_back({{"type", "image"}});
    return result;
}
inline Json normalize_transport_messages(Json messages, bool strip_historical_thinking = true) {
    // Edge-LLM's copied template has trim_content=false. HF trims the combined
    // text+image message, not each part: preserve newline immediately before an
    // image. Every Worker request ends in a user message, so historical assistant
    // messages use the simple assistant prefix and discard old think sections.
    for (auto& m : messages) {
        auto& parts = m["content"];
        if (parts.empty()) continue;
        if (parts.front().at("type") == "text") parts.front()["text"] = strip(parts.front().at("text"), true, false);
        if (parts.back().at("type") == "text") parts.back()["text"] = strip(parts.back().at("text"), false, true);
        if (strip_historical_thinking && m.at("role") == "assistant" && parts.size() == 1 && parts.front().at("type") == "text") {
            std::string text = parts.front().at("text");
            auto close = text.rfind("</think>");
            if (close != std::string::npos) {
                text.erase(0, close + 8);
                text.erase(0, text.find_first_not_of('\n'));
                parts.front()["text"] = text;
            }
        }
    }
    return messages;
}

inline std::string parse_code(const std::string& text) {
    // Python: re.findall(r'```(?:\w+\s+)?(.*?)```', text.strip(), re.DOTALL)[-1]
    static const std::regex fenced(R"(```(?:\w+\s+)?([\s\S]*?)```)");
    std::string input = strip(text), result;
    for (auto i = std::sregex_iterator(input.begin(), input.end(), fenced); i != std::sregex_iterator(); ++i) result = (*i)[1];
    return result;
}
inline std::string split_tag(const std::string& text, const std::string& tag) {
    const std::string open = "<" + tag + ">", close = "</" + tag + ">";
    auto at = text.rfind(open);
    std::string tail = at == std::string::npos ? text : text.substr(at + open.size());
    auto end = tail.find(close);
    return strip(tail.substr(0, end));
}

// Parse the allowed numeric literal calls without eval, a Python subprocess,
// or access to any other function. Parentheses, Python numeric spellings,
// keyword arguments and comments are accepted; expressions are rejected.
class LiteralCall {
public:
    struct Result { std::string method; std::vector<double> args; std::map<std::string, double> kwargs; };
    explicit LiteralCall(std::string code) : code_(std::move(code)) {}
    Result parse() {
        auto root = expression(); skip(true);
        if (at_ != code_.size() || root.kind != Node::Call) throw std::runtime_error("Expected one agent call");
        return root.call;
    }
private:
    struct Node {
        enum Kind { Name, Attribute, Number, Call } kind;
        std::string name; double number = 0; bool unary = false; Result call;
    };
    std::string code_; size_t at_ = 0; int parentheses_ = 0;
    void skip(bool trailing = false) {
        while (at_ < code_.size()) {
            unsigned char c = code_[at_];
            if (c == '\n' || c == '\r') { if (!trailing && parentheses_ == 0) break; ++at_; }
            else if (c == ' ' || c == '\t' || c == '\f') ++at_;
            else if (c == '#') { while (at_ < code_.size() && code_[at_] != '\n') ++at_; }
            else if (c == '\\' && at_ + 1 < code_.size() && code_[at_+1] == '\n') at_ += 2;
            else break;
        }
    }
    bool take(char c) {
        skip(); if (at_ < code_.size() && code_[at_] == c) {
            ++at_; if (c == '(') ++parentheses_; if (c == ')') --parentheses_; return true;
        } return false;
    }
    void need(char c) { if (!take(c)) throw std::runtime_error("Invalid action syntax"); }
    std::string identifier() {
        skip(); size_t start = at_;
        if (at_ >= code_.size() || !(std::isalpha(static_cast<unsigned char>(code_[at_])) || code_[at_] == '_')) throw std::runtime_error("Expected name");
        while (at_ < code_.size() && (std::isalnum(static_cast<unsigned char>(code_[at_])) || code_[at_] == '_')) ++at_;
        return code_.substr(start, at_ - start);
    }
    Node expression() {
        skip(); if (at_ >= code_.size()) throw std::runtime_error("Empty expression");
        Node n{Node::Number};
        if (take('(')) { n = expression(); need(')'); }
        else if (code_[at_] == '+' || code_[at_] == '-') {
            bool neg = code_[at_++] == '-';
            // ast.literal_eval permits unary +/- of a numeric literal, but not
            // nested unary expressions. Track this syntactically here.
            skip();
            if (at_ < code_.size() && (code_[at_] == '+' || code_[at_] == '-')) throw std::runtime_error("Nested unary expression");
            n = expression(); if (n.kind != Node::Number || n.unary) throw std::runtime_error("Non-literal unary expression");
            if (neg) n.number = -n.number;
            n.unary = true;
        }
        else if (std::isalpha(static_cast<unsigned char>(code_[at_])) || code_[at_] == '_') { n.kind = Node::Name; n.name = identifier(); }
        else {
            size_t start = at_;
            while (at_ < code_.size()) {
                char c = code_[at_];
                if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.') ++at_;
                else if ((c == '+' || c == '-') && at_ > start && (code_[at_-1] == 'e' || code_[at_-1] == 'E')) ++at_;
                else break;
            }
            std::string literal = code_.substr(start, at_ - start);
            const std::string digits = R"([0-9](?:_?[0-9])*)";
            static const std::regex decimal("(?:" + digits + "\\.(?:" + digits + ")?|\\." + digits + ")(?:[eE][+-]?" + digits + ")?|" + digits + "[eE][+-]?" + digits);
            static const std::regex integer(R"((?:0(?:_?0)*|[1-9](?:_?[0-9])*))");
            static const std::regex hex(R"(0[xX]_?[0-9a-fA-F](?:_?[0-9a-fA-F])*)"), oct(R"(0[oO]_?[0-7](?:_?[0-7])*)"), bin(R"(0[bB]_?[01](?:_?[01])*)");
            int base = std::regex_match(literal, hex) ? 16 : std::regex_match(literal, oct) ? 8 : std::regex_match(literal, bin) ? 2 : 10;
            bool real = std::regex_match(literal, decimal);
            if (base == 10 && !real && !std::regex_match(literal, integer)) throw std::runtime_error("Expected real numeric literal");
            literal.erase(std::remove(literal.begin(), literal.end(), '_'), literal.end());
            if (base != 10) {
                long double value = 0;
                for (size_t i = 2; i < literal.size(); ++i) { char c = literal[i]; int digit = c <= '9' ? c-'0' : std::tolower(static_cast<unsigned char>(c))-'a'+10; value = value * base + digit; }
                n.number = static_cast<double>(value);
            } else { n.number = std::strtod(literal.c_str(), nullptr); }
        }
        while (true) {
            if (take('.')) {
                if (n.kind != Node::Name || n.name != "agent") throw std::runtime_error("Only agent methods allowed");
                n.kind = Node::Attribute; n.name = identifier();
            } else if (take('(')) {
                if (n.kind != Node::Attribute) throw std::runtime_error("Expected agent method");
                Result call; call.method = n.name; bool keywords = false;
                if (!take(')')) while (true) {
                    skip(); size_t saved = at_; std::string key;
                    if (at_ < code_.size() && (std::isalpha(static_cast<unsigned char>(code_[at_])) || code_[at_] == '_')) {
                        key = identifier(); if (!take('=')) { at_ = saved; key.clear(); }
                    }
                    Node value = expression(); if (value.kind != Node::Number || !std::isfinite(value.number)) throw std::runtime_error("Action parameters must be finite numbers");
                    if (key.empty()) { if (keywords) throw std::runtime_error("Positional argument after keyword"); call.args.push_back(value.number); }
                    else { keywords = true; call.kwargs[key] = value.number; }
                    if (take(')')) break;
                    need(','); if (take(')')) break;
                }
                n.kind = Node::Call; n.call = std::move(call);
            } else break;
        }
        return n;
    }
};
inline long long python_round(double value) {
    double lo = std::floor(value), fraction = value-lo;
    if (fraction < 0.5) return static_cast<long long>(lo);
    if (fraction > 0.5) return static_cast<long long>(lo+1);
    return static_cast<long long>(std::fmod(std::abs(lo), 2.0) == 0 ? lo : lo+1);
}
inline Json executable_action(const std::string& code, int width, int height) {
    try {
        auto c = LiteralCall(strip(code)).parse();
        if (c.method == "done" || c.method == "fail") {
            if (!c.args.empty() || !c.kwargs.empty()) return nullptr;
            return c.method == "done" ? "DONE" : "FAIL";
        }
        std::vector<std::string> names;
        if (c.method == "click") names = {"x", "y"};
        else if (c.method == "wait") names = {"time"};
        else return nullptr;
        if (c.args.size() > names.size()) return nullptr;
        std::map<std::string, double> values = c.kwargs;
        for (auto& [key, value] : values) if (std::find(names.begin(), names.end(), key) == names.end()) return nullptr;
        for (size_t i = 0; i < c.args.size(); ++i) { if (values.count(names[i])) return nullptr; values[names[i]] = c.args[i]; }
        if (c.method == "wait") return values.count("time") && values["time"] < 0 ? Json(nullptr) : Json("WAIT");
        if (values.size() != 2) return nullptr;
        // Bound before integer conversion: only values that can round into the
        // screenshot are relevant, including Python's ties-to-even at -0.5.
        if (values["x"] < -0.5 || values["y"] < -0.5 || values["x"] > width || values["y"] > height) return nullptr;
        auto x = python_round(values["x"]), y = python_round(values["y"]);
        if (x < 0 || y < 0 || x >= width || y >= height) return nullptr;
        return "pyautogui.click(" + std::to_string(x) + ", " + std::to_string(y) + ")";
    } catch (const std::exception&) { return nullptr; }
}
inline Json validate_plan_response(const std::string& text, int width, int height) {
    std::string code = parse_code(text);
    static const std::regex action(R"((agent\.\w+\(\s*.*\)))");
    size_t count = std::distance(std::sregex_iterator(code.begin(), code.end(), action), std::sregex_iterator());
    Json errors = Json::array(), executable = executable_action(code, width, height);
    if (count != 1) errors.push_back("Incorrect code: There must be a single agent action in the code response.");
    if (executable.is_null()) errors.push_back("Incorrect code: The agent action must be a valid function and use valid parameters from the docstring list.");
    return {{"code", code}, {"executable", executable}, {"errors", errors}};
}
inline Json parsed_action(const std::string& executable) {
    if (executable == "WAIT") return {{"action", "wait"}};
    static const std::regex click(R"(pyautogui\.click\(([0-9]+), ([0-9]+)\))"); std::smatch match;
    if (std::regex_match(executable, match, click)) return {{"action", "click"}, {"x", std::stoi(match[1])}, {"y", std::stoi(match[2])}};
    return nullptr;
}

class Policy {
public:
    using Respond = std::function<std::string(const std::string&, const Json&, int, int)>;
    using Sleep = std::function<void(double)>;
    explicit Policy(Json contract) : contract_(std::move(contract)) {
        if (contract_.at("schema") != "agents3-stripped-worker-policy-1" ||
            contract_.at("max_tokens") != 4096 || contract_.at("temperature") != 0.0 ||
            contract_.at("top_p") != 0.8 || contract_.at("repetition_penalty") != 1.05 ||
            contract_.at("enable_thinking") != false || contract_.at("retry_limit") != 3 ||
            contract_.at("transport_retry_limit") != 3 || contract_.at("planner_message_limit") != 17 ||
            contract_.at("reflection_message_limit") != 9 || contract_.at("screenshot_width").get<int>() <= 0 ||
            contract_.at("screenshot_height").get<int>() <= 0 || !contract_.at("stop").empty())
            throw std::runtime_error("Unsupported stripped Agent S3 policy contract");
        planner_ = Json::array({message("system", contract_.at("planner_system"))});
        reflection_ = Json::array({message("system", contract_.at("reflection_system"))});
    }
    const Json& planner_messages() const { return planner_; }
    const Json& reflection_messages() const { return reflection_; }
    const Json& contract() const { return contract_; }
    int turn_count() const { return turn_; }
    Json run_turn(const Respond& respond, const Sleep& sleep) {
        strip_images();
        const auto& c = contract_;
        Json reflection = nullptr, thoughts = nullptr;
        std::string generator;
        if (turn_ == 0) {
            reflection_.push_back(message("user", c.at("initial_reflection_user_text"), true));
            generator = c.at("initial_user_text");
        } else {
            reflection_.push_back(message("user", last_plan_, true));
            auto full = safe_call(respond, sleep, "reflection", reflection_, 0);
            reflection = split_tag(full, "answer"); thoughts = split_tag(full, "thoughts");
        }
        if (!reflection.is_null() && !reflection.get<std::string>().empty()) generator += c.at("reflection_planner_prefix").get<std::string>() + reflection.get<std::string>() + "\n";
        generator += c.at("text_buffer_suffix").get<std::string>();
        planner_.push_back(message("user", generator, true));
        Json transient = planner_;
        std::string plan;
        for (int attempt = 1; attempt <= c.at("retry_limit").get<int>(); ++attempt) {
            plan = safe_call(respond, sleep, "planner", transient, attempt);
            Json checked = validate_plan_response(plan, c.at("screenshot_width"), c.at("screenshot_height"));
            if (checked.at("errors").empty()) break;
            transient.push_back(message("assistant", plan));
            std::string errors;
            for (const auto& error : checked.at("errors")) errors += (errors.empty() ? "- " : "\n- ") + error.get<std::string>();
            std::string feedback = c.at("feedback_template");
            const std::string placeholder = "FORMATTING_FEEDBACK";
            size_t pos = 0;
            while ((pos = feedback.find(placeholder, pos)) != std::string::npos) { feedback.replace(pos, placeholder.size(), errors); pos += errors.size(); }
            transient.push_back(message("user", feedback));
            sleep(c.at("retry_sleep_seconds"));
        }
        last_plan_ = plan; planner_.push_back(message("assistant", plan));
        Json checked = validate_plan_response(plan, c.at("screenshot_width"), c.at("screenshot_height"));
        std::string executable = checked.at("executable").is_null() ? "WAIT" : checked.at("executable").get<std::string>();
        ++turn_;
        if (planner_.size() > c.at("planner_message_limit").get<size_t>()) planner_.erase(planner_.begin()+1, planner_.begin()+3);
        if (reflection_.size() > c.at("reflection_message_limit").get<size_t>()) reflection_.erase(reflection_.begin()+1);
        strip_images();
        if (!transport_errors_.empty()) throw std::runtime_error("Inference service error: " + transport_errors_.back());
        return {{"plan", plan}, {"plan_code", checked.at("code")}, {"exec_code", executable},
                {"reflection", reflection}, {"reflection_thoughts", thoughts}, {"parsed", parsed_action(executable)}};
    }
private:
    Json contract_, planner_, reflection_;
    int turn_ = 0; std::string last_plan_;
    std::vector<std::string> transport_errors_;
    void strip_images() {
        for (Json* history : {&planner_, &reflection_}) for (auto& m : *history) {
            auto& parts = m["content"];
            parts.erase(std::remove_if(parts.begin(), parts.end(), [](const auto& part) { return part.at("type") != "text"; }), parts.end());
        }
    }
    std::string safe_call(const Respond& respond, const Sleep& sleep, const std::string& kind, const Json& messages, int format_attempt) {
        for (int attempt = 1; attempt <= contract_.at("transport_retry_limit").get<int>(); ++attempt) {
            try { return respond(kind, messages, format_attempt, attempt); }
            catch (const std::exception& error) { transport_errors_.push_back(error.what()); }
            sleep(contract_.at("retry_sleep_seconds"));
        }
        return "";
    }
};
} // namespace chronotope::agents3_policy
