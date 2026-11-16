#include "trace.h"
#include <cstdio>

namespace chronotope {

std::string json_escape(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o + "\"";
}

Json& Json::i(const char* k, int64_t v) { kv_.emplace_back(k, std::to_string(v)); return *this; }
Json& Json::u(const char* k, uint64_t v) { kv_.emplace_back(k, std::to_string(v)); return *this; }
Json& Json::d(const char* k, double v) { char b[64]; std::snprintf(b, sizeof b, "%.17g", v); kv_.emplace_back(k, b); return *this; }
Json& Json::s(const char* k, const std::string& v) { kv_.emplace_back(k, json_escape(v)); return *this; }
Json& Json::b(const char* k, bool v) { kv_.emplace_back(k, v ? "true" : "false"); return *this; }
Json& Json::raw(const char* k, const std::string& v) { kv_.emplace_back(k, v); return *this; }

std::string Json::str() const {
    std::string o = "{";
    for (size_t i = 0; i < kv_.size(); ++i) {
        if (i) o += ",";
        o += json_escape(kv_[i].first) + ":" + kv_[i].second;
    }
    return o + "}";
}

bool JsonlTrace::open(const std::string& path) {
    f_ = std::fopen(path.c_str(), "wx");   // never overwrite a trace
    return f_ != nullptr;
}

void JsonlTrace::append(const char* record, const Json& fields) {
    if (!f_) return;
    Json j = fields;
    std::string body = j.str();
    ++seq_;
    std::fprintf(f_, "{\"seq\":%llu,\"record\":%s,%s\n", (unsigned long long)seq_, json_escape(record).c_str(), body.c_str() + 1);
    std::fflush(f_);
}

void JsonlTrace::close() {
    if (f_) { std::fclose(f_); f_ = nullptr; }
}

}  // namespace chronotope
