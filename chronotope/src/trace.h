// trace.h — an append-only JSONL trace for the standalone loop (records after events, never rewritten).
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace chronotope {

class Json {
public:
    Json& i(const char* k, int64_t v);
    Json& u(const char* k, uint64_t v);
    Json& d(const char* k, double v);
    Json& s(const char* k, const std::string& v);
    Json& b(const char* k, bool v);
    Json& raw(const char* k, const std::string& v);   // an already-encoded JSON value
    std::string str() const;
private:
    std::vector<std::pair<std::string, std::string>> kv_;
};

std::string json_escape(const std::string& s);

class JsonlTrace {
public:
    bool open(const std::string& path);
    void append(const char* record, const Json& fields);
    void close();
    ~JsonlTrace() { close(); }
private:
    FILE* f_ = nullptr;
    uint64_t seq_ = 0;
};

}  // namespace chronotope
