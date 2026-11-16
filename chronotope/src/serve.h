// K4b: the agent as a server process on a UNIX socket (see serve.cpp).
#pragma once
#include <string>

struct chronotope_core;

namespace chronotope {

class TrtPlanner;
class ApiPlanner;

struct ServeOptions {
    int width = 320, height = 240, image_scale = 2;
};

// Blocks until the client closes; returns the process exit code.
int run_server(chronotope_core* core, const std::string& socket_path, TrtPlanner* trt, const ServeOptions& opt, ApiPlanner* api = nullptr);

}  // namespace chronotope
