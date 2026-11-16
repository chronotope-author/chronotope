// chronotope_devlink_stub.cu — keeps chronotope_main a CUDA target so CMake device-links the Edge-LLM objects; main.cu itself
// has no kernels and is compiled as C++ (nvcc's front end crashes on it).
__global__ void chronotope_devlink_stub_kernel() {}
