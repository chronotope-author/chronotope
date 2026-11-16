// Technical image-input test only: no model loading and no task episodes.
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
using namespace trt_edgellm;
using namespace trt_edgellm::rt;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    constexpr int H=480,W=640,C=3,N=2*H*W*C,P=1200*1536;
    // The real Qwen runner duplicates still images to temporal_patch_size=2.
    std::vector<unsigned char> rgb(N);
    for (int i=0;i<N;++i) rgb[i]=static_cast<unsigned char>(i%256);
    float stats[3]={0.5f,0.5f,0.5f};
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    Tensor image({2,H,W,C},DeviceType::kGPU,nvinfer1::DataType::kUINT8);
    Tensor normalized({2,H,W,C},DeviceType::kGPU,nvinfer1::DataType::kHALF);
    Tensor mean({3},DeviceType::kGPU,nvinfer1::DataType::kFLOAT);
    Tensor std({3},DeviceType::kGPU,nvinfer1::DataType::kFLOAT);
    Tensor patches({1200,1536},DeviceType::kGPU,nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpyAsync(image.rawPointer(),rgb.data(),N,cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(mean.rawPointer(),stats,sizeof(stats),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(std.rawPointer(),stats,sizeof(stats),cudaMemcpyHostToDevice,stream));
    kernel::normalizeImage(image,mean,std,normalized,stream);
    kernel::transposeToPatchQwenViT(normalized,patches,0,2,16,2,stream);
    std::vector<unsigned short> actual(P);
    CUDA_CHECK(cudaMemcpyAsync(actual.data(),patches.rawPointer(),P*sizeof(unsigned short),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::ofstream output(argv[1],std::ios::binary);
    output.write(reinterpret_cast<char const*>(actual.data()),P*sizeof(unsigned short));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return output.good() ? 0 : 3;
}
