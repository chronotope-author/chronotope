// Stateless adapter to the unchanged Chronotope vision kernels.
// No Core, control.cu, event ledger, dependency matrix or meta-control is linked.
// AgentState is zeroed ABI scratch for cut_template's Proposal fields ONLY.
#include "vision.cuh"
#include <algorithm>
#include <vector>
#include <stdexcept>

using namespace chronotope;
struct Matcher {
    int w,h,r,side,Wv,Hv,n,nb;
    cudaStream_t s{};
    std::vector<void*> allocations;
    uint8_t *p[2]; float *g[2], *rgb[2], *raw, *zero;
    AgentState *scratch; TemplateStats *stats;
    double *scores; ArgmaxPair *am,*ap; PeakStats *pk,*pp;
    template<class T> T* alloc(size_t n) {
        T* ptr=nullptr;
        if(cudaMalloc(&ptr,n*sizeof(T))!=cudaSuccess) throw std::runtime_error("allocation");
        allocations.push_back(ptr); return ptr;
    }
    Matcher(int W,int H,int R):w(W),h(H),r(R),side(2*R+1),Wv(W-side+1),Hv(H-side+1),n(Wv*Hv) {
        if (Wv<=0 || Hv<=0 || R<0) throw std::runtime_error("dimensions");
        if(cudaStreamCreate(&s)!=cudaSuccess) throw std::runtime_error("stream");
        try {
            nb=vision::argmax_blocks(n);
            for(int i=0;i<2;i++){p[i]=alloc<uint8_t>(w*h*3);g[i]=alloc<float>(w*h);rgb[i]=alloc<float>(w*h*3);}
            raw=alloc<float>(side*side);zero=alloc<float>(side*side);
            scratch=alloc<AgentState>(1);stats=alloc<TemplateStats>(1);scores=alloc<double>(n);
            am=alloc<ArgmaxPair>(1);ap=alloc<ArgmaxPair>(nb);pk=alloc<PeakStats>(1);pp=alloc<PeakStats>(nb);
        } catch (...) { for(auto p:allocations) cudaFree(p); cudaStreamDestroy(s); throw; }
    }
    ~Matcher(){for(auto p:allocations)cudaFree(p);cudaStreamDestroy(s);}
    int track(const uint8_t* before,const uint8_t* newest,int x,int y,double* out){
        AgentState host{}; host.proposal.status=PS_PROPOSED;
        host.proposal.action=Action{ACT_CLICK,x,y,0,1.f,0};
        cudaMemcpyAsync(scratch,&host,sizeof(host),cudaMemcpyHostToDevice,s);
        const uint8_t* input[2]={before,newest};
        for(int i=0;i<2;i++){
            cudaMemcpyAsync(p[i],input[i],size_t(w)*h*3,cudaMemcpyHostToDevice,s);
            vision::to_planes(p[i],PIX_RGB24,w,h,p[i],g[i],rgb[i],s);
        }
        vision::cut_template(g[0],rgb[0],w,h,r,1,scratch,raw,s);
        vision::template_stats(raw,side*side,side,1,stats,zero,s);
        vision::ncc(g[1],rgb[1],w,h,side,1,zero,stats,scores,s);
        vision::argmax(scores,n,ap,am,s);
        vision::peak_stats(scores,Wv,Hv,am,side,pp,pk,s);
        TemplateStats ht{};ArgmaxPair ha{};PeakStats hp{};
        cudaMemcpyAsync(&ht,stats,sizeof(ht),cudaMemcpyDeviceToHost,s);
        cudaMemcpyAsync(&ha,am,sizeof(ha),cudaMemcpyDeviceToHost,s);
        cudaMemcpyAsync(&hp,pk,sizeof(hp),cudaMemcpyDeviceToHost,s);
        cudaMemcpyAsync(&host,scratch,sizeof(host),cudaMemcpyDeviceToHost,s);
        if(cudaStreamSynchronize(s)!=cudaSuccess)return 1;
        auto &p=host.proposal;
        if(ht.flat){out[0]=p.tpl_dx;out[1]=p.tpl_dy;out[2]=0;out[3]=1;out[4]=FLAT_PLATEAU;}
        else {out[0]=ha.index%Wv+p.tpl_dx;out[1]=ha.index/Wv+p.tpl_dy;
              out[2]=std::clamp(ha.score,-1.,1.);out[3]=hp.second;out[4]=hp.plateau;}
        return 0;
    }
};
extern "C" void* lfu_create(int w,int h,int r){try{return new Matcher(w,h,r);}catch(...){return nullptr;}}
extern "C" void lfu_destroy(void* p){delete static_cast<Matcher*>(p);}
extern "C" int lfu_track(void* p,const uint8_t* b,const uint8_t* d,int x,int y,double* out){
    return static_cast<Matcher*>(p)->track(b,d,x,y,out);
}
