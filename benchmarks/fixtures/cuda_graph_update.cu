// Minimal CUDA capture/update lifecycle, following the current llama.cpp path.
// Each of 12 graph launches executes one kernel; the result must be 12.
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void increment(float *value) { *value += 1; }

int main() {
    float *value = nullptr, result = 0;
    cudaStream_t stream;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CHECK(cudaMalloc(&value, sizeof(float)));
    CHECK(cudaMemset(value, 0, sizeof(float)));
    CHECK(cudaStreamCreate(&stream));
    for (int i = 0; i < 12; ++i) {
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        increment<<<1, 1, 0, stream>>>(value);
        if (graph) CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamEndCapture(stream, &graph));
        if (!executable) CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        cudaGraphExecUpdateResultInfo update;
        CHECK(cudaGraphExecUpdate(executable, graph, &update));
        CHECK(cudaGraphLaunch(executable, stream));
        CHECK(cudaStreamSynchronize(stream));
    }
    CHECK(cudaMemcpy(&result, value, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaGraphExecDestroy(executable));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(value));
    std::printf("launches=12 result=%g\n", result);
    return result == 12 ? 0 : 1;
}
