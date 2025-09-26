#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

namespace py = pybind11;

__global__ void monte_carlo_kernel(float *payoffs, int simulations, float s, float k, float t, float r, float sigma){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < simulations){
        curandState_t state;
        curand_init(tid, 0, 0, &state);

        int steps = max(1, (int)lrintf(252.0f * t));
        float dt = t / steps;
        float current_price = s;

        for(int i = 0; i < steps; i++){
            float z = curand_normal(&state);
            float next_price = current_price * expf((r - (sigma * sigma)/2.0f) * dt + sigma * sqrtf(dt) * z);
            current_price = next_price;
        }
        payoffs[tid] = fmaxf(current_price - k, 0.0f);
    }
}

double monte_carlo_cuda_py(float s, float k, float t, float r, float sigma, int simulations) {
    float *d_payoffs;
    cudaMalloc((void**)&d_payoffs, simulations * sizeof(float));

    int threadsPerBlock = 256;
    int numBlocks = (simulations + threadsPerBlock - 1) / threadsPerBlock;
    monte_carlo_kernel<<<numBlocks, threadsPerBlock>>>(d_payoffs, simulations, s, k, t, r, sigma);
    cudaDeviceSynchronize();

    float *h_payoffs = new float[simulations];
    cudaMemcpy(h_payoffs, d_payoffs, simulations * sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < simulations; i++) {
        sum += h_payoffs[i];
    }
    double avg_payoff = sum / simulations;
    double option_price = avg_payoff * exp(-r * t);

    delete[] h_payoffs;
    cudaFree(d_payoffs);

    return option_price;
}

PYBIND11_MODULE(monte_carlo_pybind, m) {
    m.def("price_option", &monte_carlo_cuda_py, "CUDA-accelerated Monte Carlo Option Pricer");
}
