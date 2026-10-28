// vector_add.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    // Calculate this thread's global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accessing elements outside the arrays
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1000;
    const int bytes = N * sizeof(float);

    // Allocate host (CPU) memory
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N);

    // Allocate device (GPU) memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy input data from CPU to GPU
    cudaMemcpy(d_A, h_A.data(), bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes,
               cudaMemcpyHostToDevice);

    // Configure the grid and block
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel on the GPU
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, N
    );

    // Wait for the kernel and copy results back
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, bytes,
               cudaMemcpyDeviceToHost);

    // Display results
    std::cout << "C[0] = " << h_C[0] << '\n';
    std::cout << "C[999] = " << h_C[999] << '\n';

    // Release GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
