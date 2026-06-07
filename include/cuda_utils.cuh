#pragma once
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ─── Error checking ────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── RAII device buffer ────────────────────────────────────────────────────────

/**
 * Manages a single contiguous CUDA device allocation.
 *
 * Usage:
 *   CudaBuffer<int> d_M(host_vec);          // alloc + upload
 *   CudaBuffer<int> d_flag(1);              // alloc only (zeroed)
 *   kernel<<<...>>>(d_M.ptr(), ...);
 *   d_M.download(host_vec);                 // copy back
 */
template <typename T>
class CudaBuffer {
public:
    // Allocate and upload from a host vector.
    explicit CudaBuffer(const std::vector<T>& host)
        : size_(host.size())
    {
        CUDA_CHECK(cudaMalloc(&ptr_, size_ * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), size_ * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    // Allocate `count` elements, zero-initialized.
    explicit CudaBuffer(size_t count)
        : size_(count)
    {
        CUDA_CHECK(cudaMalloc(&ptr_, size_ * sizeof(T)));
        CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
    }

    ~CudaBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    // Non-copyable, movable.
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T*     ptr()  const { return ptr_; }
    size_t size() const { return size_; }

    // Upload from host vector (must match size).
    void upload(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), size_ * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    // Download into host vector.
    void download(std::vector<T>& host) const {
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, size_ * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    // Download a single value.
    T download_scalar() const {
        T val;
        CUDA_CHECK(cudaMemcpy(&val, ptr_, sizeof(T), cudaMemcpyDeviceToHost));
        return val;
    }

    // Zero out.
    void zero() {
        CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
    }

private:
    T*     ptr_  = nullptr;
    size_t size_ = 0;
};

#endif // CUDA_UTILS_CUH
