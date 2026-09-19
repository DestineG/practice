#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define WARP_SIZE 32

static void check_cuda(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

static void check_cublas(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", file, line, (int)status);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) check_cuda((call), __FILE__, __LINE__)
#define CUBLAS_CHECK(call) check_cublas((call), __FILE__, __LINE__)

constexpr int GEMM_BM = 128;
constexpr int GEMM_BN = 128;
constexpr int GEMM_BK = 64;
constexpr int GEMM_WM = 32;
constexpr int GEMM_WN = 64;
constexpr int GEMM_THREADS = 256;

__global__ void gemm_naive(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
    C[row * N + col] = sum;
}

template <bool SWIZZLE>
__device__ __forceinline__ int shared_offset(int row, int logical_col, int leading_dimension) {
    if (!SWIZZLE) return row * leading_dimension + logical_col;
    int segment = logical_col >> 6;
    int segment_col = logical_col & 63;
    int chunk = segment_col >> 3;
    int in_chunk = segment_col & 7;
    int physical_chunk = (chunk + (row & 7)) & 7;
    return row * leading_dimension + segment * 64 + physical_chunk * 8 + in_chunk;
}

template <int BM, int BN, int BK, bool SWIZZLE>
__device__ __forceinline__ void load_AB_tile_from_global_to_shared(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* As, __nv_bfloat16* Bs, int row_base, int col_base, int k_base, int M, int N, int K) {
    int thread = threadIdx.x;
    constexpr int A_CHUNKS_PER_ROW = BK / 8;
    constexpr int B_CHUNKS_PER_ROW = BN / 8;
    constexpr int A_CHUNKS = BM * A_CHUNKS_PER_ROW;
    constexpr int B_CHUNKS = BK * B_CHUNKS_PER_ROW;

    #pragma unroll
    for (int linear_chunk = thread; linear_chunk < A_CHUNKS; linear_chunk += blockDim.x) {
        int row = linear_chunk / A_CHUNKS_PER_ROW;
        int chunk = linear_chunk % A_CHUNKS_PER_ROW;
        int col = chunk * 8;
        uint4 value = *reinterpret_cast<const uint4*>(A + (row_base + row) * K + k_base + col);
        int destination = shared_offset<SWIZZLE>(row, col, BK);
        *reinterpret_cast<uint4*>(As + destination) = value;
    }

    #pragma unroll
    for (int linear_chunk = thread; linear_chunk < B_CHUNKS; linear_chunk += blockDim.x) {
        int row = linear_chunk / B_CHUNKS_PER_ROW;
        int chunk = linear_chunk % B_CHUNKS_PER_ROW;
        int col = chunk * 8;
        uint4 value = *reinterpret_cast<const uint4*>(B + (k_base + row) * N + col_base + col);
        int destination = shared_offset<SWIZZLE>(row, col, BN);
        *reinterpret_cast<uint4*>(Bs + destination) = value;
    }
}

__device__ __forceinline__ void cp_async_16(void* shared_destination, const void* global_source) {
    uint32_t shared_address = (uint32_t)__cvta_generic_to_shared(shared_destination);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(shared_address), "l"(global_source));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

template <int BM, int BN, int BK, bool SWIZZLE>
__device__ __forceinline__ void load_AB_tile_from_global_to_shared_async(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* As, __nv_bfloat16* Bs, int row_base, int col_base, int k_base, int N, int K) {
    int thread = threadIdx.x;
    constexpr int A_CHUNKS_PER_ROW = BK / 8;
    constexpr int B_CHUNKS_PER_ROW = BN / 8;
    constexpr int A_CHUNKS = BM * A_CHUNKS_PER_ROW;
    constexpr int B_CHUNKS = BK * B_CHUNKS_PER_ROW;

    #pragma unroll
    for (int linear_chunk = thread; linear_chunk < A_CHUNKS; linear_chunk += blockDim.x) {
        int row = linear_chunk / A_CHUNKS_PER_ROW;
        int chunk = linear_chunk % A_CHUNKS_PER_ROW;
        int col = chunk * 8;
        int destination = shared_offset<SWIZZLE>(row, col, BK);
        cp_async_16(As + destination, A + (row_base + row) * K + k_base + col);
    }

    #pragma unroll
    for (int linear_chunk = thread; linear_chunk < B_CHUNKS; linear_chunk += blockDim.x) {
        int row = linear_chunk / B_CHUNKS_PER_ROW;
        int chunk = linear_chunk % B_CHUNKS_PER_ROW;
        int col = chunk * 8;
        int destination = shared_offset<SWIZZLE>(row, col, BN);
        cp_async_16(Bs + destination, B + (k_base + row) * N + col_base + col);
    }
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* registers, const __nv_bfloat16* address) {
    uint32_t shared_address = (uint32_t)__cvta_generic_to_shared(address);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]), "=r"(registers[3]) : "r"(shared_address));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* registers, const __nv_bfloat16* address) {
    uint32_t shared_address = (uint32_t)__cvta_generic_to_shared(address);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" : "=r"(registers[0]), "=r"(registers[1]) : "r"(shared_address));
}

template <int BK, bool SWIZZLE>
__device__ __forceinline__ void load_A_tile_from_shared_to_registers(const __nv_bfloat16* As, uint32_t* A_r, int warp_row, int warp_mma_m, int mma_k, int lane) {
    int matrix = lane >> 3;
    int matrix_row = lane & 7;
    int row = warp_row + warp_mma_m * 16 + (matrix & 1) * 8 + matrix_row;
    int col = mma_k * 16 + (matrix >> 1) * 8;
    ldmatrix_x4(A_r, As + shared_offset<SWIZZLE>(row, col, BK));
}

template <int BN, bool SWIZZLE>
__device__ __forceinline__ void load_B_tile_from_shared_to_registers(const __nv_bfloat16* Bs, uint32_t* B_r, int warp_col, int warp_mma_n, int mma_k, int lane) {
    int matrix = (lane >> 3) & 1;
    int matrix_row = lane & 7;
    int row = mma_k * 16 + matrix * 8 + matrix_row;
    int col = warp_col + warp_mma_n * 8;
    ldmatrix_x2_trans(B_r, Bs + shared_offset<SWIZZLE>(row, col, BN));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(const uint32_t* A_r, const uint32_t* B_r, float* C_r) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(C_r[0]), "+f"(C_r[1]), "+f"(C_r[2]), "+f"(C_r[3])
        : "r"(A_r[0]), "r"(A_r[1]), "r"(A_r[2]), "r"(A_r[3]), "r"(B_r[0]), "r"(B_r[1]));
}

template <int BM, int BN, int WM, int WN>
__device__ __forceinline__ void store_C_tile_from_registers_to_global(const float* C_r, float* C, int row_base, int col_base, int N, int warp_row, int warp_col, int lane) {
    constexpr int NUM_WARP_MMA_M = WM / 16;
    constexpr int NUM_WARP_MMA_N = WN / 8;
    int row_in_fragment = lane >> 2;
    int col_in_fragment = (lane & 3) * 2;
    #pragma unroll
    for (int warp_mma_m = 0; warp_mma_m < NUM_WARP_MMA_M; ++warp_mma_m) {
        #pragma unroll
        for (int warp_mma_n = 0; warp_mma_n < NUM_WARP_MMA_N; ++warp_mma_n) {
            int fragment = warp_mma_m * NUM_WARP_MMA_N + warp_mma_n;
            const float* values = C_r + fragment * 4;
            int row = row_base + warp_row + warp_mma_m * 16 + row_in_fragment;
            int col = col_base + warp_col + warp_mma_n * 8 + col_in_fragment;
            *reinterpret_cast<float2*>(C + row * N + col) = make_float2(values[0], values[1]);
            *reinterpret_cast<float2*>(C + (row + 8) * N + col) = make_float2(values[2], values[3]);
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, bool SWIZZLE, bool ASYNC>
__global__ void gemm_bf16_align(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
    static_assert(BM == 128 && BN == 128 && BK == 64 && WM == 32 && WN == 64, "first aligned kernel supports only the requested tile");
    constexpr int A_STAGE_ELEMENTS = BM * BK;
    constexpr int B_STAGE_ELEMENTS = BK * BN;
    constexpr int STAGE_ELEMENTS = A_STAGE_ELEMENTS + B_STAGE_ELEMENTS;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    __nv_bfloat16* shared = reinterpret_cast<__nv_bfloat16*>(shared_raw);

    int row_base = blockIdx.y * BM;
    int col_base = blockIdx.x * BN;
    int thread = threadIdx.x;
    int warp = thread / WARP_SIZE;
    int lane = thread % WARP_SIZE;
    constexpr int NUM_WARP_N = BN / WN;
    int warp_m = warp / NUM_WARP_N;
    int warp_n = warp % NUM_WARP_N;
    int warp_row = warp_m * WM;
    int warp_col = warp_n * WN;
    constexpr int NUM_WARP_MMA_M = WM / 16;
    constexpr int NUM_WARP_MMA_N = WN / 8;
    constexpr int NUM_MMA_K = BK / 16;

    uint32_t A_r[NUM_WARP_MMA_M][4];
    uint32_t B_r[NUM_WARP_MMA_N][2];
    float C_r[NUM_WARP_MMA_M * NUM_WARP_MMA_N][4] = {};

    int tile_count = K / BK;
    int read_stage = 0;
    int write_stage = ASYNC ? 1 : 0;

    if (ASYNC) {
        __nv_bfloat16* As_write = shared;
        __nv_bfloat16* Bs_write = As_write + A_STAGE_ELEMENTS;
        load_AB_tile_from_global_to_shared_async<BM, BN, BK, SWIZZLE>(A, B, As_write, Bs_write, row_base, col_base, 0, N, K);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
    }

    for (int k_tile = 0; k_tile < tile_count; ++k_tile) {
        __nv_bfloat16* As_read = shared + read_stage * STAGE_ELEMENTS;
        __nv_bfloat16* Bs_read = As_read + A_STAGE_ELEMENTS;

        if (!ASYNC) {
            load_AB_tile_from_global_to_shared<BM, BN, BK, SWIZZLE>(A, B, As_read, Bs_read, row_base, col_base, k_tile * BK, M, N, K);
            __syncthreads();
        } else if (k_tile + 1 < tile_count) {
            __nv_bfloat16* As_write = shared + write_stage * STAGE_ELEMENTS;
            __nv_bfloat16* Bs_write = As_write + A_STAGE_ELEMENTS;
            load_AB_tile_from_global_to_shared_async<BM, BN, BK, SWIZZLE>(A, B, As_write, Bs_write, row_base, col_base, (k_tile + 1) * BK, N, K);
            cp_async_commit();
        }

        #pragma unroll
        for (int mma_k = 0; mma_k < NUM_MMA_K; ++mma_k) {
            #pragma unroll
            for (int warp_mma_m = 0; warp_mma_m < NUM_WARP_MMA_M; ++warp_mma_m) load_A_tile_from_shared_to_registers<BK, SWIZZLE>(As_read, A_r[warp_mma_m], warp_row, warp_mma_m, mma_k, lane);
            #pragma unroll
            for (int warp_mma_n = 0; warp_mma_n < NUM_WARP_MMA_N; ++warp_mma_n) load_B_tile_from_shared_to_registers<BN, SWIZZLE>(Bs_read, B_r[warp_mma_n], warp_col, warp_mma_n, mma_k, lane);
            #pragma unroll
            for (int warp_mma_m = 0; warp_mma_m < NUM_WARP_MMA_M; ++warp_mma_m) {
                #pragma unroll
                for (int warp_mma_n = 0; warp_mma_n < NUM_WARP_MMA_N; ++warp_mma_n) {
                    int fragment = warp_mma_m * NUM_WARP_MMA_N + warp_mma_n;
                    mma_m16n8k16_bf16(A_r[warp_mma_m], B_r[warp_mma_n], C_r[fragment]);
                }
            }
        }

        if (!ASYNC) {
            __syncthreads();
        } else if (k_tile + 1 < tile_count) {
            cp_async_wait_all();
            __syncthreads();
            int old_read_stage = read_stage;
            read_stage = write_stage;
            write_stage = old_read_stage;
        }
    }

    store_C_tile_from_registers_to_global<BM, BN, WM, WN>(&C_r[0][0], C, row_base, col_base, N, warp_row, warp_col, lane);
}

enum KernelKind { KERNEL_CUBLAS, KERNEL_NAIVE, KERNEL_ROW_MAJOR, KERNEL_SWIZZLE, KERNEL_ASYNC };

struct TestContext {
    int M;
    int N;
    int K;
    size_t a_count;
    size_t b_count;
    size_t c_count;
    __nv_bfloat16* A;
    __nv_bfloat16* B;
    float* output;
    float* reference;
    __nv_bfloat16* d_A;
    __nv_bfloat16* d_B;
    float* d_C;
    cublasHandle_t cublas;

    TestContext(int M_value, int N_value, int K_value) : M(M_value), N(N_value), K(K_value), a_count((size_t)M * K), b_count((size_t)K * N), c_count((size_t)M * N), A((__nv_bfloat16*)malloc(a_count * sizeof(__nv_bfloat16))), B((__nv_bfloat16*)malloc(b_count * sizeof(__nv_bfloat16))), output((float*)malloc(c_count * sizeof(float))), reference((float*)malloc(c_count * sizeof(float))), d_A(nullptr), d_B(nullptr), d_C(nullptr), cublas(nullptr) {
        if (A == nullptr || B == nullptr || output == nullptr || reference == nullptr) { fprintf(stderr, "Host allocation failed\n"); exit(EXIT_FAILURE); }
        for (size_t i = 0; i < a_count; ++i) A[i] = __float2bfloat16((float)((int)((i * 17u + 3u) % 101u) - 50) / 100.0f);
        for (size_t i = 0; i < b_count; ++i) B[i] = __float2bfloat16((float)((int)((i * 29u + 7u) % 103u) - 51) / 100.0f);
        CUDA_CHECK(cudaMalloc(&d_A, a_count * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_B, b_count * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_C, c_count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, A, a_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B, b_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasCreate(&cublas));
    }

    ~TestContext() {
        if (cublas != nullptr) cublasDestroy(cublas);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(A);
        free(B);
        free(output);
        free(reference);
    }

    TestContext(const TestContext&) = delete;
    TestContext& operator=(const TestContext&) = delete;
};

static void launch_cublas(TestContext& context) {
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(context.cublas, CUBLAS_OP_N, CUBLAS_OP_N, context.N, context.M, context.K, &alpha, context.d_B, CUDA_R_16BF, context.N, context.d_A, CUDA_R_16BF, context.K, &beta, context.d_C, CUDA_R_32F, context.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void launch_kernel(TestContext& context, KernelKind kind) {
    if (kind == KERNEL_CUBLAS) {
        launch_cublas(context);
    } else if (kind == KERNEL_NAIVE) {
        dim3 block(16, 16);
        dim3 grid((context.N + 15) / 16, (context.M + 15) / 16);
        gemm_naive<<<grid, block>>>(context.d_A, context.d_B, context.d_C, context.M, context.N, context.K);
    } else {
        dim3 grid(context.N / GEMM_BN, context.M / GEMM_BM);
        size_t shared_bytes = (size_t)(GEMM_BM * GEMM_BK + GEMM_BK * GEMM_BN) * sizeof(__nv_bfloat16);
        if (kind == KERNEL_ROW_MAJOR) {
            gemm_bf16_align<GEMM_BM, GEMM_BN, GEMM_BK, GEMM_WM, GEMM_WN, false, false><<<grid, GEMM_THREADS, shared_bytes>>>(context.d_A, context.d_B, context.d_C, context.M, context.N, context.K);
        } else if (kind == KERNEL_SWIZZLE) {
            gemm_bf16_align<GEMM_BM, GEMM_BN, GEMM_BK, GEMM_WM, GEMM_WN, true, false><<<grid, GEMM_THREADS, shared_bytes>>>(context.d_A, context.d_B, context.d_C, context.M, context.N, context.K);
        } else {
            shared_bytes *= 2;
            CUDA_CHECK(cudaFuncSetAttribute(gemm_bf16_align<GEMM_BM, GEMM_BN, GEMM_BK, GEMM_WM, GEMM_WN, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes));
            gemm_bf16_align<GEMM_BM, GEMM_BN, GEMM_BK, GEMM_WM, GEMM_WN, true, true><<<grid, GEMM_THREADS, shared_bytes>>>(context.d_A, context.d_B, context.d_C, context.M, context.N, context.K);
        }
    }
}

struct ErrorResult { double max_absolute; double mean_absolute; double relative_l2; double max_relative; };

static ErrorResult compare_results(const float* actual, const float* reference, size_t count) {
    ErrorResult result = {0.0, 0.0, 0.0, 0.0};
    double square_error = 0.0;
    double square_reference = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double difference = fabs((double)actual[i] - (double)reference[i]);
        double reference_abs = fabs((double)reference[i]);
        if (difference > result.max_absolute) result.max_absolute = difference;
        result.mean_absolute += difference;
        square_error += difference * difference;
        square_reference += (double)reference[i] * (double)reference[i];
        if (reference_abs > 1e-6) {
            double relative = difference / reference_abs;
            if (relative > result.max_relative) result.max_relative = relative;
        }
    }
    result.mean_absolute /= (double)count;
    result.relative_l2 = sqrt(square_error / square_reference);
    return result;
}

static const char* kernel_name(KernelKind kind) {
    if (kind == KERNEL_CUBLAS) return "cublas";
    if (kind == KERNEL_NAIVE) return "naive";
    if (kind == KERNEL_ROW_MAJOR) return "rowmajor";
    if (kind == KERNEL_SWIZZLE) return "swizzle";
    return "async";
}

static void make_reference(TestContext& context) {
    launch_cublas(context);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(context.reference, context.d_C, context.c_count * sizeof(float), cudaMemcpyDeviceToHost));
}

static bool verify_kernel(TestContext& context, KernelKind kind) {
    launch_kernel(context, kind);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(context.output, context.d_C, context.c_count * sizeof(float), cudaMemcpyDeviceToHost));
    ErrorResult error = compare_results(context.output, context.reference, context.c_count);
    bool passed = error.max_absolute <= 1e-3 && error.relative_l2 <= 1e-5;
    printf("%-12s %-4s max_abs=% .6e mean_abs=% .6e rel_l2=% .6e max_rel=% .6e\n", kernel_name(kind), passed ? "PASS" : "FAIL", error.max_absolute, error.mean_absolute, error.relative_l2, error.max_relative);
    return passed;
}

static float benchmark_kernel(TestContext& context, KernelKind kind, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) launch_kernel(context, kind);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch_kernel(context, kind);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return milliseconds / (float)iterations;
}

static bool parse_positive_int(const char* text, int* value) {
    errno = 0;
    char* end = nullptr;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX) return false;
    *value = (int)parsed;
    return true;
}

static bool parse_kernel(const char* text, KernelKind* kind) {
    if (strcmp(text, "cublas") == 0) *kind = KERNEL_CUBLAS;
    else if (strcmp(text, "naive") == 0) *kind = KERNEL_NAIVE;
    else if (strcmp(text, "rowmajor") == 0) *kind = KERNEL_ROW_MAJOR;
    else if (strcmp(text, "swizzle") == 0) *kind = KERNEL_SWIZZLE;
    else if (strcmp(text, "async") == 0) *kind = KERNEL_ASYNC;
    else return false;
    return true;
}

static void print_usage(const char* program) {
    fprintf(stderr, "Usage:\n  %s verify <all|cublas|naive|rowmajor|swizzle|async> M N K\n  %s benchmark <all|cublas|naive|rowmajor|swizzle|async> M N K [warmup] [iterations]\n  %s profile <cublas|naive|rowmajor|swizzle|async> M N K\n", program, program, program);
}

int main(int argc, char** argv) {
    if (argc < 6 || argc > 8) { print_usage(argv[0]); return EXIT_FAILURE; }
    const char* mode = argv[1];
    const char* target = argv[2];
    int M = 0;
    int N = 0;
    int K = 0;
    if (!parse_positive_int(argv[3], &M) || !parse_positive_int(argv[4], &N) || !parse_positive_int(argv[5], &K)) { print_usage(argv[0]); return EXIT_FAILURE; }
    if (M % GEMM_BM != 0 || N % GEMM_BN != 0 || K % GEMM_BK != 0) { fprintf(stderr, "Aligned kernel requires M%%128=0, N%%128=0, K%%64=0\n"); return EXIT_FAILURE; }
    int warmup = 10;
    int iterations = 100;
    if (argc >= 7 && !parse_positive_int(argv[6], &warmup)) { print_usage(argv[0]); return EXIT_FAILURE; }
    if (argc >= 8 && !parse_positive_int(argv[7], &iterations)) { print_usage(argv[0]); return EXIT_FAILURE; }
    bool all = strcmp(target, "all") == 0;
    KernelKind selected = KERNEL_CUBLAS;
    if (!all && !parse_kernel(target, &selected)) { print_usage(argv[0]); return EXIT_FAILURE; }

    TestContext context(M, N, K);
    if (strcmp(mode, "verify") == 0) {
        if (argc != 6) { print_usage(argv[0]); return EXIT_FAILURE; }
        make_reference(context);
        bool passed = true;
        KernelKind kernels[] = {KERNEL_CUBLAS, KERNEL_NAIVE, KERNEL_ROW_MAJOR, KERNEL_SWIZZLE, KERNEL_ASYNC};
        for (int i = 0; i < 5; ++i) if (all || selected == kernels[i]) passed = verify_kernel(context, kernels[i]) && passed;
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (strcmp(mode, "benchmark") == 0) {
        KernelKind kernels[] = {KERNEL_CUBLAS, KERNEL_NAIVE, KERNEL_ROW_MAJOR, KERNEL_SWIZZLE, KERNEL_ASYNC};
        printf("M=%d N=%d K=%d warmup=%d iterations=%d\n", M, N, K, warmup, iterations);
        printf("%-12s %12s %12s\n", "Kernel", "Time (ms)", "TFLOPS");
        for (int i = 0; i < 5; ++i) {
            if (!all && selected != kernels[i]) continue;
            float milliseconds = benchmark_kernel(context, kernels[i], warmup, iterations);
            double tflops = 2.0 * (double)M * (double)N * (double)K / ((double)milliseconds * 1e9);
            printf("%-12s %12.4f %12.3f\n", kernel_name(kernels[i]), milliseconds, tflops);
        }
        return EXIT_SUCCESS;
    }
    if (strcmp(mode, "profile") == 0 && !all && argc == 6) {
        launch_kernel(context, selected);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return EXIT_SUCCESS;
    }
    print_usage(argv[0]);
    return EXIT_FAILURE;
}
