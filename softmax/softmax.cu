#include <cfloat>
#include <cmath>
#include <random>
#include <cudnn.h>

#define WARP_SIZE 32

float* create_input(int batch, int num_classes) {
    int num_elements = batch * num_classes;

    float* input = new float[num_elements];

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);

    for (int i = 0; i < num_elements; ++i) {
        input[i] = distribution(generator);
    }

    return input;
}

void softmax_cudnn_benchmark(int batch, int num_classes) {
    int num_elements = batch * num_classes;
    size_t bytes = num_elements * sizeof(float);

    float* input = create_input(batch, num_classes);
    float* output = new float[num_elements];

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(
        d_input,
        input,
        bytes,
        cudaMemcpyHostToDevice
    );

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t tensor_desc;

    cudnnCreate(&handle);
    cudnnCreateTensorDescriptor(&tensor_desc);

    // 把 [batch, num_classes] 描述成 [N, C, H, W]
    cudnnSetTensor4dDescriptor(
        tensor_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch,
        num_classes,
        1,
        1
    );

    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnSoftmaxForward(
        handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        tensor_desc,
        d_input,
        &beta,
        tensor_desc,
        d_output
    );

    cudaMemcpy(
        output,
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    );

    cudnnDestroyTensorDescriptor(tensor_desc);
    cudnnDestroy(handle);

    delete[] input;
    delete[] output;

    cudaFree(d_input);
    cudaFree(d_output);
}

// 一个线程处理一个样本的 softmax 计算
__global__ void softmax_naive(const float* input, float* output, int batch, int num_classes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch) {
        return;
    }

    int offset = row * num_classes;

    float row_max = -FLT_MAX;
    for (int i = 0; i < num_classes; ++i) {
        row_max = fmaxf(input[offset + i], row_max);
    }

    float row_sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        row_sum += expf(input[offset + i] - row_max);
    }

    for (int i = 0; i < num_classes; ++i) {
        output[offset + i] =
            expf(input[offset + i] - row_max) / row_sum;
    }
}

template <int BSIZE>
void softmax_naive_launcher(
    const float* input,
    float* output,
    int batch,
    int num_classes
) {
    int blocks = (batch + BSIZE - 1) / BSIZE;
    softmax_naive<<<blocks, BSIZE>>>(input, output, batch, num_classes);
}

template <int BSIZE>
void softmax_naive_benchmark(int batch, int num_classes) {
    float* input = create_input(batch, num_classes);
    float* output = new float[batch * num_classes];

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, batch * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch * num_classes * sizeof(float));

    cudaMemcpy(d_input, input, batch * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_naive_launcher<BSIZE>(d_input, d_output, batch, num_classes);

    cudaMemcpy(output, d_output, batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
}

// 一个线程处理一个样本的 softmax 计算
__global__ void softmax_vectorized(const float* input, float* output, int batch, int num_classes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch) {
        return;
    }

    int offset = row * num_classes;
    const float4 *input4 = reinterpret_cast<const float4*>(input + offset);
    float4* output4 = reinterpret_cast<float4*>(output + offset);
    int vec_num = num_classes / 4;

    float row_max = -FLT_MAX;
    for (int i = 0; i < vec_num; i++) {
        float4 val = input4[i];
        row_max = fmaxf(row_max, val.x);
        row_max = fmaxf(row_max, val.y);
        row_max = fmaxf(row_max, val.z);
        row_max = fmaxf(row_max, val.w);
    }

    float row_sum = 0;
    for (int i = 0; i < vec_num; i++) {
        float4 val = input4[i];
        row_sum += expf(val.x - row_max);
        row_sum += expf(val.y - row_max);
        row_sum += expf(val.z - row_max);
        row_sum += expf(val.w - row_max);
    }

    for (int i = 0; i < vec_num; i++) {
        float4 val = input4[i];
        float4 out;
        out.x = expf(val.x - row_max) / row_sum;
        out.y = expf(val.y - row_max) / row_sum;
        out.z = expf(val.z - row_max) / row_sum;
        out.w = expf(val.w - row_max) / row_sum;
        output4[i] = out;
    }
}

template <int BSIZE>
void softmax_vectorized_launcher(
    const float* input,
    float* output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0) {
        return;
    }

    unsigned long long input_addr =
        reinterpret_cast<unsigned long long>(input);

    unsigned long long output_addr =
        reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int blocks = (batch + BSIZE - 1) / BSIZE;

    softmax_vectorized<<<blocks, BSIZE>>>(
        input,
        output,
        batch,
        num_classes
    );
}

template <int BSIZE>
void softmax_vectorized_benchmark(int batch, int num_classes) {
    float* input = create_input(batch, num_classes);
    float* output = new float[batch * num_classes];

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, batch * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch * num_classes * sizeof(float));

    cudaMemcpy(d_input, input, batch * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_vectorized_launcher<BSIZE>(d_input, d_output, batch, num_classes);

    cudaMemcpy(output, d_output, batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
}

// 一个warp处理一个样本的 softmax 计算
__global__ void softmax_warp(const float *input, float *output, int batch, int num_classes) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * blockDim.x / WARP_SIZE + warp_id;

    if (row >= batch) return;
    int row_offset = row * num_classes;

    float row_tile_max = -FLT_MAX;
    for (int i = lane_id; i < num_classes; i+=WARP_SIZE) {
        row_tile_max = fmaxf(row_tile_max, input[row_offset + i]);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        row_tile_max = fmaxf(row_tile_max, __shfl_xor_sync(0xffffffff, row_tile_max, offset));
    }

    float row_tile_sum = 0;
    for (int i = lane_id; i < num_classes; i+=WARP_SIZE) {
        row_tile_sum += expf(input[row_offset + i] - row_tile_max);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        row_tile_sum += __shfl_xor_sync(0xffffffff, row_tile_sum, offset);
    }

    for (int i = lane_id; i < num_classes; i+=WARP_SIZE) {
        output[row_offset + i] = expf(input[row_offset + i] - row_tile_max) / row_tile_sum;
    }
}

template <int BSIZE>
void softmax_warp_launcher(
    const float* input,
    float* output,
    int batch,
    int num_classes
) {
    int warps_per_block = BSIZE / WARP_SIZE;
    int blocks = (batch + warps_per_block - 1) / warps_per_block;
    softmax_warp<<<blocks, BSIZE>>>(input, output, batch, num_classes);
}

template <int BSIZE>
void softmax_warp_benchmark(int batch, int num_classes) {
    float* input = create_input(batch, num_classes);
    float* output = new float[batch * num_classes];

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, batch * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch * num_classes * sizeof(float));

    cudaMemcpy(d_input, input, batch * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_warp_launcher<BSIZE>(d_input, d_output, batch, num_classes);

    cudaMemcpy(output, d_output, batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void softmax_warp_vectorized(const float *input, float *output, int batch, int num_classes) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * blockDim.x / WARP_SIZE + warp_id;

    if (row >= batch) return;
    int row_offset = row * num_classes;
    const float4 *input4 = reinterpret_cast<const float4*>(input + row_offset);
    float4 *output4 = reinterpret_cast<float4*>(output + row_offset);
    int vec_num = num_classes / 4;

    float row_tile_max = -FLT_MAX;
    for (int i = lane_id; i < vec_num; i+=WARP_SIZE) {
        float4 val = input4[i];
        row_tile_max = fmaxf(row_tile_max, val.x);
        row_tile_max = fmaxf(row_tile_max, val.y);
        row_tile_max = fmaxf(row_tile_max, val.z);
        row_tile_max = fmaxf(row_tile_max, val.w);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        row_tile_max = fmaxf(row_tile_max, __shfl_xor_sync(0xffffffff, row_tile_max, offset));
    }

    float row_tile_sum = 0;
    for (int i = lane_id; i < vec_num; i+=WARP_SIZE) {
        float4 val = input4[i];
        row_tile_sum += expf(val.x - row_tile_max);
        row_tile_sum += expf(val.y - row_tile_max);
        row_tile_sum += expf(val.z - row_tile_max);
        row_tile_sum += expf(val.w - row_tile_max);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        row_tile_sum += __shfl_xor_sync(0xffffffff, row_tile_sum, offset);
    }

    for (int i = lane_id; i < vec_num; i += WARP_SIZE) {
        float4 val = input4[i];
        float4 out;
        out.x = expf(val.x - row_tile_max) / row_tile_sum;
        out.y = expf(val.y - row_tile_max) / row_tile_sum;
        out.z = expf(val.z - row_tile_max) / row_tile_sum;
        out.w = expf(val.w - row_tile_max) / row_tile_sum;
        output4[i] = out;
    }
}

template <int BSIZE>
void softmax_warp_vectorized_launcher(
    const float *input,
    float *output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0) {
        return;
    }

    unsigned long long input_addr = reinterpret_cast<unsigned long long>(input);
    unsigned long long output_addr = reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int warps_per_block = BSIZE / WARP_SIZE;
    int blocks = (batch + warps_per_block - 1) / warps_per_block;

    softmax_warp_vectorized<<<blocks, BSIZE>>>(input, output, batch, num_classes);
}

template <int BSIZE>
void softmax_warp_vectorized_benchmark(int batch, int num_classes) {
    float* input = create_input(batch, num_classes);
    float* output = new float[batch * num_classes];

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, batch * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch * num_classes * sizeof(float));

    cudaMemcpy(d_input, input, batch * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_warp_vectorized_launcher<BSIZE>(d_input, d_output, batch, num_classes);

    cudaMemcpy(output, d_output, batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
}

struct MAX_SUM {
    float max;
    float sum;
};

__device__ __forceinline__ MAX_SUM merge_max_sum(MAX_SUM ms1, MAX_SUM ms2) {
    MAX_SUM ret;

    bool first_is_max = ms1.max >= ms2.max;
    ret.max = fmaxf(ms1.max, ms2.max);
    float scale = expf(-fabsf(ms1.max - ms2.max));
    ret.sum = first_is_max? ms1.sum + ms2.sum * scale : ms1.sum * scale + ms2.sum;

    return ret;
}

__global__ void softmax_warp_vectorized_online(const float *input, float *output, int batch, int num_classes) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * blockDim.x / WARP_SIZE + warp_id;

    if (row >= batch) return;
    int row_offset = row * num_classes;
    const float4 *input4 = reinterpret_cast<const float4*>(input + row_offset);
    float4 *output4 = reinterpret_cast<float4*>(output + row_offset);
    int vec_num = num_classes / 4;

    MAX_SUM row_tile_max_sum = {-FLT_MAX, 0.0f};
    for (int i = lane_id; i < vec_num; i += WARP_SIZE) {
        float4 val = input4[i];

        row_tile_max_sum = merge_max_sum(row_tile_max_sum, MAX_SUM{val.x, 1.0f});
        row_tile_max_sum = merge_max_sum(row_tile_max_sum, MAX_SUM{val.y, 1.0f});
        row_tile_max_sum = merge_max_sum(row_tile_max_sum, MAX_SUM{val.z, 1.0f});
        row_tile_max_sum = merge_max_sum(row_tile_max_sum, MAX_SUM{val.w, 1.0f});
    }
    // for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    //     MAX_SUM other;
    //     other.max = __shfl_xor_sync(0xffffffff, row_tile_max_sum.max, offset);
    //     other.sum = __shfl_xor_sync(0xffffffff, row_tile_max_sum.sum, offset);
    //     row_tile_max_sum = merge_max_sum(row_tile_max_sum, other);
    // }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        MAX_SUM other;
        other.max = __shfl_down_sync(0xffffffff, row_tile_max_sum.max, offset);
        other.sum = __shfl_down_sync(0xffffffff, row_tile_max_sum.sum, offset);
        row_tile_max_sum = merge_max_sum(row_tile_max_sum, other);
    }
    row_tile_max_sum.max = __shfl_sync(0xffffffff, row_tile_max_sum.max, 0);
    row_tile_max_sum.sum = __shfl_sync(0xffffffff, row_tile_max_sum.sum, 0);


    for (int i = lane_id; i < vec_num; i += WARP_SIZE) {
        float4 val = input4[i];
        float4 out;
        out.x = expf(val.x - row_tile_max_sum.max) / row_tile_max_sum.sum;
        out.y = expf(val.y - row_tile_max_sum.max) / row_tile_max_sum.sum;
        out.z = expf(val.z - row_tile_max_sum.max) / row_tile_max_sum.sum;
        out.w = expf(val.w - row_tile_max_sum.max) / row_tile_max_sum.sum;
        output4[i] = out;
    }
}

template <int BSIZE>
void softmax_warp_vectorized_online_launcher(
    const float *input,
    float *output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0) {
        return;
    }

    unsigned long long input_addr = reinterpret_cast<unsigned long long>(input);
    unsigned long long output_addr = reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int warps_per_block = BSIZE / WARP_SIZE;
    int blocks = (batch + warps_per_block - 1) / warps_per_block;

    softmax_warp_vectorized_online<<<blocks, BSIZE>>>(input, output, batch, num_classes);
}

template <int BSIZE>
void softmax_warp_vectorized_online_benchmark(int batch, int num_classes) {
    float* input = create_input(batch, num_classes);
    float* output = new float[batch * num_classes];

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, batch * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch * num_classes * sizeof(float));

    cudaMemcpy(d_input, input, batch * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_warp_vectorized_online_launcher<BSIZE>(d_input, d_output, batch, num_classes);

    cudaMemcpy(output, d_output, batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void softmax_block_vectorized_shared(const float* input, float* output, int batch, int num_classes) {
    int row_idx = blockIdx.x;
    if (row_idx >= batch) return;

    int local_thread_idx = threadIdx.x;
    int warp_idx = local_thread_idx / WARP_SIZE;
    int lane_idx = local_thread_idx % WARP_SIZE;
    int num_warps_per_block = blockDim.x / WARP_SIZE;
    
    int row_offset = row_idx * num_classes;
    int num_vec4 = num_classes / 4;

    extern __shared__ float4 shared_vec4[];
    float4 *shared_row = shared_vec4;
    float *shared_warp = reinterpret_cast<float*>(shared_vec4 + num_vec4);

    const float4 *input4 = reinterpret_cast<const float4*>(input + row_offset);
    float4 *output4 = reinterpret_cast<float4*>(output + row_offset);

    // 数据加载：global -> shared
    for (int i = local_thread_idx; i < num_vec4; i+=blockDim.x) {
        shared_row[i] = input4[i];
    }
    // 保证数据完整加载
    __syncthreads();

    // 计算 max
    float thread_max = -FLT_MAX;
    for (int i = local_thread_idx; i < num_vec4; i+=blockDim.x) {
        float4 val4 = shared_row[i];
        thread_max = fmaxf(thread_max, val4.x);
        thread_max = fmaxf(thread_max, val4.y);
        thread_max = fmaxf(thread_max, val4.z);
        thread_max = fmaxf(thread_max, val4.w);
    }
    // warp内规约
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    // 将warp_max写入shared
    if (lane_idx == 0) shared_warp[warp_idx] = thread_max;
    // 保证warp_max完整写入
    __syncthreads();
    // warp间规约
    if (warp_idx == 0) {
        thread_max = -FLT_MAX;
        if (lane_idx < num_warps_per_block) {
            thread_max = shared_warp[lane_idx];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
        if (lane_idx == 0) shared_warp[0] = thread_max;
    }
    // 保证max写入shared后再进行读取
    __syncthreads();
    // 读取max
    thread_max = shared_warp[0];
    // 保证所有线程读取完成，再复用shared_warp
    __syncthreads();

    // 计算 sum
    float thread_sum = 0;
    for (int i = local_thread_idx; i < num_vec4; i+=blockDim.x) {
        float4 val4 = shared_row[i];
        val4.x = expf(val4.x - thread_max);
        val4.y = expf(val4.y - thread_max);
        val4.z = expf(val4.z - thread_max);
        val4.w = expf(val4.w - thread_max);
        thread_sum += val4.x + val4.y + val4.z + val4.w;
        shared_row[i] = val4;
    }
    // warp内规约
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    // 将warp_sum写入shared
    if (lane_idx == 0) shared_warp[warp_idx] = thread_sum;
    // 保证warp_sum完整写入
    __syncthreads();
    // warp间规约
    if (warp_idx == 0) {
        thread_sum = 0;
        if (lane_idx < num_warps_per_block) {
            thread_sum = shared_warp[lane_idx];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (lane_idx == 0) shared_warp[0] = thread_sum;
    }
    // 保证sum写入shared后再进行读取
    __syncthreads();
    // 读取sum
    thread_sum = shared_warp[0];

    // 写回
    for (int i = local_thread_idx; i < num_vec4; i+=blockDim.x) {
        float4 val4 = shared_row[i];
        val4.x = val4.x / thread_sum;
        val4.y = val4.y / thread_sum;
        val4.z = val4.z / thread_sum;
        val4.w = val4.w / thread_sum;
        output4[i] = val4;
    }
}

template <int BSIZE>
void softmax_block_vectorized_shared_launcher(
    const float *input,
    float *output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0 || num_classes > 4096) {
        return;
    }

    unsigned long long input_addr = reinterpret_cast<unsigned long long>(input);
    unsigned long long output_addr = reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int num_vec4 = num_classes / 4;
    int warps_per_block = BSIZE / WARP_SIZE;
    int shared_bytes = num_vec4 * sizeof(float4) + warps_per_block * sizeof(float);

    softmax_block_vectorized_shared<<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
}

template <int BSIZE>
void softmax_block_vectorized_shared_benchmark(int batch, int num_classes) {
    float* input = create_input(batch, num_classes);
    float* output = new float[batch * num_classes];

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, batch * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch * num_classes * sizeof(float));

    cudaMemcpy(d_input, input, batch * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_block_vectorized_shared_launcher<BSIZE>(d_input, d_output, batch, num_classes);

    cudaMemcpy(output, d_output, batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
}

template <int MAX_NUM_CLASSES, int BSIZE>
__global__ void softmax_block_vectorized_register(const float* input, float* output, int batch, int num_classes) {
    int row_idx = blockIdx.x;
    if (row_idx >= batch) return;

    int local_thread_idx = threadIdx.x;
    int warp_idx = local_thread_idx / WARP_SIZE;
    int lane_idx = local_thread_idx % WARP_SIZE;
    int num_warps_per_block = blockDim.x / WARP_SIZE;
    
    int row_offset = row_idx * num_classes;
    const float4 *input4 = reinterpret_cast<const float4*>(input + row_offset);
    float4 *output4 = reinterpret_cast<float4*>(output + row_offset);

    constexpr int MAX_NUM_VEC4 = MAX_NUM_CLASSES / 4;
    constexpr int MAX_VECS_PER_THREAD = (MAX_NUM_VEC4 + BSIZE - 1) / BSIZE;
    float4 thread_input4[MAX_VECS_PER_THREAD];
    int num_vec4 = num_classes / 4;
    extern __shared__ float shared_warp[];

    // 数据加载：global -> register
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; reg_idx++) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;
        if (vec_idx < num_vec4) thread_input4[reg_idx] = input4[vec_idx];
    }

    // 计算 max
    // 线程内部计算max
    float thread_max = -FLT_MAX;
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; reg_idx++) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;
        if (vec_idx < num_vec4) {
            float4 val4 = thread_input4[reg_idx];
            thread_max = fmaxf(thread_max, val4.x);
            thread_max = fmaxf(thread_max, val4.y);
            thread_max = fmaxf(thread_max, val4.z);
            thread_max = fmaxf(thread_max, val4.w);
        }
    }
    // warp内部计算max
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    // 将warp_max写入shared
    if (lane_idx == 0) shared_warp[warp_idx] = thread_max;
    // 保证warp_max完整写入
    __syncthreads();
    // warp间规约
    if (warp_idx == 0) {
        thread_max = -FLT_MAX;
        if (lane_idx < num_warps_per_block) {
            thread_max = shared_warp[lane_idx];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
        if (lane_idx == 0) shared_warp[0] = thread_max;
    }
    // 保证max写入shared后再进行读取
    __syncthreads();
    // 读取max
    thread_max = shared_warp[0];
    // 保证所有线程读取完成，再复用shared_warp
    __syncthreads();

    // 计算 sum
    // 线程内部计算sum
    float thread_sum = 0;
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; reg_idx++) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;
        if (vec_idx < num_vec4) {
            float4 val4 = thread_input4[reg_idx];
            val4.x = expf(val4.x - thread_max);
            val4.y = expf(val4.y - thread_max);
            val4.z = expf(val4.z - thread_max);
            val4.w = expf(val4.w - thread_max);
            thread_sum += val4.x + val4.y + val4.z + val4.w;
            thread_input4[reg_idx] = val4;
        }
    }
    // warp内部计算sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    // 将warp_sum写入shared
    if (lane_idx == 0) shared_warp[warp_idx] = thread_sum;
    // 保证warp_sum完整写入
    __syncthreads();
    // warp间规约
    if (warp_idx == 0) {
        thread_sum = 0;
        if (lane_idx < num_warps_per_block) {
            thread_sum = shared_warp[lane_idx];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (lane_idx == 0) shared_warp[0] = thread_sum;
    }
    // 保证sum写入shared后再进行读取
    __syncthreads();
    // 读取sum
    thread_sum = shared_warp[0];

    // 写回
    #pragma unroll
    for (int reg_idx = 0; reg_idx < MAX_VECS_PER_THREAD; reg_idx++) {
        int vec_idx = local_thread_idx + reg_idx * BSIZE;
        if (vec_idx < num_vec4) {
            float4 val4 = thread_input4[reg_idx];
            val4.x = val4.x / thread_sum;
            val4.y = val4.y / thread_sum;
            val4.z = val4.z / thread_sum;
            val4.w = val4.w / thread_sum;
            output4[vec_idx] = val4;
        }
    }
}

template <int BSIZE>
void softmax_block_vectorized_register_launcher(
    const float *input,
    float *output,
    int batch,
    int num_classes
) {
    if (num_classes % 4 != 0 || num_classes > 4096) {
        return;
    }

    unsigned long long input_addr = reinterpret_cast<unsigned long long>(input);
    unsigned long long output_addr = reinterpret_cast<unsigned long long>(output);

    if (input_addr % alignof(float4) != 0) {
        return;
    }

    if (output_addr % alignof(float4) != 0) {
        return;
    }

    int warps_per_block = BSIZE / WARP_SIZE;
    int shared_bytes = warps_per_block * sizeof(float);

    if (num_classes <= 128) {
        softmax_block_vectorized_register<128, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 256) {
        softmax_block_vectorized_register<256, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 512) {
        softmax_block_vectorized_register<512, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 1024) {
        softmax_block_vectorized_register<1024, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 2048) {
        softmax_block_vectorized_register<2048, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else if (num_classes <= 4096) {
        softmax_block_vectorized_register<4096, BSIZE><<<batch, BSIZE, shared_bytes>>>(input, output, batch, num_classes);
    } else {
        return;
    }
}

template <int BSIZE>
void softmax_block_vectorized_register_benchmark(int batch, int num_classes) {
    float* input = create_input(batch, num_classes);
    float* output = new float[batch * num_classes];

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, batch * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch * num_classes * sizeof(float));

    cudaMemcpy(d_input, input, batch * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_block_vectorized_register_launcher<BSIZE>(d_input, d_output, batch, num_classes);

    cudaMemcpy(output, d_output, batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char *argv[]) {
    int batch = std::atoi(argv[1]);
    int num_classes = std::atoi(argv[2]);
    softmax_cudnn_benchmark(batch, num_classes);
    softmax_naive_benchmark<256>(batch, num_classes);
    softmax_vectorized_benchmark<256>(batch, num_classes);
    softmax_warp_benchmark<256>(batch, num_classes);
    softmax_warp_vectorized_benchmark<256>(batch, num_classes);
    softmax_warp_vectorized_online_benchmark<256>(batch, num_classes);
    softmax_block_vectorized_shared_benchmark<128>(batch, num_classes);
    softmax_block_vectorized_shared_benchmark<256>(batch, num_classes);
    softmax_block_vectorized_register_benchmark<128>(batch, num_classes);
    softmax_block_vectorized_register_benchmark<256>(batch, num_classes);

    return 0;
}