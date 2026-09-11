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

// 一个block处理一行，一行先加载到shared再进行处理
__global__ void softmax_warp_vectorized_online_shared(
    const float* input,
    float* output,
    int batch,
    int num_classes
) {
    int row = blockIdx.x;

    if (row >= batch) {
        return;
    }

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    int row_offset = row * num_classes;
    int vec_num = num_classes / 4;

    const float4* input4 =
        reinterpret_cast<const float4*>(input + row_offset);

    float4* output4 =
        reinterpret_cast<float4*>(output + row_offset);

    /*
     * Shared Memory布局：
     *
     * shared_vec[0 ... vec_num-1]
     *     保存一整行输入
     *
     * warp_max_sum[0 ... warps_per_block-1]
     *     保存每个warp的局部归约结果
     */
    extern __shared__ float4 shared_vec[];

    MAX_SUM* warp_max_sum =
        reinterpret_cast<MAX_SUM*>(shared_vec + vec_num);

    // 1. 一个block协作把一整行加载到Shared Memory
    for (int i = tid; i < vec_num; i += blockDim.x) {
        shared_vec[i] = input4[i];
    }

    __syncthreads();

    // 2. 每个线程处理Shared Memory中的一部分数据
    MAX_SUM thread_max_sum = {-FLT_MAX, 0.0f};

    for (int i = tid; i < vec_num; i += blockDim.x) {
        float4 val = shared_vec[i];

        thread_max_sum = merge_max_sum(
            thread_max_sum,
            MAX_SUM{val.x, 1.0f}
        );

        thread_max_sum = merge_max_sum(
            thread_max_sum,
            MAX_SUM{val.y, 1.0f}
        );

        thread_max_sum = merge_max_sum(
            thread_max_sum,
            MAX_SUM{val.z, 1.0f}
        );

        thread_max_sum = merge_max_sum(
            thread_max_sum,
            MAX_SUM{val.w, 1.0f}
        );
    }

    // 3. 每个warp内部归约
    for (int offset = WARP_SIZE / 2;
         offset > 0;
         offset >>= 1) {

        MAX_SUM other;

        other.max = __shfl_down_sync(
            0xffffffff,
            thread_max_sum.max,
            offset
        );

        other.sum = __shfl_down_sync(
            0xffffffff,
            thread_max_sum.sum,
            offset
        );

        thread_max_sum = merge_max_sum(
            thread_max_sum,
            other
        );
    }

    // 每个warp的lane 0写入Shared Memory
    if (lane_id == 0) {
        warp_max_sum[warp_id] = thread_max_sum;
    }

    __syncthreads();

    // 4. 第一个warp负责合并所有warp的结果
    if (warp_id == 0) {
        MAX_SUM block_max_sum = {-FLT_MAX, 0.0f};

        if (lane_id < warps_per_block) {
            block_max_sum = warp_max_sum[lane_id];
        }

        for (int offset = WARP_SIZE / 2;
             offset > 0;
             offset >>= 1) {

            MAX_SUM other;

            other.max = __shfl_down_sync(
                0xffffffff,
                block_max_sum.max,
                offset
            );

            other.sum = __shfl_down_sync(
                0xffffffff,
                block_max_sum.sum,
                offset
            );

            block_max_sum = merge_max_sum(
                block_max_sum,
                other
            );
        }

        if (lane_id == 0) {
            warp_max_sum[0] = block_max_sum;
        }
    }

    __syncthreads();

    // 5. 所有线程读取最终max和sum
    MAX_SUM row_max_sum = warp_max_sum[0];

    // 6. 从Shared读取原始数据并计算输出
    for (int i = tid; i < vec_num; i += blockDim.x) {
        float4 val = shared_vec[i];
        float4 out;

        out.x = expf(val.x - row_max_sum.max) / row_max_sum.sum;
        out.y = expf(val.y - row_max_sum.max) / row_max_sum.sum;
        out.z = expf(val.z - row_max_sum.max) / row_max_sum.sum;
        out.w = expf(val.w - row_max_sum.max) / row_max_sum.sum;

        output4[i] = out;
    }
}

template <int BSIZE>
void softmax_warp_vectorized_online_shared_launcher(
    const float* input,
    float* output,
    int batch,
    int num_classes
) {
    static_assert(
        BSIZE % WARP_SIZE == 0,
        "BSIZE must be divisible by WARP_SIZE"
    );

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

    int vec_num = num_classes / 4;
    int warps_per_block = BSIZE / WARP_SIZE;

    int shared_bytes =
        vec_num * sizeof(float4) +
        warps_per_block * sizeof(MAX_SUM);

    // 一个block处理一行
    int blocks = batch;

    softmax_warp_vectorized_online_shared
        <<<blocks, BSIZE, shared_bytes>>>(
            input,
            output,
            batch,
            num_classes
        );
}

template <int BSIZE>
void softmax_warp_vectorized_online_shared_benchmark(int batch, int num_classes) {
    float* input = create_input(batch, num_classes);
    float* output = new float[batch * num_classes];

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, batch * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch * num_classes * sizeof(float));

    cudaMemcpy(d_input, input, batch * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_warp_vectorized_online_shared_launcher<BSIZE>(d_input, d_output, batch, num_classes);

    cudaMemcpy(output, d_output, batch * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
}

/*
后续优化思路
    1. 蝶形merge节省了广播但是有冗余运算，使用时需要同时权衡两者对全局效率的影响
    2. warp 特化，n warps load/store，m warps compute
    3. 根据 num_classes 选择不同的 kernel，num_classes 小到足够被shared mem覆盖时可以考虑shared mem版本，小到足够被 registers 覆盖是考虑reg版本
*/

int main(void) {
    int batch = 4096;
    int num_classes = 4096;

    softmax_cudnn_benchmark(batch, num_classes);

    softmax_naive_benchmark<32>(batch, num_classes);
    softmax_naive_benchmark<64>(batch, num_classes);
    softmax_naive_benchmark<128>(batch, num_classes);
    softmax_naive_benchmark<256>(batch, num_classes);
    softmax_naive_benchmark<512>(batch, num_classes);

    softmax_vectorized_benchmark<32>(batch, num_classes);
    softmax_vectorized_benchmark<64>(batch, num_classes);
    softmax_vectorized_benchmark<128>(batch, num_classes);
    softmax_vectorized_benchmark<256>(batch, num_classes);
    softmax_vectorized_benchmark<512>(batch, num_classes);

    softmax_warp_benchmark<32>(batch, num_classes);
    softmax_warp_benchmark<64>(batch, num_classes);
    softmax_warp_benchmark<128>(batch, num_classes);
    softmax_warp_benchmark<256>(batch, num_classes);
    softmax_warp_benchmark<512>(batch, num_classes);

    softmax_warp_vectorized_benchmark<32>(batch, num_classes);
    softmax_warp_vectorized_benchmark<64>(batch, num_classes);
    softmax_warp_vectorized_benchmark<128>(batch, num_classes);
    softmax_warp_vectorized_benchmark<256>(batch, num_classes);
    softmax_warp_vectorized_benchmark<512>(batch, num_classes);

    softmax_warp_vectorized_online_benchmark<32>(batch, num_classes);
    softmax_warp_vectorized_online_benchmark<64>(batch, num_classes);
    softmax_warp_vectorized_online_benchmark<128>(batch, num_classes);
    softmax_warp_vectorized_online_benchmark<256>(batch, num_classes);
    softmax_warp_vectorized_online_benchmark<512>(batch, num_classes);

    softmax_warp_vectorized_online_shared_benchmark<32>(batch, num_classes);
    softmax_warp_vectorized_online_shared_benchmark<64>(batch, num_classes);
    softmax_warp_vectorized_online_shared_benchmark<128>(batch, num_classes);
    softmax_warp_vectorized_online_shared_benchmark<256>(batch, num_classes);
    softmax_warp_vectorized_online_shared_benchmark<512>(batch, num_classes);
    softmax_warp_vectorized_online_shared_benchmark<1024>(batch, num_classes);

    return 0;
}