#include<cuda_bf16.h>
#include<stdint.h>


__device__ __forceinline__
void cp_async_16B(
    const void* gmem_ptr,
    void* smem_ptr
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__
void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__
void cp_async_wait_group() {
    asm volatile(
        "cp.async.wait_group %0;\n"
        :
        :"n"(N)
        :
    );
}

// ldmatrix 通用 swizzle
__device__ __forceinline__ int shared_offset_Bank8x128B(
    int element_idx,
    int stride,
    int element_per_row,
    int element_size)
{
    int row = element_idx / stride;
    int col = element_idx % stride;

    int element_per_128B = 128 / element_size;
    int row_8x128B = row % 8;
    int col_8x128B = col % element_per_128B;

    int group_base = element_idx - element_idx % element_per_128B;
    return group_base + (col_8x128B + row_8x128B * element_per_row) % element_per_128B;
}

// shape: (NUM_ROWS, ELEMENT_PER_ROW)
template<int NUM_ROWS, int ELEMENT_PER_ROW, bool ASYNC>
__device__ __forceinline__
void cp_tile_from_global_to_shared(
    const __nv_bfloat16* src,
    int src_stride,
    __nv_bfloat16* smem,
    int valid_rows
) {
    const uint4* src_vec = reinterpret_cast<const uint4*>(src);
    uint4* smem_vec = reinterpret_cast<uint4*>(smem);

    constexpr int element_per_row = ELEMENT_PER_ROW / (sizeof(uint4) / sizeof(__nv_bfloat16));
    int src_vec_stride = src_stride / (sizeof(uint4) / sizeof(__nv_bfloat16));

    // load valid rows
    int num_valid_element_vec = valid_rows * element_per_row;
    for (
        int element_vec_idx = threadIdx.x;
        element_vec_idx < num_valid_element_vec;
        element_vec_idx+=blockDim.x
    ) {
        int element_vec_row = element_vec_idx / element_per_row;
        int element_vec_col = element_vec_idx % element_per_row;
        int element_vec_gmem_idx = element_vec_row * src_vec_stride + element_vec_col;

        int element_vec_smem_idx = shared_offset_Bank8x128B(element_vec_idx, element_per_row, 1, sizeof(uint4));
        if constexpr (!ASYNC) {
            smem_vec[element_vec_smem_idx] = src_vec[element_vec_gmem_idx];
        } else {
            cp_async_16B(src_vec + element_vec_gmem_idx, smem_vec + element_vec_smem_idx);
        }
    }

    // padding 0
    for (
        int element_vec_idx = threadIdx.x + num_valid_element_vec;
        element_vec_idx < NUM_ROWS * element_per_row;
        element_vec_idx+=blockDim.x
    ) {
        int element_vec_smem_idx = shared_offset_Bank8x128B(element_vec_idx, element_per_row, 1, sizeof(uint4));
        smem_vec[element_vec_smem_idx] = make_uint4(0, 0, 0, 0);
    }
}

template<int Br, int HEAD_DIM>
__device__ __forceinline__
void cp_qtile_from_global_to_shared(
    const __nv_bfloat16 *q,
    int q_stride,
    __nv_bfloat16 *smem_q,
    int nums_valid_row_q_tile
) {
    cp_tile_from_global_to_shared<Br, HEAD_DIM>(
        q,
        q_stride,
        smem_q,
        nums_valid_row_q_tile
    );
}

template<int Bc, int HEAD_DIM, bool ASYNC>
__device__ __forceinline__
void cp_kvtile_from_global_to_shared(
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    int kv_stride,
    __nv_bfloat16 *smem_k,
    __nv_bfloat16 *smem_v,
    int nums_valid_row_kv_tile
) {
    cp_tile_from_global_to_shared<Bc, HEAD_DIM, ASYNC>(
        k,
        kv_stride,
        smem_k,
        nums_valid_row_kv_tile
    );
    cp_tile_from_global_to_shared<Bc, HEAD_DIM, ASYNC>(
        v,
        kv_stride,
        smem_v,
        nums_valid_row_kv_tile
    );
}

// cp qtile
// cp.async ktile
// cp.async_commit ktile
template<bool Is_last>
__device__
void process_kv_tile() {
    // cp.async vtile
    // cp.async_commit vtile

    // cp.async_wait<1> ktile
    // stile init
    // stile = qtile @ ktile^T

    // if !Is_last
    // cp.async ktile
    // cp.async_commit ktile

    // stile mask
    // stile scale
    // stile local max
    // online softmax update global max
    // ptile = exp(stile - global max)
    // ptile local sum
    // online softmax update global sum

    // if !Is_last
    // cp.async_wait<1> vtile
    // else
    // cp.async_wait<0> vtile

    // o_acc apply global max
    // o_acc += ptile @ vtile

    // if Is_last
    // o_acc apply global sum
    // o_acc swizzle to smem_q
    // smem_q coalesce to global
}

__global__ void fa2(...) {

    // init

    // shared:
    // Q, K, V, softmax temporary

    // registers:
    // O accumulator
    // online max
    // online sum
    // init

    // prologue:
    // load Q
    // async load K0
    // commit K0

    // for:
    // process_kv_tile<false>

    // process_kv_tile<true>
}