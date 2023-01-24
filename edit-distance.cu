#include "edit-distance.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <c10/cuda/CUDAStream.h>


template <typename scalar_t>
__device__ __forceinline__ int is_include(
        scalar_t token,
        const scalar_t* __restrict__ subset,
        const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (token == subset[i]) {
            return true;
        }
    }
    return false;
}

template <typename scalar_t>
__device__ __forceinline__ int tokens_length(
        const scalar_t* __restrict__ first,
        const scalar_t* __restrict__ last,
        const scalar_t* __restrict__ separator,
        const size_t separator_size,
        const scalar_t* __restrict__ blank,
        const size_t blank_size) {

    int length = 0;
    bool prev_separator = true;
    auto iter = (scalar_t*)first;

    while (iter < last) {

        if (is_include(*iter, blank, blank_size)) {
            ++iter;
            continue;
        }

        if (separator_size == 0) {
            ++length;
        }
        else {
            bool curr_separator = is_include(*iter, separator, separator_size);
            if (!curr_separator && prev_separator) {
                ++length;
            }
            prev_separator = curr_separator;
        }

        ++iter;
    }

    return length;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t* first_token(
        const scalar_t* __restrict__ first,
        const scalar_t* __restrict__ last,
        const scalar_t* __restrict__ separator,
        const size_t separator_size,
        const scalar_t* __restrict__ blank,
        const size_t blank_size) {
    auto iter = (scalar_t*)first;
    while (iter < last) {
        if (is_include(*iter, blank, blank_size)) {
            ++iter;
            continue;
        }
        if (separator_size == 0) {
            break;
        }
        if (!is_include(*iter, separator, separator_size)) {
            break;
        }
        ++iter;
    }
    return iter;
}

template <typename scalar_t>
__device__ __forceinline__ bool compare_tokens(
        const scalar_t* __restrict__ a,
        const scalar_t* __restrict__ b,
        const scalar_t* __restrict__ c,
        const scalar_t* __restrict__ d,
        const scalar_t* __restrict__ separator,
        const size_t separator_size,
        const scalar_t* __restrict__ blank,
        const size_t blank_size) {
    auto iter1 = (scalar_t*)a;
    auto iter2 = (scalar_t*)b;
    while (true) {

        while (iter1 < c && is_include(*iter1, blank, blank_size)) {
            ++iter1;
        }

        while (iter2 < d && is_include(*iter2, blank, blank_size)) {
            ++iter2;
        }

        if (iter1 == c && iter2 == d) {
            return true;
        }

        if (iter1 == c && iter2 < d) {
            if (separator_size == 0) {
                return false;
            }
            return is_include(*iter2, separator, separator_size);
        }

        if (iter2 == d && iter1 < c) {
            if (separator_size == 0) {
                return false;
            }
            return is_include(*iter1, separator, separator_size);
        }

        if (separator_size == 0) {
            return *iter1 == *iter2;
        }

        if (is_include(*iter1, separator, separator_size) && is_include(*iter2, separator, separator_size)) {
            return true;
        }

        if (*iter1 != *iter2) {
            return false;
        }

        ++iter1;
        ++iter2;
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t* next_token(
        const scalar_t* __restrict__ first,
        const scalar_t* __restrict__ last,
        const scalar_t* __restrict__ separator,
        const size_t separator_size,
        const scalar_t* __restrict__ blank,
        const size_t blank_size) {
    auto iter = (scalar_t*)first;
    while (iter < last) {
        if (is_include(*iter, blank, blank_size)) {
            ++iter;
            continue;
        }
        if (separator_size == 0) {
            return ++iter;
        }
        if (is_include(*iter, separator, separator_size)) {
            break;
        }
        ++iter;
    }
    return first_token(iter, last, separator, separator_size, blank, blank_size);
}

template <typename scalar_t>
__global__ void levenshtein_distance_kernel(
        const scalar_t* __restrict__ source,
        const scalar_t* __restrict__ target,
        const int* __restrict__ source_length,
        const int* __restrict__ target_length,
        const size_t source_size,
        const size_t target_size,
        const scalar_t* __restrict__ separator, const size_t separator_size,
        const scalar_t* __restrict__ blank, const size_t blank_size,
        int* __restrict__ operations) {

    extern __shared__ short errors[];

    const int i = blockIdx.x;

    auto errors_prev = errors;
    auto errors_curr = errors + (target_size + 1) * 4;

    const scalar_t* hyp_begin = source + i * source_size;
    const scalar_t* ref_begin = target + i * target_size;

    const scalar_t* hyp_end = hyp_begin + source_length[i];
    const scalar_t* ref_end = ref_begin + target_length[i];

    int hyp_size = tokens_length(hyp_begin, hyp_end,
            separator, separator_size,
            blank, blank_size);

    int ref_size = tokens_length(ref_begin, ref_end,
            separator, separator_size,
            blank, blank_size);

    for (int r = 0; r <= ref_size; ++r) {
        errors_prev[r*4+0] = 0; // ins_num
        errors_prev[r*4+1] = r; // del_num
        errors_prev[r*4+2] = 0; // sub_num
        errors_prev[r*4+3] = r; // total_cost
    }

    auto hyp = first_token(hyp_begin, hyp_end, separator, separator_size,
                           blank, blank_size);

    for (int h = 1; h <= hyp_size; ++h) {

        errors_curr[0] = errors_prev[0] + 1;    // ins_num
        errors_curr[1] = errors_prev[1];        // del_num
        errors_curr[2] = errors_prev[2];        // sub_num
        errors_curr[3] = errors_prev[3] + 1;    // total_cost

        auto ref = first_token(ref_begin, ref_end, separator, separator_size,
                               blank, blank_size);

        for (int r = 1; r <= ref_size; ++r) {

            int r4 = r * 4;
            int p4 = r4 - 4;

            int ins_err = errors_prev[r4+3] + 1;
            int del_err = errors_curr[p4+3] + 1;
            int sub_err = errors_prev[p4+3];

            int d;

            if (compare_tokens(hyp, ref, hyp_end, ref_end, separator, separator_size,
                               blank, blank_size)) {
                d = 0;
            } else {
                d = 1;
                sub_err++;
            }

            if (sub_err < ins_err && sub_err < del_err) {

                errors_curr[r4+0] = errors_prev[p4+0];        // ins_num
                errors_curr[r4+1] = errors_prev[p4+1];        // del_num
                errors_curr[r4+2] = errors_prev[p4+2] + d;    // sub_num
                errors_curr[r4+3] = sub_err;                  // total_cost

            } else if (del_err < ins_err) {

                errors_curr[r4+0] = errors_curr[p4+0];        // ins_num
                errors_curr[r4+1] = errors_curr[p4+1] + 1;    // del_num
                errors_curr[r4+2] = errors_curr[p4+2];        // sub_num
                errors_curr[r4+3] = del_err;                  // total_cost

            } else {

                errors_curr[r4+0] = errors_prev[r4+0] + 1;    // ins_num
                errors_curr[r4+1] = errors_prev[r4+1];        // del_num
                errors_curr[r4+2] = errors_prev[r4+2];        // sub_num
                errors_curr[r4+3] = ins_err;                  // total_cost
            }

            ref = next_token(ref, ref_end, separator, separator_size,
                             blank, blank_size);
        }

        // alternate for the next recursion
        short* temp = errors_prev;
        errors_prev = errors_curr;
        errors_curr = temp;

        hyp = next_token(hyp, hyp_end, separator, separator_size,
                         blank, blank_size);
    }

    operations[i*4+0] = errors_prev[ref_size*4+0]; // ins
    operations[i*4+1] = errors_prev[ref_size*4+1]; // del
    operations[i*4+2] = errors_prev[ref_size*4+2]; // sub
    operations[i*4+3] = ref_size;
}

template <typename scalar_t>
__global__ void collapse_repeated_kernel(
        scalar_t* __restrict__ source,
        int* __restrict__ length,
        const size_t size) {

    const int i = threadIdx.x;

    if (length[i] <= 0) {
        return;
    }

    const scalar_t* iter = source + i * size;
    const scalar_t* last = iter + length[i];

    auto target = (scalar_t*)iter;

    int n = 1;
    ++iter;

    while (iter < last) {

        if (*iter == *target) {
            ++iter;
            continue;
        }

        ++target;

        *target = *iter;

        ++iter;
        ++n;
    }

    length[i] = n;
}

template <typename scalar_t>
__global__ void remove_blank_kernel(
        scalar_t* __restrict__ source,
        int* __restrict__ length,
        const size_t size,
        const scalar_t* __restrict__ blank, const size_t blank_size) {

    const int i = threadIdx.x;

    if (length[i] <= 0) {
        return;
    }

    const scalar_t* iter = source + i * size;
    const scalar_t* last = iter + length[i];

    auto target = (scalar_t*)iter;

    int n = 0;

    while (iter < last) {

        if (is_include(*iter, blank, blank_size)) {
            ++iter;
            continue;
        }

        *target = *iter;

        ++target;
        ++iter;
        ++n;
    }

    length[i] = n;
}

template <typename scalar_t>
__global__ void strip_separator_kernel(
        scalar_t* __restrict__ source,
        int* __restrict__ length,
        const size_t size,
        const scalar_t* __restrict__ separator, const size_t separator_size) {

    const int i = threadIdx.x;

    if (length[i] <= 0) {
        return;
    }

    const scalar_t* iter = source + i * size;
    const scalar_t* last = iter + length[i];

    auto target = (scalar_t*)iter;

    int n = 0;
    int p = 0;

    while (iter < last) {

        if (is_include(*iter, separator, separator_size)) {
            if (n == p) {
                ++iter;
                continue;
            }
            p = n + 1;
        }

        *target = *iter;

        ++target;
        ++iter;
        ++n;
    }

    if (n > 0 && n == p) {
        --n;
    }

    length[i] = n;
}

void CollapseRepeatedCuda(
        torch::Tensor source,
        torch::Tensor length) {
    const auto batch_size = source.size(0);
    auto stream = c10::cuda::getCurrentCUDAStream(source.device().index());
    AT_DISPATCH_ALL_TYPES(source.scalar_type(), "collapse_repeated", ([&] {
        collapse_repeated_kernel<scalar_t><<<1, batch_size, 0, stream>>>(
            source.data<scalar_t>(),
            length.data<int>(),
            source.size(1));
    }));
}

void RemoveBlankCuda(
        torch::Tensor source,
        torch::Tensor length,
        torch::Tensor blank) {
    const auto batch_size = source.size(0);
    auto stream = c10::cuda::getCurrentCUDAStream(source.device().index());
    AT_DISPATCH_ALL_TYPES(source.scalar_type(), "remove_blank", ([&] {
        remove_blank_kernel<scalar_t><<<1, batch_size, 0, stream>>>(
            source.data<scalar_t>(),
            length.data<int>(),
            source.size(1),
            blank.data<scalar_t>(),
            blank.ndimension() * blank.numel());
    }));
}

void StripSeparatorCuda(
        torch::Tensor source,
        torch::Tensor length,
        torch::Tensor separator) {
    const auto batch_size = source.size(0);
    auto stream = c10::cuda::getCurrentCUDAStream(source.device().index());
    AT_DISPATCH_ALL_TYPES(source.scalar_type(), "strip_separator", ([&] {
        strip_separator_kernel<scalar_t><<<1, batch_size, 0, stream>>>(
            source.data<scalar_t>(),
            length.data<int>(),
            source.size(1),
            separator.data<scalar_t>(),
            separator.ndimension() * separator.numel());
    }));
}

torch::Tensor LevenshteinDistanceCuda(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length,
        torch::Tensor blank,
        torch::Tensor separator) {

    const auto batch_size = source.size(0);
    const auto shared_size = (target.size(1) + 1) * 4 * 2 * sizeof(short);

    at::TensorOptions options(source.device());

    options = options.dtype(at::ScalarType::Int);

    auto operations = torch::empty({batch_size, 4}, options);

    auto stream = c10::cuda::getCurrentCUDAStream(source.device().index());

    AT_DISPATCH_ALL_TYPES(source.scalar_type(), "levenshtein_distance", ([&] {
        levenshtein_distance_kernel<scalar_t><<<batch_size, 1, shared_size, stream>>>(
            source.data<scalar_t>(),
            target.data<scalar_t>(),
            source_length.data<int>(),
            target_length.data<int>(),
            source.size(1),
            target.size(1),
            separator.data<scalar_t>(),
            separator.ndimension() * separator.numel(),
            blank.data<scalar_t>(),
            blank.ndimension() * blank.numel(),
            operations.data<int>());
    }));

    return operations;
}
