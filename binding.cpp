#include "edit-distance.h"

#include <torch/types.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void CollapseRepeated(
        torch::Tensor source,
        torch::Tensor length) {

    CHECK_INPUT(source);
    CHECK_INPUT(length);

    CollapseRepeatedCuda(source, length);
}

void RemoveBlank(
        torch::Tensor source,
        torch::Tensor length,
        torch::Tensor blank) {

    CHECK_INPUT(source);
    CHECK_INPUT(length);
    CHECK_INPUT(blank);

    RemoveBlankCuda(source, length, blank);
}

void StripSeparator(
        torch::Tensor source,
        torch::Tensor length,
        torch::Tensor separator) {

    CHECK_INPUT(source);
    CHECK_INPUT(length);
    CHECK_INPUT(separator);

    StripSeparatorCuda(source, length, separator);
}

torch::Tensor LevenshteinDistance(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length,
        torch::Tensor blank,
        torch::Tensor separator) {

    CHECK_INPUT(source);
    CHECK_INPUT(target);
    CHECK_INPUT(source_length);
    CHECK_INPUT(target_length);
    CHECK_INPUT(blank);
    CHECK_INPUT(separator);

    return LevenshteinDistanceCuda(source, target, source_length, target_length, blank, separator);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("collapse_repeated", &CollapseRepeated, "Merge repeated tokens");
    m.def("remove_blank", &RemoveBlank, "Remove blank");
    m.def("strip_separator", &StripSeparator, "Strip separator");
    m.def("levenshtein_distance", &LevenshteinDistance, "Levenshtein distance");
}
