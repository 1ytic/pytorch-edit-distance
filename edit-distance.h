#pragma once

#include <torch/extension.h>

void CollapseRepeatedCuda(
        torch::Tensor source,
        torch::Tensor length);

void RemoveBlankCuda(
        torch::Tensor source,
        torch::Tensor length,
        torch::Tensor blank);

void StripSeparatorCuda(
        torch::Tensor source,
        torch::Tensor length,
        torch::Tensor separator);

torch::Tensor LevenshteinDistanceCuda(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length,
        torch::Tensor blank,
        torch::Tensor separator);
