
#pragma once

#include <torch/torch.h>

class LayerNormImpl : public torch::nn::Module
{
public:

    explicit LayerNormImpl(
        int hidden_size,
        double eps = 1e-5
    );

    torch::Tensor forward(torch::Tensor x);

private:

    torch::nn::LayerNorm norm{nullptr};
};

TORCH_MODULE(LayerNorm);

