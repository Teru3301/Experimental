
#pragma once

#include <torch/torch.h>

#include "model/modules/attention.hpp"
#include "model/modules/layer_norm.hpp"
#include "model/modules/mlp.hpp"

class TransformerBlockImpl : public torch::nn::Module
{
public:

    TransformerBlockImpl(
        int hidden_size,
        int heads,
        int ffn_hidden,
        double dropout = 0.1
    );

    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor mask = {}
    );

private:

    LayerNorm norm1{nullptr};

    LayerNorm norm2{nullptr};

    MultiHeadAttention attention{nullptr};

    MLP ffn{nullptr};

    torch::nn::Dropout dropout{nullptr};
};

TORCH_MODULE(TransformerBlock);

