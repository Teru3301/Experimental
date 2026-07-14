
#pragma once

#include <torch/torch.h>

#include "model/modules/layer_norm.hpp"
#include "model/modules/mlp.hpp"
#include "model/modules/transformer_block.hpp"

class CustomBlockImpl : public torch::nn::Module
{
public:

    CustomBlockImpl(
        MLP shared_mlp,
        int hidden_size,
        int shared_dim,
        int attention_heads,
        int ffn_hidden,
        const std::vector<int>& private_hidden,
        double dropout = 0.1);

    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor mask = {});

private:

    MLP shared_mlp_{nullptr};

    MLP private_mlp_{nullptr};

    LayerNorm shared_norm_{nullptr};

    LayerNorm private_norm_{nullptr};

    TransformerBlock transformer_{nullptr};

    int shared_dim_;

    int private_dim_;
};

TORCH_MODULE(CustomBlock);

