
#pragma once

#include <torch/torch.h>

class MultiHeadAttentionImpl : public torch::nn::Module
{
public:
    MultiHeadAttentionImpl(
        int hidden_size,
        int num_heads,
        double dropout = 0.0,
        bool bias = true
    );

    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor mask = {}
    );

private:
    int hidden_size_;
    int num_heads_;
    int head_dim_;

    torch::nn::Linear q_proj{nullptr};
    torch::nn::Linear k_proj{nullptr};
    torch::nn::Linear v_proj{nullptr};
    torch::nn::Linear out_proj{nullptr};

    torch::nn::Dropout dropout{nullptr};
};

TORCH_MODULE(MultiHeadAttention);

