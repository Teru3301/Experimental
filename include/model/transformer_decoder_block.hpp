
#pragma once

#include <torch/torch.h>
#include "model/perceptron.hpp"



class TransformerDecoderBlock : public torch::nn::Module
{
    torch::nn::MultiheadAttention self_attn_{nullptr};
    torch::nn::MultiheadAttention cross_attn_{nullptr};
    std::shared_ptr<Perceptron> ffn_{nullptr};
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};
    torch::nn::LayerNorm norm3_{nullptr};
    torch::nn::Dropout dropout1_{nullptr};
    torch::nn::Dropout dropout2_{nullptr};
    torch::nn::Dropout dropout3_{nullptr};
    torch::Device device_{torch::kCPU};

public:
    TransformerDecoderBlock(int64_t d_model, int64_t nhead, std::vector<int64_t> ffn_sizes, double dropout = 0.1);
    torch::Tensor forward(torch::Tensor x, torch::Tensor encoder_out, torch::Tensor attn_mask = {}, torch::Tensor padding_mask = {});
    torch::Device device() const { return device_; }
};

