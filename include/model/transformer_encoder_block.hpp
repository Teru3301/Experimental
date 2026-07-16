
#pragma once

#include <torch/torch.h>
#include "model/perceptron.hpp"



class TransformerEncoderBlock : public torch::nn::Module
{
    torch::nn::Linear q_proj_{nullptr};
    torch::nn::Linear k_proj_{nullptr};
    torch::nn::Linear v_proj_{nullptr};
    torch::nn::Linear o_proj_{nullptr};
    
    std::shared_ptr<Perceptron> ffn_{nullptr};
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};
    torch::nn::Dropout dropout1_{nullptr};
    torch::nn::Dropout dropout2_{nullptr};
    
    int64_t d_model_;
    int64_t nhead_;
    int64_t head_dim_;
    torch::Device device_{torch::kCPU};

public:
    TransformerEncoderBlock(int64_t d_model, int64_t nhead, std::vector<int64_t> ffn_sizes, double dropout = 0.1);
    torch::Tensor forward(torch::Tensor x, torch::Tensor padding_mask = {});
    torch::Device device() const { return device_; }
};

