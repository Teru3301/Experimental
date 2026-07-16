
#pragma once

#include <torch/torch.h>
#include <vector>
#include "model/perceptron.hpp"



class TransformerDecoderBlock : public torch::nn::Module
{
    // Self-attention проекции
    torch::nn::Linear self_q_proj_{nullptr};
    torch::nn::Linear self_k_proj_{nullptr};
    torch::nn::Linear self_v_proj_{nullptr};
    torch::nn::Linear self_o_proj_{nullptr};
    
    // Cross-attention проекции
    torch::nn::Linear cross_q_proj_{nullptr};
    torch::nn::Linear cross_k_proj_{nullptr};
    torch::nn::Linear cross_v_proj_{nullptr};
    torch::nn::Linear cross_o_proj_{nullptr};
    
    std::shared_ptr<Perceptron> ffn_{nullptr};
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};
    torch::nn::LayerNorm norm3_{nullptr};
    torch::nn::Dropout dropout1_{nullptr};
    torch::nn::Dropout dropout2_{nullptr};
    torch::nn::Dropout dropout3_{nullptr};
    
    int64_t d_model_;
    int64_t nhead_;
    int64_t head_dim_;
    torch::Device device_{torch::kCPU};

public:
    TransformerDecoderBlock(int64_t d_model, int64_t nhead, std::vector<int64_t> ffn_sizes, double dropout = 0.1);
    
    // Обычный forward (обучение)
    torch::Tensor forward(torch::Tensor x, torch::Tensor encoder_out, 
                          torch::Tensor attn_mask = {}, torch::Tensor padding_mask = {});
    
    // Forward с KV-cache (генерация)
    torch::Tensor forward_kv(torch::Tensor x, torch::Tensor encoder_out,
                             std::vector<torch::Tensor>& self_kv_cache,
                             std::vector<torch::Tensor>& cross_kv_cache);
    
    torch::Device device() const { return device_; }
};

