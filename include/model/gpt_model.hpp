
#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include "model/gpt_block.hpp"



class GPTModel : public torch::nn::Module
{
    torch::nn::Embedding token_embedding_{nullptr};
    torch::nn::Dropout embed_dropout_{nullptr};
    
    std::vector<std::shared_ptr<GPTBlock>> blocks_;
    
    torch::nn::LayerNorm final_norm_{nullptr};
    torch::nn::Linear lm_head_{nullptr};
    
    int64_t d_model_;
    int64_t vocab_size_;
    int64_t max_seq_len_;
    torch::Device device_{torch::kCPU};

public:
    GPTModel(int64_t vocab_size, int64_t d_model, int64_t max_seq_len,
             int64_t nhead, int64_t num_blocks,
             std::vector<int64_t> ffn_sizes, double dropout = 0.1);
    
    // Обучение: (B, S) → (B, S, vocab_size)
    torch::Tensor forward(torch::Tensor x);
    
    // Генерация: (1, S) → (1, S + new_tokens)
    torch::Tensor generate(torch::Tensor prompt, int64_t max_new_tokens,
                           double temperature = 0.8, int64_t top_k = 0);
    
    torch::Device device() const { return device_; }
};

