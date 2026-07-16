
#pragma once

#include <torch/torch.h>
#include "model/perceptron.hpp"
#include "model/attention.hpp"



class GPTBlock : public torch::nn::Module
{
    std::shared_ptr<Attention> attn_{nullptr};
    std::shared_ptr<Perceptron> ffn_{nullptr};
    
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};
    torch::nn::Dropout dropout1_{nullptr};
    torch::nn::Dropout dropout2_{nullptr};

public:
    GPTBlock(int64_t d_model, int64_t nhead, int64_t max_seq_len,
             std::vector<int64_t> ffn_sizes, double dropout = 0.1);
    
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor forward_kv(torch::Tensor x, KVCache& cache, int64_t pos);
};

