
#pragma once

#include <torch/torch.h>
#include <vector>



struct KVCache
{
    torch::Tensor key;
    torch::Tensor value;
    int64_t size = 0;  // сколько токенов уже записано
};



class Attention : public torch::nn::Module
{
    torch::nn::Linear q_proj_{nullptr};
    torch::nn::Linear k_proj_{nullptr};
    torch::nn::Linear v_proj_{nullptr};
    torch::nn::Linear o_proj_{nullptr};
    
    int64_t d_model_;
    int64_t nhead_;
    int64_t head_dim_;
    int64_t max_seq_len_;
    double dropout_;
    
    // RoPE кэш
    torch::Tensor rope_cos_{};
    torch::Tensor rope_sin_{};
    
    void build_rope_cache(int64_t max_len);
    void apply_rope(torch::Tensor& q, torch::Tensor& k, int64_t start_pos);

public:
    Attention(int64_t d_model, int64_t nhead, int64_t max_seq_len, double dropout = 0.1);
    
    // Обучение: causal attention на всю последовательность
    torch::Tensor forward(torch::Tensor x);
    
    // Инференс: один токен с KV-cache
    torch::Tensor forward_kv(torch::Tensor x, KVCache& cache, int64_t pos);
};

