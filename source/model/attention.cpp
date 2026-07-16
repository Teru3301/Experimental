
#include "model/attention.hpp"
#include <cmath>


Attention::Attention(int64_t d_model, int64_t nhead, int64_t max_seq_len, double dropout)
    : d_model_(d_model)
    , nhead_(nhead)
    , head_dim_(d_model / nhead)
    , max_seq_len_(max_seq_len)
    , dropout_(dropout)
{
    TORCH_CHECK(
        d_model % nhead == 0,
        "d_model must be divisible by nhead, got d_model=", d_model, ", nhead=", nhead);
    
    q_proj_ = register_module("q_proj", torch::nn::Linear(d_model, d_model));
    k_proj_ = register_module("k_proj", torch::nn::Linear(d_model, d_model));
    v_proj_ = register_module("v_proj", torch::nn::Linear(d_model, d_model));
    o_proj_ = register_module("o_proj", torch::nn::Linear(d_model, d_model));
    
    build_rope_cache(max_seq_len);
}


void Attention::build_rope_cache(int64_t max_len)
{
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    
    auto i = torch::arange(0, head_dim_, 2, torch::kFloat64);
    auto theta = torch::pow(10000.0, -i / head_dim_);
    
    auto positions = torch::arange(max_len, torch::kFloat64).unsqueeze(1);
    auto freqs = positions * theta.unsqueeze(0);
    auto emb = torch::cat(std::vector<torch::Tensor>{freqs, freqs}, -1);
    
    rope_cos_ = emb.cos().to(torch::kFloat32).to(device);
    rope_sin_ = emb.sin().to(torch::kFloat32).to(device);
}


void Attention::apply_rope(torch::Tensor& q, torch::Tensor& k, int64_t start_pos)
{
    int64_t S = q.size(2);
    
    auto cos = rope_cos_.slice(0, start_pos, start_pos + S)
        .unsqueeze(0).unsqueeze(0);
    auto sin = rope_sin_.slice(0, start_pos, start_pos + S)
        .unsqueeze(0).unsqueeze(0);
    
    auto q1 = q.slice(-1, 0, head_dim_ / 2);
    auto q2 = q.slice(-1, head_dim_ / 2, head_dim_);
    auto k1 = k.slice(-1, 0, head_dim_ / 2);
    auto k2 = k.slice(-1, head_dim_ / 2, head_dim_);
    
    auto cos1 = cos.slice(-1, 0, head_dim_ / 2);
    auto sin1 = sin.slice(-1, 0, head_dim_ / 2);
    auto cos2 = cos.slice(-1, head_dim_ / 2, head_dim_);
    auto sin2 = sin.slice(-1, head_dim_ / 2, head_dim_);
    
    q = torch::cat({q1 * cos1 - q2 * sin1, q2 * cos2 + q1 * sin2}, -1);
    k = torch::cat({k1 * cos1 - k2 * sin1, k2 * cos2 + k1 * sin2}, -1);
}


torch::Tensor Attention::forward(torch::Tensor x)
{
    int64_t B = x.size(0);
    int64_t S = x.size(1);
    
    auto q = q_proj_->forward(x).reshape({B, S, nhead_, head_dim_}).transpose(1, 2);
    auto k = k_proj_->forward(x).reshape({B, S, nhead_, head_dim_}).transpose(1, 2);
    auto v = v_proj_->forward(x).reshape({B, S, nhead_, head_dim_}).transpose(1, 2);
    
    apply_rope(q, k, 0);
    
    auto mask = torch::ones({S, S}, q.device()).triu(1).to(torch::kBool);
    mask = mask.unsqueeze(0).unsqueeze(0);
    
    double p = this->is_training() ? dropout_ : 0.0;
    auto attn_out = torch::scaled_dot_product_attention(q, k, v, mask, p);
    
    attn_out = attn_out.transpose(1, 2).reshape({B, S, d_model_});
    return o_proj_->forward(attn_out);
}


torch::Tensor Attention::forward_kv(torch::Tensor x, KVCache& cache, int64_t pos)
{
    int64_t B = x.size(0);
    auto device = x.device();
    
    auto q = q_proj_->forward(x).reshape({B, 1, nhead_, head_dim_}).transpose(1, 2);
    auto k_new = k_proj_->forward(x).reshape({B, 1, nhead_, head_dim_}).transpose(1, 2);
    auto v_new = v_proj_->forward(x).reshape({B, 1, nhead_, head_dim_}).transpose(1, 2);
    
    apply_rope(q, k_new, pos);
    
    if (cache.size == 0) {
        cache.key = torch::zeros({B, nhead_, max_seq_len_, head_dim_}, 
            torch::TensorOptions().dtype(k_new.dtype()).device(device));
        cache.value = torch::zeros({B, nhead_, max_seq_len_, head_dim_},
            torch::TensorOptions().dtype(v_new.dtype()).device(device));
    }
    
    cache.key.slice(2, pos, pos + 1).copy_(k_new);
    cache.value.slice(2, pos, pos + 1).copy_(v_new);
    cache.size = pos + 1;
    
    auto k_cache = cache.key.slice(2, 0, cache.size);
    auto v_cache = cache.value.slice(2, 0, cache.size);
    
    auto attn_out = torch::scaled_dot_product_attention(q, k_cache, v_cache, {}, 0.0);
    
    attn_out = attn_out.transpose(1, 2).reshape({B, 1, d_model_});
    return o_proj_->forward(attn_out);
}

