
#include <memory>
#include "model/transformer_decoder_block.hpp"


TransformerDecoderBlock::TransformerDecoderBlock(int64_t d_model, int64_t nhead, std::vector<int64_t> ffn_sizes, double dropout)
    : d_model_(d_model)
    , nhead_(nhead)
    , head_dim_(d_model / nhead)
{
    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    self_q_proj_ = register_module("self_q_proj", torch::nn::Linear(d_model, d_model));
    self_k_proj_ = register_module("self_k_proj", torch::nn::Linear(d_model, d_model));
    self_v_proj_ = register_module("self_v_proj", torch::nn::Linear(d_model, d_model));
    self_o_proj_ = register_module("self_o_proj", torch::nn::Linear(d_model, d_model));

    cross_q_proj_ = register_module("cross_q_proj", torch::nn::Linear(d_model, d_model));
    cross_k_proj_ = register_module("cross_k_proj", torch::nn::Linear(d_model, d_model));
    cross_v_proj_ = register_module("cross_v_proj", torch::nn::Linear(d_model, d_model));
    cross_o_proj_ = register_module("cross_o_proj", torch::nn::Linear(d_model, d_model));

    norm1_ = register_module("norm1", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({d_model})));
    
    norm2_ = register_module("norm2", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({d_model})));
    
    norm3_ = register_module("norm3", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({d_model})));

    dropout1_ = register_module("dropout1", torch::nn::Dropout(dropout));
    dropout2_ = register_module("dropout2", torch::nn::Dropout(dropout));
    dropout3_ = register_module("dropout3", torch::nn::Dropout(dropout));

    std::vector<int64_t> full_ffn;
    full_ffn.push_back(d_model);
    for (auto& s : ffn_sizes) full_ffn.push_back(s);
    full_ffn.push_back(d_model);
    
    ffn_ = register_module("ffn", std::make_shared<Perceptron>(full_ffn));

    this->to(device_);
}


torch::Tensor TransformerDecoderBlock::forward(torch::Tensor x, torch::Tensor encoder_out, 
                                               torch::Tensor attn_mask, torch::Tensor padding_mask)
{
    if (x.device() != device_) x = x.to(device_);
    if (encoder_out.device() != device_) encoder_out = encoder_out.to(device_);
    if (attn_mask.defined() && attn_mask.device() != device_) attn_mask = attn_mask.to(device_);
    if (padding_mask.defined() && padding_mask.device() != device_) padding_mask = padding_mask.to(device_);

    int64_t B = x.size(0);
    int64_t S = x.size(1);

    // Pre-LN: Masked Self-Attention
    torch::Tensor residual = x;
    torch::Tensor x_norm = norm1_->forward(x);
    
    auto q = self_q_proj_->forward(x_norm).reshape({B, S, nhead_, head_dim_}).transpose(1, 2);
    auto k = self_k_proj_->forward(x_norm).reshape({B, S, nhead_, head_dim_}).transpose(1, 2);
    auto v = self_v_proj_->forward(x_norm).reshape({B, S, nhead_, head_dim_}).transpose(1, 2);
    
    // Causal mask
    auto causal = torch::ones({S, S}, device_).triu(1).to(torch::kBool);
    auto mask = causal.unsqueeze(0).unsqueeze(0);
    
    double dropout_p = dropout1_->is_training() ? 0.1 : 0.0;
    auto self_attn_out = torch::scaled_dot_product_attention(q, k, v, mask, dropout_p);
    
    self_attn_out = self_attn_out.transpose(1, 2).reshape({B, S, d_model_});
    self_attn_out = self_o_proj_->forward(self_attn_out);
    
    x = residual + dropout1_->forward(self_attn_out);

    // Pre-LN: Cross-Attention
    residual = x;
    x_norm = norm2_->forward(x);
    
    int64_t S_enc = encoder_out.size(0);
    
    q = cross_q_proj_->forward(x_norm).reshape({B, S, nhead_, head_dim_}).transpose(1, 2);
    k = cross_k_proj_->forward(encoder_out.transpose(0, 1)).reshape({B, S_enc, nhead_, head_dim_}).transpose(1, 2);
    v = cross_v_proj_->forward(encoder_out.transpose(0, 1)).reshape({B, S_enc, nhead_, head_dim_}).transpose(1, 2);
    
    double cross_dropout_p = dropout2_->is_training() ? 0.1 : 0.0;
    auto cross_attn_out = torch::scaled_dot_product_attention(q, k, v, {}, cross_dropout_p);
    
    cross_attn_out = cross_attn_out.transpose(1, 2).reshape({B, S, d_model_});
    cross_attn_out = cross_o_proj_->forward(cross_attn_out);
    
    x = residual + dropout2_->forward(cross_attn_out);

    // Pre-LN: FFN
    residual = x;
    x_norm = norm3_->forward(x);
    x = residual + dropout3_->forward(ffn_->forward(x_norm));

    return x;
}


torch::Tensor TransformerDecoderBlock::forward_kv(torch::Tensor x, torch::Tensor encoder_out,
                                                   std::vector<torch::Tensor>& self_kv_cache,
                                                   std::vector<torch::Tensor>& cross_kv_cache)
{
    if (x.device() != device_) x = x.to(device_);
    if (encoder_out.device() != device_) encoder_out = encoder_out.to(device_);

    int64_t B = x.size(0);

    // Pre-LN: Masked Self-Attention с KV-cache
    torch::Tensor residual = x;
    torch::Tensor x_norm = norm1_->forward(x);
    
    auto q = self_q_proj_->forward(x_norm).reshape({B, 1, nhead_, head_dim_}).transpose(1, 2);
    auto k_new = self_k_proj_->forward(x_norm).reshape({B, 1, nhead_, head_dim_}).transpose(1, 2);
    auto v_new = self_v_proj_->forward(x_norm).reshape({B, 1, nhead_, head_dim_}).transpose(1, 2);
    
    if (self_kv_cache.empty()) {
        self_kv_cache.push_back(k_new);
        self_kv_cache.push_back(v_new);
    } else {
        self_kv_cache[0] = torch::cat({self_kv_cache[0], k_new}, 2);
        self_kv_cache[1] = torch::cat({self_kv_cache[1], v_new}, 2);
    }
    
    auto self_attn_out = torch::scaled_dot_product_attention(
        q, self_kv_cache[0], self_kv_cache[1], {}, 0.0);
    
    self_attn_out = self_attn_out.transpose(1, 2).reshape({B, 1, d_model_});
    self_attn_out = self_o_proj_->forward(self_attn_out);
    
    x = residual + self_attn_out;

    // Pre-LN: Cross-Attention с KV-cache
    residual = x;
    x_norm = norm2_->forward(x);
    
    q = cross_q_proj_->forward(x_norm).reshape({B, 1, nhead_, head_dim_}).transpose(1, 2);
    
    if (cross_kv_cache.empty()) {
        int64_t S_enc = encoder_out.size(0);
        auto k_enc = cross_k_proj_->forward(encoder_out.transpose(0, 1))
            .reshape({B, S_enc, nhead_, head_dim_}).transpose(1, 2);
        auto v_enc = cross_v_proj_->forward(encoder_out.transpose(0, 1))
            .reshape({B, S_enc, nhead_, head_dim_}).transpose(1, 2);
        cross_kv_cache.push_back(k_enc);
        cross_kv_cache.push_back(v_enc);
    }
    
    auto cross_attn_out = torch::scaled_dot_product_attention(
        q, cross_kv_cache[0], cross_kv_cache[1], {}, 0.0);
    
    cross_attn_out = cross_attn_out.transpose(1, 2).reshape({B, 1, d_model_});
    cross_attn_out = cross_o_proj_->forward(cross_attn_out);
    
    x = residual + cross_attn_out;

    // Pre-LN: FFN
    residual = x;
    x_norm = norm3_->forward(x);
    x = residual + ffn_->forward(x_norm);

    return x;
}

