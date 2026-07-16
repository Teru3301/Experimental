
#include <memory>
#include "model/transformer_encoder_block.hpp"


TransformerEncoderBlock::TransformerEncoderBlock(int64_t d_model, int64_t nhead, std::vector<int64_t> ffn_sizes, double dropout)
    : d_model_(d_model)
    , nhead_(nhead)
    , head_dim_(d_model / nhead)
{
    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    q_proj_ = register_module("q_proj", torch::nn::Linear(d_model, d_model));
    k_proj_ = register_module("k_proj", torch::nn::Linear(d_model, d_model));
    v_proj_ = register_module("v_proj", torch::nn::Linear(d_model, d_model));
    o_proj_ = register_module("o_proj", torch::nn::Linear(d_model, d_model));

    norm1_ = register_module("norm1", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({d_model})));
    
    norm2_ = register_module("norm2", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({d_model})));

    dropout1_ = register_module("dropout1", torch::nn::Dropout(dropout));
    dropout2_ = register_module("dropout2", torch::nn::Dropout(dropout));

    // Строим FFN
    std::vector<int64_t> full_ffn;
    full_ffn.push_back(d_model);
    for (auto& s : ffn_sizes) full_ffn.push_back(s);
    full_ffn.push_back(d_model);
    
    ffn_ = register_module("ffn", std::make_shared<Perceptron>(full_ffn));

    this->to(device_);
}


torch::Tensor TransformerEncoderBlock::forward(torch::Tensor x, torch::Tensor padding_mask)
{
    if (x.device() != device_) {
        x = x.to(device_);
    }
    
    if (padding_mask.defined() && padding_mask.device() != device_) {
        padding_mask = padding_mask.to(device_);
    }

    // Pre-LN: Self-Attention
    torch::Tensor residual = x;
    torch::Tensor x_norm = norm1_->forward(x);
    
    int64_t B = x_norm.size(0);
    int64_t S = x_norm.size(1);
    
    auto q = q_proj_->forward(x_norm);
    auto k = k_proj_->forward(x_norm);
    auto v = v_proj_->forward(x_norm);
    
    q = q.reshape({B, S, nhead_, head_dim_}).transpose(1, 2);
    k = k.reshape({B, S, nhead_, head_dim_}).transpose(1, 2);
    v = v.reshape({B, S, nhead_, head_dim_}).transpose(1, 2);
    
    double dropout_p = dropout1_->is_training() ? 0.1 : 0.0;
    
    torch::Tensor attn_out;
    if (padding_mask.defined()) {
        // padding_mask: (B, S) bool — true где padding
        // Для attention: расширяем до (B, 1, 1, S)
        auto mask = padding_mask.unsqueeze(1).unsqueeze(1);
        attn_out = torch::scaled_dot_product_attention(q, k, v, mask, dropout_p);
    } else {
        attn_out = torch::scaled_dot_product_attention(q, k, v, {}, dropout_p);
    }
    
    attn_out = attn_out.transpose(1, 2).reshape({B, S, d_model_});
    attn_out = o_proj_->forward(attn_out);
    
    x = residual + dropout1_->forward(attn_out);

    // Pre-LN: FFN
    residual = x;
    x_norm = norm2_->forward(x);
    x = residual + dropout2_->forward(ffn_->forward(x_norm));

    return x;
}

