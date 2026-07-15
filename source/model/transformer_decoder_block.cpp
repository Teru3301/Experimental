
#include <memory>
#include "model/transformer_decoder_block.hpp"


TransformerDecoderBlock::TransformerDecoderBlock(int64_t d_model, int64_t nhead, std::vector<int64_t> ffn_sizes, double dropout)
{
    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // Self-attention с causal mask
    self_attn_ = register_module("self_attn", torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(d_model, nhead)
            .dropout(dropout)));

    // Cross-attention к encoder output
    cross_attn_ = register_module("cross_attn", torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(d_model, nhead)
            .dropout(dropout)));

    norm1_ = register_module("norm1", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({d_model})));
    
    norm2_ = register_module("norm2", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({d_model})));
    
    norm3_ = register_module("norm3", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({d_model})));

    dropout1_ = register_module("dropout1", torch::nn::Dropout(dropout));
    dropout2_ = register_module("dropout2", torch::nn::Dropout(dropout));
    dropout3_ = register_module("dropout3", torch::nn::Dropout(dropout));

    // Строим FFN
    std::vector<int64_t> full_ffn;
    full_ffn.push_back(d_model);
    for (auto& s : ffn_sizes) full_ffn.push_back(s);
    full_ffn.push_back(d_model);
    
    ffn_ = register_module("ffn", std::make_shared<Perceptron>(full_ffn));

    // Перемещаем всё на GPU
    this->to(device_);
}


torch::Tensor TransformerDecoderBlock::forward(torch::Tensor x, torch::Tensor encoder_out, torch::Tensor attn_mask, torch::Tensor padding_mask)
{
    // Проверяем, что вход на правильном устройстве
    if (x.device() != device_) {
        x = x.to(device_);
    }
    
    if (encoder_out.device() != device_) {
        encoder_out = encoder_out.to(device_);
    }
    
    if (attn_mask.defined() && attn_mask.device() != device_) {
        attn_mask = attn_mask.to(device_);
    }
    
    if (padding_mask.defined() && padding_mask.device() != device_) {
        padding_mask = padding_mask.to(device_);
    }

    // Pre-LN: Masked Self-Attention
    torch::Tensor residual = x;
    torch::Tensor x_norm = norm1_->forward(x);
    
    // Транспонируем для attention: (batch, seq, features) -> (seq, batch, features)
    x_norm = x_norm.transpose(0, 1);
    
    // Self-attention: forward(query, key, value, key_padding_mask, need_weights, attn_mask)
    // attn_mask — causal mask, key_padding_mask — padding
    auto self_attn_result = self_attn_->forward(
        x_norm, x_norm, x_norm, 
        padding_mask,      // key_padding_mask
        true,              // need_weights
        attn_mask);        // attn_mask
    torch::Tensor self_attn_out = std::get<0>(self_attn_result);
    
    // Транспонируем обратно
    self_attn_out = self_attn_out.transpose(0, 1);
    
    x = residual + dropout1_->forward(self_attn_out);

    // Pre-LN: Cross-Attention (decoder -> encoder)
    residual = x;
    x_norm = norm2_->forward(x);
    x_norm = x_norm.transpose(0, 1);
    
    // Cross-attention: forward(query, key, value)
    auto cross_attn_result = cross_attn_->forward(x_norm, encoder_out, encoder_out);
    torch::Tensor cross_attn_out = std::get<0>(cross_attn_result);
    
    cross_attn_out = cross_attn_out.transpose(0, 1);
    
    x = residual + dropout2_->forward(cross_attn_out);

    // Pre-LN: FFN
    residual = x;
    x_norm = norm3_->forward(x);
    x = residual + dropout3_->forward(ffn_->forward(x_norm));

    return x;
}

