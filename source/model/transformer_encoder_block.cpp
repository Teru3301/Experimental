
#include <memory>
#include "model/transformer_encoder_block.hpp"


TransformerEncoderBlock::TransformerEncoderBlock(int64_t d_model, int64_t nhead, std::vector<int64_t> ffn_sizes, double dropout)
{
    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    attn_ = register_module("attn", torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(d_model, nhead)
            .dropout(dropout)));

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

    // Перемещаем всё на GPU
    this->to(device_);
}


torch::Tensor TransformerEncoderBlock::forward(torch::Tensor x, torch::Tensor padding_mask)
{
    // Проверяем, что вход на правильном устройстве
    if (x.device() != device_) {
        x = x.to(device_);
    }
    
    // Если маска задана и на другом устройстве - перемещаем
    if (padding_mask.defined() && padding_mask.device() != device_) {
        padding_mask = padding_mask.to(device_);
    }

    // Pre-LN: Self-Attention
    // Вход: (batch, seq, features) -> транспонируем в (seq, batch, features)
    torch::Tensor residual = x;
    torch::Tensor x_norm = norm1_->forward(x);
    x_norm = x_norm.transpose(0, 1);
    
    // MultiheadAttention forward:
    // forward(query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights)
    auto attn_result = attn_->forward(x_norm, x_norm, x_norm, padding_mask);
    torch::Tensor attn_out = std::get<0>(attn_result);
    
    // Транспонируем обратно: (seq, batch, features) -> (batch, seq, features)
    attn_out = attn_out.transpose(0, 1);
    
    x = residual + dropout1_->forward(attn_out);

    // Pre-LN: FFN (Perceptron)
    residual = x;
    x_norm = norm2_->forward(x);
    x = residual + dropout2_->forward(ffn_->forward(x_norm));

    return x;
}

