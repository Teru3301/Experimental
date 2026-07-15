
#include <memory>
#include "model/transformer_block.hpp"


TransformerBlock::TransformerBlock(int64_t d_model, int64_t nhead, std::vector<int64_t> ffn_sizes, double dropout)
{
    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // Создаем слои (без batch_first, если не поддерживается)
    attn_ = torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(d_model, nhead)
            .dropout(dropout));

    norm1_ = torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({d_model}));
    
    norm2_ = torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({d_model}));

    dropout1_ = torch::nn::Dropout(dropout);
    dropout2_ = torch::nn::Dropout(dropout);

    // Строим FFN
    std::vector<int64_t> full_ffn;
    full_ffn.push_back(d_model);
    for (auto& s : ffn_sizes) full_ffn.push_back(s);
    full_ffn.push_back(d_model);
    
    ffn_ = std::make_shared<Perceptron>(full_ffn);

    // Регистрируем все модули
    register_module("attn", attn_);
    register_module("norm1", norm1_);
    register_module("norm2", norm2_);
    register_module("dropout1", dropout1_);
    register_module("dropout2", dropout2_);
    register_module("ffn", ffn_);

    // Перемещаем всё на GPU
    this->to(device_);
}


torch::Tensor TransformerBlock::forward(torch::Tensor x, torch::Tensor attn_mask)
{
    // Проверяем, что вход на правильном устройстве
    if (x.device() != device_) {
        x = x.to(device_);
    }
    
    // Если маска задана и на другом устройстве - перемещаем
    if (attn_mask.defined() && attn_mask.device() != device_) {
        attn_mask = attn_mask.to(device_);
    }

    // Pre-LN: Self-Attention
    // MultiheadAttention ожидает (seq_len, batch, features) если batch_first=false
    // Транспонируем если нужно: (batch, seq_len, features) -> (seq_len, batch, features)
    torch::Tensor residual = x;
    torch::Tensor x_norm = norm1_->forward(x);
    
    // Транспонируем для attention: (batch, seq, features) -> (seq, batch, features)
    x_norm = x_norm.transpose(0, 1);
    
    // MultiheadAttention возвращает tuple (output, attention_weights)
    auto attn_result = attn_->forward(x_norm, x_norm, x_norm, attn_mask);
    torch::Tensor attn_out = std::get<0>(attn_result);
    
    // Транспонируем обратно: (seq, batch, features) -> (batch, seq, features)
    attn_out = attn_out.transpose(0, 1);
    
    x = residual + dropout1_->forward(attn_out);

    // Pre-LN: FFN (Perceptron)
    residual = x;
    x_norm = norm2_->forward(x);
    x = residual + dropout2_->forward(ffn_->forward(x_norm));

    return x; // Остается на GPU
}

