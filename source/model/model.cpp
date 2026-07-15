
#include "model/model.hpp"
#include <cmath>


TransformerModel::TransformerModel(int64_t vocab_size, int64_t d_model, int64_t max_seq_len,
                                   int64_t nhead, int64_t num_encoder_blocks, int64_t num_decoder_blocks,
                                   std::vector<int64_t> ffn_sizes, double dropout)
    : d_model_(d_model)
    , vocab_size_(vocab_size)
    , nhead_(nhead)
    , head_dim_(d_model / nhead)
{
    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // Embeddings (только токены, позиции через RoPE)
    token_embedding_ = register_module("token_embedding", 
        torch::nn::Embedding(vocab_size, d_model));
    
    embed_dropout_ = register_module("embed_dropout",
        torch::nn::Dropout(dropout));

    // Encoder blocks
    for (int64_t i = 0; i < num_encoder_blocks; ++i) {
        auto block = std::make_shared<TransformerEncoderBlock>(
            d_model, nhead, ffn_sizes, dropout);
        encoder_blocks_.push_back(register_module(
            "encoder_block_" + std::to_string(i), block));
    }

    // Decoder blocks
    for (int64_t i = 0; i < num_decoder_blocks; ++i) {
        auto block = std::make_shared<TransformerDecoderBlock>(
            d_model, nhead, ffn_sizes, dropout);
        decoder_blocks_.push_back(register_module(
            "decoder_block_" + std::to_string(i), block));
    }

    // Output projection
    output_proj_ = register_module("output_proj",
        torch::nn::Linear(d_model, vocab_size));
    
    final_norm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));

    // Предвычисляем RoPE до разумной длины
    build_rope_cache(max_seq_len);

    // Перемещаем всё на GPU
    this->to(device_);
}


void TransformerModel::build_rope_cache(int64_t max_len)
{
    // theta_i = rope_theta ^ (-2i/d)
    // Вычисляем: 1.0 / (rope_theta ^ (arange(0, head_dim, 2) / head_dim))
    auto i = torch::arange(0, head_dim_, 2, torch::kFloat64);
    auto theta = torch::pow(rope_theta_, -i / head_dim_);
    
    // Позиции: (max_len, 1)
    auto positions = torch::arange(max_len, torch::kFloat64).unsqueeze(1);
    
    // Частоты: (max_len, head_dim/2)
    auto freqs = positions * theta.unsqueeze(0);
    
    // (max_len, head_dim) — дублируем для cos и sin половин
    auto emb = torch::cat(std::vector<torch::Tensor>{freqs, freqs}, -1);
    
    rope_cos_ = emb.cos().to(torch::kFloat32).to(device_);
    rope_sin_ = emb.sin().to(torch::kFloat32).to(device_);
}


torch::Tensor TransformerModel::apply_rope(torch::Tensor x, int64_t start_pos)
{
    // x: (batch, seq, d_model) — применяем RoPE ко всем головам сразу
    int64_t seq_len = x.size(1);
    int64_t batch_size = x.size(0);
    
    // Разбиваем на головы: (batch, seq, d_model) -> (batch, seq, nhead, head_dim)
    auto x_reshaped = x.reshape({batch_size, seq_len, nhead_, head_dim_});
    
    // Берём нужный фрагмент кэша
    auto cos = rope_cos_.slice(0, start_pos, start_pos + seq_len)
        .unsqueeze(0).unsqueeze(2);  // (1, seq, 1, head_dim)
    auto sin = rope_sin_.slice(0, start_pos, start_pos + seq_len)
        .unsqueeze(0).unsqueeze(2);
    
    // Поворачиваем: первая половина и вторая половина head_dim
    auto x1 = x_reshaped.slice(-1, 0, head_dim_ / 2);
    auto x2 = x_reshaped.slice(-1, head_dim_ / 2, head_dim_);
    
    auto cos1 = cos.slice(-1, 0, head_dim_ / 2);
    auto sin1 = sin.slice(-1, 0, head_dim_ / 2);
    auto cos2 = cos.slice(-1, head_dim_ / 2, head_dim_);
    auto sin2 = sin.slice(-1, head_dim_ / 2, head_dim_);
    
    auto rotated = torch::cat({
        x1 * cos1 - x2 * sin1,
        x2 * cos2 + x1 * sin2
    }, -1);
    
    // Обратно в (batch, seq, d_model)
    return rotated.reshape({batch_size, seq_len, d_model_});
}


torch::Tensor TransformerModel::forward(torch::Tensor src_ids, torch::Tensor tgt_ids,
                                        torch::Tensor src_padding_mask, torch::Tensor tgt_padding_mask)
{
    // Encode
    torch::Tensor encoder_out = encode(src_ids, src_padding_mask);
    
    // Decode
    torch::Tensor decoder_out = decode(tgt_ids, encoder_out, tgt_padding_mask);
    
    return decoder_out;
}


torch::Tensor TransformerModel::encode(torch::Tensor src_ids, torch::Tensor src_padding_mask)
{
    if (src_ids.device() != device_) {
        src_ids = src_ids.to(device_);
    }
    
    if (src_padding_mask.defined() && src_padding_mask.device() != device_) {
        src_padding_mask = src_padding_mask.to(device_);
    }

    // Token embeddings + масштабирование
    auto x = token_embedding_->forward(src_ids) * std::sqrt(static_cast<double>(d_model_));
    
    // Применяем RoPE
    x = apply_rope(x);
    x = embed_dropout_->forward(x);

    // Encoder blocks
    for (auto& block : encoder_blocks_) {
        x = block->forward(x, src_padding_mask);
    }

    // Транспонируем для decoder cross-attention: (batch, seq, features) -> (seq, batch, features)
    return x.transpose(0, 1);
}


torch::Tensor TransformerModel::decode(torch::Tensor tgt_ids, torch::Tensor encoder_out,
                                       torch::Tensor tgt_padding_mask)
{
    if (tgt_ids.device() != device_) {
        tgt_ids = tgt_ids.to(device_);
    }
    
    if (encoder_out.device() != device_) {
        encoder_out = encoder_out.to(device_);
    }
    
    if (tgt_padding_mask.defined() && tgt_padding_mask.device() != device_) {
        tgt_padding_mask = tgt_padding_mask.to(device_);
    }

    // Token embeddings + масштабирование
    auto x = token_embedding_->forward(tgt_ids) * std::sqrt(static_cast<double>(d_model_));
    
    // Применяем RoPE
    x = apply_rope(x);
    x = embed_dropout_->forward(x);

    // Decoder blocks
    for (auto& block : decoder_blocks_) {
        x = block->forward(x, encoder_out, {}, tgt_padding_mask);
    }

    // Final norm + output projection
    x = final_norm_->forward(x);
    return output_proj_->forward(x);
}

