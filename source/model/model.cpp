
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

    token_embedding_ = register_module("token_embedding", 
        torch::nn::Embedding(vocab_size, d_model));
    
    embed_dropout_ = register_module("embed_dropout",
        torch::nn::Dropout(dropout));

    for (int64_t i = 0; i < num_encoder_blocks; ++i) {
        auto block = std::make_shared<TransformerEncoderBlock>(
            d_model, nhead, ffn_sizes, dropout);
        encoder_blocks_.push_back(register_module(
            "encoder_block_" + std::to_string(i), block));
    }

    for (int64_t i = 0; i < num_decoder_blocks; ++i) {
        auto block = std::make_shared<TransformerDecoderBlock>(
            d_model, nhead, ffn_sizes, dropout);
        decoder_blocks_.push_back(register_module(
            "decoder_block_" + std::to_string(i), block));
    }

    output_proj_ = register_module("output_proj",
        torch::nn::Linear(d_model, vocab_size));
    
    final_norm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));

    build_rope_cache(max_seq_len);

    this->to(device_);
}


void TransformerModel::build_rope_cache(int64_t max_len)
{
    auto i = torch::arange(0, head_dim_, 2, torch::kFloat64);
    auto theta = torch::pow(rope_theta_, -i / head_dim_);
    
    auto positions = torch::arange(max_len, torch::kFloat64).unsqueeze(1);
    
    auto freqs = positions * theta.unsqueeze(0);
    
    auto emb = torch::cat(std::vector<torch::Tensor>{freqs, freqs}, -1);
    
    rope_cos_ = emb.cos().to(torch::kFloat32).to(device_);
    rope_sin_ = emb.sin().to(torch::kFloat32).to(device_);
}


torch::Tensor TransformerModel::apply_rope(torch::Tensor x, int64_t start_pos)
{
    int64_t seq_len = x.size(1);
    int64_t batch_size = x.size(0);
    
    auto x_reshaped = x.reshape({batch_size, seq_len, nhead_, head_dim_});
    
    auto cos = rope_cos_.slice(0, start_pos, start_pos + seq_len)
        .unsqueeze(0).unsqueeze(2);
    auto sin = rope_sin_.slice(0, start_pos, start_pos + seq_len)
        .unsqueeze(0).unsqueeze(2);
    
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
    
    return rotated.reshape({batch_size, seq_len, d_model_});
}


torch::Tensor TransformerModel::forward(torch::Tensor src_ids, torch::Tensor tgt_ids,
                                        torch::Tensor src_padding_mask, torch::Tensor tgt_padding_mask)
{
    torch::Tensor encoder_out = encode(src_ids, src_padding_mask);
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

    auto x = token_embedding_->forward(src_ids) * std::sqrt(static_cast<double>(d_model_));
    
    x = apply_rope(x);
    x = embed_dropout_->forward(x);

    for (auto& block : encoder_blocks_) {
        x = block->forward(x, src_padding_mask);
    }

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

    auto x = token_embedding_->forward(tgt_ids) * std::sqrt(static_cast<double>(d_model_));
    
    x = apply_rope(x);
    x = embed_dropout_->forward(x);

    for (auto& block : decoder_blocks_) {
        x = block->forward(x, encoder_out, {}, tgt_padding_mask);
    }

    x = final_norm_->forward(x);
    return output_proj_->forward(x);
}

torch::Tensor TransformerModel::generate(torch::Tensor src_ids, int64_t max_len, int64_t bos_token)
{
    this->eval();
    torch::NoGradGuard no_grad;
    
    auto encoder_out = encode(src_ids);
    
    auto tgt = torch::full({1, 1}, bos_token, 
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    
    int64_t num_blocks = decoder_blocks_.size();
    std::vector<std::vector<torch::Tensor>> self_kv_caches(num_blocks);
    std::vector<std::vector<torch::Tensor>> cross_kv_caches(num_blocks);
    
    for (int64_t step = 0; step < max_len; ++step) {
        // Берём только последний токен как (1, 1)
        auto last_token = tgt.index({"...", 
            torch::indexing::Slice(tgt.size(1)-1, tgt.size(1))});
        
        auto x = token_embedding_->forward(last_token) 
            * std::sqrt(static_cast<double>(d_model_));
        
        x = apply_rope(x, step);
        
        for (int64_t i = 0; i < num_blocks; ++i) {
            x = decoder_blocks_[i]->forward_kv(x, encoder_out, 
                self_kv_caches[i], cross_kv_caches[i]);
        }
        
        x = final_norm_->forward(x);
        auto logits = output_proj_->forward(x);  // (1, 1, vocab_size)
        
        auto next_token = logits.argmax(-1);  // (1, 1) — сохраняет размерность!
        
        tgt = torch::cat({tgt, next_token}, 1);  // (1, N) + (1, 1) = (1, N+1)
    }
    
    return tgt;
}
