
#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include "model/transformer_encoder_block.hpp"
#include "model/transformer_decoder_block.hpp"



class TransformerModel : public torch::nn::Module
{
    torch::nn::Embedding token_embedding_{nullptr};
    torch::nn::Dropout embed_dropout_{nullptr};
    
    std::vector<std::shared_ptr<TransformerEncoderBlock>> encoder_blocks_;
    std::vector<std::shared_ptr<TransformerDecoderBlock>> decoder_blocks_;
    
    torch::nn::Linear output_proj_{nullptr};
    torch::nn::LayerNorm final_norm_{nullptr};
    
    int64_t d_model_;
    int64_t vocab_size_;
    int64_t nhead_;
    int64_t head_dim_;
    double rope_theta_{10000.0};
    torch::Device device_{torch::kCPU};
    
    torch::Tensor rope_cos_{};
    torch::Tensor rope_sin_{};
    
    void build_rope_cache(int64_t max_len);
    torch::Tensor apply_rope(torch::Tensor x, int64_t start_pos = 0);

public:
    TransformerModel(int64_t vocab_size, int64_t d_model, int64_t max_seq_len, 
                     int64_t nhead, int64_t num_encoder_blocks, int64_t num_decoder_blocks,
                     std::vector<int64_t> ffn_sizes, double dropout = 0.1);
    
    torch::Tensor forward(torch::Tensor src_ids, torch::Tensor tgt_ids,
                          torch::Tensor src_padding_mask = {}, torch::Tensor tgt_padding_mask = {});
    
    torch::Tensor encode(torch::Tensor src_ids, torch::Tensor src_padding_mask = {});
    torch::Tensor decode(torch::Tensor tgt_ids, torch::Tensor encoder_out,
                         torch::Tensor tgt_padding_mask = {});
    
    // Генерация с KV-cache
    torch::Tensor generate(torch::Tensor src_ids, int64_t max_len, int64_t bos_token = 1);
    
    torch::Device device() const { return device_; }
};

