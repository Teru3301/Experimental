
#include <memory>
#include "model/gpt_block.hpp"


GPTBlock::GPTBlock(int64_t d_model, int64_t nhead, int64_t max_seq_len,
                   std::vector<int64_t> ffn_sizes, double dropout)
{
    attn_ = register_module("attn", 
        std::make_shared<Attention>(d_model, nhead, max_seq_len, dropout));

    norm1_ = register_module("norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));

    norm2_ = register_module("norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));

    dropout1_ = register_module("dropout1", torch::nn::Dropout(dropout));
    dropout2_ = register_module("dropout2", torch::nn::Dropout(dropout));

    std::vector<int64_t> full_ffn;
    full_ffn.push_back(d_model);
    for (auto s : ffn_sizes) full_ffn.push_back(s);
    full_ffn.push_back(d_model);

    ffn_ = register_module("ffn", std::make_shared<Perceptron>(full_ffn));
}


torch::Tensor GPTBlock::forward(torch::Tensor x)
{
    // Pre-LN → Attention → Dropout → Residual
    auto residual = x;
    x = norm1_->forward(x);
    x = attn_->forward(x);
    x = dropout1_->forward(x);
    x = residual + x;

    // Pre-LN → FFN → Dropout → Residual
    residual = x;
    x = norm2_->forward(x);
    x = ffn_->forward(x);
    x = dropout2_->forward(x);
    x = residual + x;

    return x;
}


torch::Tensor GPTBlock::forward_kv(torch::Tensor x, KVCache& cache, int64_t pos)
{
    // Pre-LN → Attention (KV-cache) → Residual
    auto residual = x;
    x = norm1_->forward(x);
    x = attn_->forward_kv(x, cache, pos);
    x = residual + x;

    // Pre-LN → FFN → Residual (без dropout в инференсе)
    residual = x;
    x = norm2_->forward(x);
    x = ffn_->forward(x);
    x = residual + x;

    return x;
}

