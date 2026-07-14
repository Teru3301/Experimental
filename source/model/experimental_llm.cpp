
#include "model/experimental_llm.hpp"

ExperimentalLLMImpl::ExperimentalLLMImpl(ModelConfig config)
    : cfg_(std::move(config))
{
    embedding_ = register_module("embedding",
        Embedding(cfg_.vocab_size, cfg_.sequence_length, cfg_.hidden_size));

    for (int i = 0; i < cfg_.first_transformers; ++i)
    {
        first_blocks_.push_back(register_module(
            "first_block_" + std::to_string(i),
            TransformerBlock(cfg_.hidden_size, cfg_.attention_heads,
                             cfg_.ffn_hidden, cfg_.dropout)));
    }

    int shared_dim = cfg_.hidden_size / 2;
    shared_mlp_ = register_module("shared_mlp",
        MLP(cfg_.hidden_size, cfg_.shared_mlp_hidden, shared_dim));

    for (int i = 0; i < cfg_.custom_layers; ++i)
    {
        custom_blocks_.push_back(register_module(
            "custom_block_" + std::to_string(i),
            CustomBlock(shared_mlp_, cfg_.hidden_size, shared_dim,
                        cfg_.attention_heads, cfg_.ffn_hidden,
                        cfg_.private_mlp_hidden, cfg_.dropout)));
    }

    for (int i = 0; i < cfg_.last_transformers; ++i)
    {
        last_blocks_.push_back(register_module(
            "last_block_" + std::to_string(i),
            TransformerBlock(cfg_.hidden_size, cfg_.attention_heads,
                             cfg_.ffn_hidden, cfg_.dropout)));
    }

    final_norm_ = register_module("final_norm", LayerNorm(cfg_.hidden_size));
    head_ = register_module("head", torch::nn::Linear(cfg_.hidden_size, cfg_.vocab_size));
}

torch::Tensor ExperimentalLLMImpl::forward(torch::Tensor input_ids)
{
    auto x = embedding_->forward(input_ids);   // [B, T, H]

    int seq_len = x.size(1);
    auto mask = torch::ones({seq_len, seq_len}, x.device()).tril();
    mask = mask.unsqueeze(0).unsqueeze(1);     // [1, 1, T, T]

    for (auto& block : first_blocks_)
        x = block->forward(x, mask);
    for (auto& block : custom_blocks_)
        x = block->forward(x, mask);
    for (auto& block : last_blocks_)
        x = block->forward(x, mask);

    x = final_norm_->forward(x);
    return head_->forward(x);   // [B, T, vocab_size]
}

