
#include "model/gpt_model.hpp"
#include <cmath>


GPTModel::GPTModel(int64_t vocab_size, int64_t d_model, int64_t max_seq_len,
                   int64_t nhead, int64_t num_blocks,
                   std::vector<int64_t> ffn_sizes, double dropout)
    : d_model_(d_model)
    , vocab_size_(vocab_size)
    , max_seq_len_(max_seq_len)
{
    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    token_embedding_ = register_module("token_embedding", 
        torch::nn::Embedding(vocab_size, d_model));
    
    embed_dropout_ = register_module("embed_dropout",
        torch::nn::Dropout(dropout));

    for (int64_t i = 0; i < num_blocks; ++i) {
        auto block = std::make_shared<GPTBlock>(
            d_model, nhead, max_seq_len, ffn_sizes, dropout);
        blocks_.push_back(register_module(
            "block_" + std::to_string(i), block));
    }

    final_norm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    
    lm_head_ = register_module("lm_head",
        torch::nn::Linear(d_model, vocab_size));
    lm_head_->weight = token_embedding_->weight;

    this->to(device_);
}


torch::Tensor GPTModel::forward(torch::Tensor x)
{
    if (x.device() != device_) x = x.to(device_);
    
    auto h = token_embedding_->forward(x) * std::sqrt(static_cast<double>(d_model_));
    h = embed_dropout_->forward(h);
    
    for (auto& block : blocks_) {
        h = block->forward(h);
    }
    
    h = final_norm_->forward(h);
    return lm_head_->forward(h);
}


torch::Tensor GPTModel::generate(torch::Tensor prompt, int64_t max_new_tokens,
                                 double temperature, int64_t top_k)
{
    this->eval();
    torch::NoGradGuard no_grad;
    
    if (prompt.device() != device_) prompt = prompt.to(device_);
    
    int64_t prompt_len = prompt.size(1);
    TORCH_CHECK(
        prompt_len + max_new_tokens <= max_seq_len_,
        "Sequence exceeds max_seq_len");
    auto h = token_embedding_->forward(prompt) * std::sqrt(static_cast<double>(d_model_));
    
    std::vector<KVCache> caches(blocks_.size());
    
/*
    // Префилл: по одному токену
    for (int64_t i = 0; i < prompt_len; ++i) {
        auto token_h = h.index({torch::indexing::Slice(), 
                                 torch::indexing::Slice(i, i + 1),
                                 torch::indexing::Slice()});
        for (size_t b = 0; b < blocks_.size(); ++b) {
            token_h = blocks_[b]->forward_kv(token_h, caches[b], i);
        }
    }
    
    auto tokens = prompt;  // (1, prompt_len)
    
    // Эмбеддинг последнего токена для старта генерации
    auto last_h = h.index({torch::indexing::Slice(),
                            torch::indexing::Slice(-1, torch::indexing::None),
                            torch::indexing::Slice()});
*/



torch::Tensor last_h;

// Префилл: пропускаем все токены через все блоки,
// одновременно заполняя KV-cache.
for (int64_t i = 0; i < prompt_len; ++i)
{
    last_h = h.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(i, i + 1),
        torch::indexing::Slice()
    });

    for (size_t b = 0; b < blocks_.size(); ++b)
    {
        last_h = blocks_[b]->forward_kv(last_h, caches[b], i);
    }
}

auto tokens = prompt;



    
    for (int64_t step = 0; step < max_new_tokens; ++step) {
        int64_t pos = prompt_len + step;
        
        for (size_t b = 0; b < blocks_.size(); ++b) {
            last_h = blocks_[b]->forward_kv(last_h, caches[b], pos);
        }
        
        last_h = final_norm_->forward(last_h);         // (1, 1, d_model)
        auto logits = lm_head_->forward(last_h);        // (1, 1, vocab)
        
        // Берём логиты: (vocab,)
        auto next_logits = logits.index({0, 0, torch::indexing::Slice()});
        next_logits = next_logits.slice(0, 0, vocab_size_);
        
        if (temperature > 0) next_logits = next_logits / temperature;
        
        int64_t k = top_k;
        if (k > 0 && k < next_logits.size(0)) {
            auto [values, indices] = torch::topk(next_logits, k);
            auto min_val = values[-1];
            next_logits = torch::where(next_logits < min_val, -1e9, next_logits);
        }
        
        auto probs = torch::softmax(next_logits, -1);
        auto next_token = torch::multinomial(probs, 1);  // (1,)
        next_token = next_token.unsqueeze(0);             // (1, 1)
        
        tokens = torch::cat({tokens, next_token}, 1);    // (1, L) cat (1, 1) = ok
        
        // Эмбеддинг нового токена для следующего шага
        last_h = token_embedding_->forward(next_token) * std::sqrt(static_cast<double>(d_model_));
    }
    
    return tokens;
}

