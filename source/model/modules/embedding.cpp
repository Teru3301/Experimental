
#include "model/modules/embedding.hpp"

EmbeddingImpl::EmbeddingImpl(
    int vocab_size,
    int sequence_length,
    int hidden_size)
{
    token_embedding = register_module(
        "token_embedding",
        torch::nn::Embedding(vocab_size, hidden_size));

    position_embedding = register_module(
        "position_embedding",
        torch::nn::Embedding(sequence_length, hidden_size));
}

torch::Tensor EmbeddingImpl::forward(torch::Tensor tokens)
{
    auto batch = tokens.size(0);
    auto seq = tokens.size(1);

    auto positions = torch::arange(
        seq,
        torch::TensorOptions().dtype(torch::kLong).device(tokens.device()));

    positions = positions.unsqueeze(0).expand({batch, seq});

    return token_embedding(tokens)
         + position_embedding(positions);
}

