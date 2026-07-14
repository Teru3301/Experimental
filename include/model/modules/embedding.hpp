
#pragma once

#include <torch/torch.h>

class EmbeddingImpl : public torch::nn::Module
{
public:

    EmbeddingImpl(
        int vocab_size,
        int sequence_length,
        int hidden_size
    );

    torch::Tensor forward(torch::Tensor tokens);

private:

    torch::nn::Embedding token_embedding{nullptr};

    torch::nn::Embedding position_embedding{nullptr};
};

TORCH_MODULE(Embedding);

