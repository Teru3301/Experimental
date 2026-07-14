#pragma once

#include <vector>

struct ModelConfig
{
    int vocab_size = 32000;

    int sequence_length = 512;

    int hidden_size = 768;

    int attention_heads = 12;

    int ffn_hidden = 3072;

    int first_transformers = 2;

    int custom_layers = 8;

    int last_transformers = 2;

    std::vector<int> shared_mlp_hidden;

    std::vector<int> private_mlp_hidden;

    double dropout = 0.1;
};
