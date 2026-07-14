
#include "model/modules/transformer_block.hpp"

TransformerBlockImpl::TransformerBlockImpl(
    int hidden_size,
    int heads,
    int ffn_hidden,
    double dropout_p)
{
    norm1 = register_module(
        "norm1",
        LayerNorm(hidden_size));

    norm2 = register_module(
        "norm2",
        LayerNorm(hidden_size));

    attention = register_module(
        "attention",
        MultiHeadAttention(
            hidden_size,
            heads,
            dropout_p));

    ffn = register_module(
        "ffn",
        MLP(
            hidden_size,
            std::vector<int>{ffn_hidden},
            hidden_size));

    dropout = register_module(
        "dropout",
        torch::nn::Dropout(dropout_p));
}

torch::Tensor TransformerBlockImpl::forward(
    torch::Tensor x,
    torch::Tensor mask)
{
    auto residual = x;

    x = norm1(x);

    x = attention(x, mask);

    x = dropout(x);

    x = residual + x;

    residual = x;

    x = norm2(x);

    x = ffn(x);

    x = dropout(x);

    x = residual + x;

    return x;
}

