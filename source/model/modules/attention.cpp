
#include "model/modules/attention.hpp"

#include <cmath>

MultiHeadAttentionImpl::MultiHeadAttentionImpl(
    int hidden_size,
    int num_heads,
    double dropout_p,
    bool bias)
    :
    hidden_size_(hidden_size),
    num_heads_(num_heads),
    head_dim_(hidden_size / num_heads)
{
    if (hidden_size % num_heads != 0)
        throw std::runtime_error(
            "hidden_size must be divisible by num_heads");


q_proj = register_module(
    "q_proj",
    torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size, hidden_size).bias(bias)));

k_proj = register_module(
    "k_proj",
    torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size, hidden_size).bias(bias)));

v_proj = register_module(
    "v_proj",
    torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size, hidden_size).bias(bias)));

out_proj = register_module(
    "out_proj",
    torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size, hidden_size).bias(bias)));


    dropout = register_module(
        "dropout",
        torch::nn::Dropout(dropout_p));
}

torch::Tensor MultiHeadAttentionImpl::forward(
    torch::Tensor x,
    torch::Tensor mask)
{
    const auto batch = x.size(0);
    const auto seq = x.size(1);

    auto q = q_proj(x);
    auto k = k_proj(x);
    auto v = v_proj(x);

    q = q.view({batch, seq, num_heads_, head_dim_}).transpose(1, 2);
    k = k.view({batch, seq, num_heads_, head_dim_}).transpose(1, 2);
    v = v.view({batch, seq, num_heads_, head_dim_}).transpose(1, 2);

    auto scores = torch::matmul(
        q,
        k.transpose(-2, -1));

    scores /= std::sqrt((double)head_dim_);

    if (mask.defined())
    {
        scores = scores.masked_fill(mask == 0, -1e9);
    }

    auto attn = torch::softmax(scores, -1);

    attn = dropout(attn);

    auto context = torch::matmul(attn, v);

    context = context.transpose(1, 2).contiguous();

    context = context.view(
        {batch, seq, hidden_size_});

    return out_proj(context);
}

