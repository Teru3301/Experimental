
#include "model/modules/custom_block.hpp"

CustomBlockImpl::CustomBlockImpl(
    MLP shared_mlp,
    int hidden_size,
    int shared_dim,
    int attention_heads,
    int ffn_hidden,
    const std::vector<int>& private_hidden,
    double dropout)
{
    shared_dim_ = shared_dim;
    private_dim_ = hidden_size - shared_dim;

    shared_mlp_ = shared_mlp;   // не регистрируем!

    private_mlp_ = register_module(
        "private_mlp",
        MLP(hidden_size, private_hidden, private_dim_));

    shared_norm_ = register_module(
        "shared_norm",
        LayerNorm(shared_dim_));

    private_norm_ = register_module(
        "private_norm",
        LayerNorm(private_dim_));

    transformer_ = register_module(
        "transformer",
        TransformerBlock(
            hidden_size,
            attention_heads,
            ffn_hidden,
            dropout));
}

torch::Tensor CustomBlockImpl::forward(
    torch::Tensor x,
    torch::Tensor mask)
{
    auto shared_out = shared_mlp_->forward(x);    // [B, T, shared_dim_]
    shared_out = shared_norm_(shared_out);

    auto private_out = private_mlp_->forward(x);  // [B, T, private_dim_]
    private_out = private_norm_(private_out);

    auto combined = torch::cat({shared_out, private_out}, -1);
    return transformer_->forward(combined, mask);
}

