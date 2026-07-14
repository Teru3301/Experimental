
#include "model/modules/layer_norm.hpp"

LayerNormImpl::LayerNormImpl(
    int hidden_size,
    double eps)
{
    norm = register_module(
        "layer_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden_size}).eps(eps)
        ));
}

torch::Tensor LayerNormImpl::forward(torch::Tensor x)
{
    return norm(x);
}

