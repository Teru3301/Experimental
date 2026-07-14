
#include "model/modules/mlp.hpp"

MLPImpl::MLPImpl(
    int input_size,
    const std::vector<int>& hidden_layers,
    int output_size,
    bool bias)
{
    network = register_module("network", torch::nn::Sequential());

    int current = input_size;

    for (size_t i = 0; i < hidden_layers.size(); ++i)
    {
        network->push_back(
            torch::nn::Linear(
                torch::nn::LinearOptions(current, hidden_layers[i]).bias(bias)
            )
        );

        network->push_back(torch::nn::GELU());

        current = hidden_layers[i];
    }

    network->push_back(
        torch::nn::Linear(
            torch::nn::LinearOptions(current, output_size).bias(bias)
        )
    );
}

torch::Tensor MLPImpl::forward(torch::Tensor x)
{
    return network->forward(x);
}

