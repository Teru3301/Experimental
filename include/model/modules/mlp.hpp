
#pragma once

#include <torch/torch.h>
#include <vector>

class MLPImpl : public torch::nn::Module
{
public:
    MLPImpl(
        int input_size,
        const std::vector<int>& hidden_layers,
        int output_size,
        bool bias = true
    );

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential network{nullptr};
};

TORCH_MODULE(MLP);

