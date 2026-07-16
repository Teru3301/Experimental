
#pragma once

#include <torch/torch.h>



class Perceptron : public torch::nn::Module
{
    torch::nn::Sequential layers_{nullptr};

public:
    Perceptron(std::vector<int64_t> sizes);
    torch::Tensor forward(torch::Tensor x);
    void fit(torch::Tensor inputs, torch::Tensor targets, int epochs = 10, int batch = 64, double lr = 0.001);
    torch::Tensor predict(torch::Tensor x);
};

