
#include "model/perceptron.hpp"


Perceptron::Perceptron(std::vector<int64_t> sizes)
{
    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    layers_ = torch::nn::Sequential();
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        layers_->push_back(torch::nn::Linear(sizes[i], sizes[i + 1]));
        if (i < sizes.size() - 2) layers_->push_back(torch::nn::ReLU());
    }
    
    // Регистрируем ДО перемещения на устройство
    register_module("layers", layers_);
    // Перемещаем ВСЁ на GPU сразу после регистрации
    this->to(device_);
}


torch::Tensor Perceptron::forward(torch::Tensor x)
{
    // Проверяем, что тензор на правильном устройстве
    if (x.device() != device_) {
        x = x.to(device_);
    }
    return layers_->forward(x);
}


void Perceptron::fit(torch::Tensor inputs, torch::Tensor targets, int epochs, int batch, double lr)
{
    // Перемещаем данные на GPU один раз
    inputs = inputs.to(device_);
    targets = targets.to(device_);

    // Оптимизатор и функция потерь автоматически будут на GPU,
    // так как parameters() уже на GPU
    auto opt = torch::optim::Adam(this->parameters(), lr);
    auto loss_fn = torch::nn::CrossEntropyLoss();
    
    // Перемещаем loss function на GPU
    loss_fn->to(device_);
    
    this->train();

    for (int e = 0; e < epochs; ++e) {
        float loss_sum = 0;
        int n = 0;
        auto idx = torch::randperm(inputs.size(0), device_);

        for (int i = 0; i < inputs.size(0); i += batch) {
            int end = std::min(i + batch, (int)inputs.size(0));
            auto x = inputs.index_select(0, idx.slice(0, i, end));
            auto y = targets.index_select(0, idx.slice(0, i, end));

            opt.zero_grad();
            // x и y уже на GPU, forward возвращает на GPU
            auto output = this->forward(x);
            auto loss = loss_fn->forward(output, y);
            loss.backward();
            opt.step();

            loss_sum += loss.item<float>();
            n++;
        }
        printf("Epoch %d/%d | Loss: %.4f\n", e + 1, epochs, loss_sum / n);
    }
}


torch::Tensor Perceptron::predict(torch::Tensor x)
{
    this->eval();
    torch::NoGradGuard g;
    
    // Перемещаем вход на GPU если нужно
    if (x.device() != device_) {
        x = x.to(device_);
    }
    
    // ВСЕ вычисления на GPU, НЕ переносим на CPU!
    auto output = this->forward(x);
    return output.argmax(1); // Оставляем на GPU
}

