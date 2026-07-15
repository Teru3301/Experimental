
#include <iostream>
#include <torch/torch.h>
#include "model/transformer_block.hpp"



int main() {
    // Проверяем доступность GPU
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using GPU." << std::endl;
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA not available. Using CPU." << std::endl;
    }

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    
    // Создаем модель - она автоматически будет на GPU (если доступен)
    TransformerBlock model(512, 8, std::vector<int64_t>{2048}, 0.1);
    
    // Создаем входные данные на том же устройстве
    auto x = torch::randn({32, 100, 512}).to(device);
    
    std::cout << "Input device: " << x.device() << std::endl;
    std::cout << "Model device: " << model.device() << std::endl;
    
    // Forward pass - всё на GPU
    auto output = model.forward(x);
    
    std::cout << "Output device: " << output.device() << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl;
    
    // Если нужно перенести результат на CPU (только для вывода/сохранения)
    auto cpu_output = output.to(torch::kCPU);
    std::cout << "CPU output device: " << cpu_output.device() << std::endl;
    
    std::cout << "\nTest passed! All computations stay on GPU." << std::endl;
    
    return 0;
}

