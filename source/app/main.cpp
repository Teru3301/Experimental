
#include <torch/torch.h>
#include <iostream>
#include <random>
#include "model/model.hpp"


// Генератор случайных последовательностей для теста
std::pair<torch::Tensor, torch::Tensor> generate_batch(int64_t batch_size, int64_t seq_len, int64_t vocab_size)
{
    auto src = torch::randint(1, vocab_size, {batch_size, seq_len});
    auto tgt = torch::randint(1, vocab_size, {batch_size, seq_len});
    return {src, tgt};
}


int main()
{
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    
    // Параметры модели
    int64_t vocab_size = 1000;
    int64_t d_model = 256;
    int64_t max_seq_len = 32768;  // RoPE позволяет большой контекст
    int64_t nhead = 4;
    int64_t num_encoder_blocks = 3;
    int64_t num_decoder_blocks = 3;
    std::vector<int64_t> ffn_sizes = {512};
    double dropout = 0.1;
    
    // Создаём модель
    auto model = std::make_shared<TransformerModel>(
        vocab_size, d_model, max_seq_len, nhead,
        num_encoder_blocks, num_decoder_blocks,
        ffn_sizes, dropout);
    
    model->train();
    
    // Оптимизатор и функция потерь
    auto opt = torch::optim::Adam(model->parameters(), 0.0001);
    auto loss_fn = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().ignore_index(0));
    
    int64_t batch_size = 16;
    int64_t seq_len = 32;
    int epochs = 100;
    
    std::cout << "Starting training..." << std::endl;
    
    for (int e = 0; e < epochs; ++e) {
        // Генерируем случайные данные
        auto [src, tgt] = generate_batch(batch_size, seq_len, vocab_size);
        
        // Перемещаем данные на устройство модели
        auto device = model->device();
        src = src.to(device);
        tgt = tgt.to(device);
        
        // tgt_input: все токены кроме последнего
        // tgt_output: все токены кроме первого
        auto tgt_input = tgt.slice(1, 0, seq_len - 1);
        auto tgt_output = tgt.slice(1, 1, seq_len);
        
        opt.zero_grad();
        
        // Forward pass
        auto output = model->forward(src, tgt_input);
        
        // output: (batch, seq_len-1, vocab_size)
        // tgt_output: (batch, seq_len-1)
        output = output.reshape({-1, vocab_size});
        tgt_output = tgt_output.reshape({-1});
        
        auto loss = loss_fn->forward(output, tgt_output);
        loss.backward();
        
        // Gradient clipping
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        
        opt.step();
        
        if ((e + 1) % 10 == 0) {
            std::cout << "Epoch " << e + 1 << "/" << epochs 
                      << " | Loss: " << loss.item<float>() << std::endl;
        }
    }
    
    // Сохраняем модель
    torch::save(model, "transformer_model.pt");
    std::cout << "Model saved to transformer_model.pt" << std::endl;
    
    // Тест генерации
    model->eval();
    torch::NoGradGuard no_grad;
    
    std::cout << "\nTesting generation..." << std::endl;
    
    // Начальный токен
    auto device = model->device();
    auto src_input = torch::randint(1, vocab_size, {1, 10}, 
        torch::TensorOptions().dtype(torch::kLong).device(device));
    auto tgt_input = torch::tensor({{1}}, 
        torch::TensorOptions().dtype(torch::kLong).device(device));
    
    int64_t max_generate = 20;
    
    for (int64_t i = 0; i < max_generate; ++i) {
        auto output = model->forward(src_input, tgt_input);
        
        // Берём последний токен
        auto next_token_logits = output.slice(1, -1, output.size(1)).squeeze(1);
        auto next_token = next_token_logits.argmax(1).unsqueeze(0);
        
        tgt_input = torch::cat({tgt_input, next_token}, 1);
        
        if (next_token.item<int64_t>() == 2) {  // <eos> token
            break;
        }
    }
    
    std::cout << "Generated sequence: " << tgt_input << std::endl;
    
    return 0;
}

