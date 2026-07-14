
#include "logger/logger.hpp"
#include "model/experimental_llm.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>

// ----------------------- UTF-8 токенизатор -----------------------
class UTF8Tokenizer {
public:
    std::unordered_map<std::string, int> token2idx;
    std::vector<std::string> idx2token;
    int vocab_size = 0;

    static std::string utf8_char_at(const std::string& str, size_t& pos) {
        if (pos >= str.size()) return "";
        unsigned char c = static_cast<unsigned char>(str[pos]);
        size_t len = 1;
        if      ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else { pos++; return "?"; }
        std::string ch = str.substr(pos, len);
        pos += len;
        return ch;
    }

    explicit UTF8Tokenizer(const std::string& text) {
        for (size_t i = 0; i < text.size(); ) {
            std::string ch = utf8_char_at(text, i);
            if (token2idx.find(ch) == token2idx.end()) {
                token2idx[ch] = idx2token.size();
                idx2token.push_back(ch);
            }
        }
        vocab_size = idx2token.size();
    }

    std::vector<int> encode(const std::string& str) const {
        std::vector<int> tokens;
        for (size_t i = 0; i < str.size(); ) {
            std::string ch = utf8_char_at(str, i);
            auto it = token2idx.find(ch);
            if (it != token2idx.end()) {
                tokens.push_back(it->second);
            }
        }
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) const {
        std::string result;
        for (int idx : tokens) {
            if (idx >= 0 && idx < static_cast<int>(idx2token.size())) {
                result += idx2token[idx];
            }
        }
        return result;
    }
};

// ----------------------- Загрузка текста -----------------------
std::string load_text() {
    std::ifstream file("input.txt");
    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    return "Привет мир! Это тестовый текст для обучения.";
}

// ----------------------- Потоковая генерация -----------------------
void generate_streaming(
    ExperimentalLLM& model,
    const UTF8Tokenizer& tokenizer,
    const std::string& prompt,
    int max_new_tokens,
    int context_len,
    double temperature = 0.8,
    int top_k = 20)
{
    model->eval();
    auto device = model->parameters().front().device();

    auto tokens = tokenizer.encode(prompt);
    if (tokens.empty()) {
        std::cout << "(no valid tokens)" << std::flush;
        return;
    }

    auto input = torch::tensor(tokens, torch::kLong).unsqueeze(0).to(device);
    int vocab_size = tokenizer.vocab_size;

    std::cout << prompt << std::flush;

    for (int i = 0; i < max_new_tokens; ++i) {
        if (input.size(1) > context_len) {
            input = input.slice(1, input.size(1) - context_len, input.size(1));
        }

        torch::NoGradGuard no_grad;
        auto logits = model->forward(input);
        auto next_logits = logits[0][-1] / temperature;

        // Обрезаем до размера словаря
        if (next_logits.size(0) > vocab_size) {
            next_logits = next_logits.slice(0, 0, vocab_size);
        }

        // Top-k фильтрация
        if (top_k > 0 && top_k < vocab_size) {
            auto [values, indices] = torch::topk(next_logits, top_k);
            auto min_val = values[-1];
            next_logits = torch::where(next_logits < min_val, -1e9, next_logits);
        }

        auto probs = torch::softmax(next_logits, -1);
        auto next_token = torch::multinomial(probs, 1).item<int64_t>();

        if (next_token >= vocab_size) break;  // защита от выхода за границы

        std::cout << tokenizer.decode({static_cast<int>(next_token)}) << std::flush;

        input = torch::cat({input, torch::tensor({{next_token}}, device)}, 1);
    }
    std::cout << std::endl;
}

// ----------------------- Интерактивный режим -----------------------
void interactive_mode(
    ExperimentalLLM& model,
    const UTF8Tokenizer& tokenizer,
    int context_len,
    int max_new_tokens = 200,
    double temperature = 0.8,
    int top_k = 20)
{
    /*
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Интерактивный режим генерации" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Параметры:" << std::endl;
    std::cout << "  max_new_tokens: " << max_new_tokens << std::endl;
    std::cout << "  temperature: " << temperature << std::endl;
    std::cout << "  top_k: " << top_k << std::endl;
    std::cout << "  context_len: " << context_len << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nКоманды:" << std::endl;
    std::cout << "  :q, :quit    - выход" << std::endl;
    std::cout << "  :t VALUE     - температура (0.1-5.0)" << std::endl;
    std::cout << "  :k VALUE     - top_k (0=выкл, 5-100)" << std::endl;
    std::cout << "  :n VALUE     - макс. токенов (10-2000)" << std::endl;
    std::cout << "  :h, :help    - помощь" << std::endl;
    std::cout << "========================================\n" << std::endl;
    */

    std::string prompt;
    while (true) {
        std::cout << "Запрос> " << std::flush;
        std::getline(std::cin, prompt);

        if (prompt.empty()) continue;

        if (prompt == ":q" || prompt == ":quit") {
            std::cout << "Выход." << std::endl;
            break;
        }
        if (prompt == ":h" || prompt == ":help") {
            std::cout << "\nКоманды:  :q  :t VAL  :k VAL  :n VAL  :h\n" << std::endl;
            continue;
        }
        if (prompt.rfind(":t ", 0) == 0) {
            temperature = std::stod(prompt.substr(3));
            std::cout << "[temperature = " << temperature << "]" << std::endl;
            continue;
        }
        if (prompt.rfind(":k ", 0) == 0) {
            top_k = std::stoi(prompt.substr(3));
            std::cout << "[top_k = " << top_k << "]" << std::endl;
            continue;
        }
        if (prompt.rfind(":n ", 0) == 0) {
            max_new_tokens = std::stoi(prompt.substr(3));
            std::cout << "[max_new_tokens = " << max_new_tokens << "]" << std::endl;
            continue;
        }

        std::cout << "\n";
        generate_streaming(model, tokenizer, prompt,
                          max_new_tokens, context_len,
                          temperature, top_k);
        std::cout << std::endl;
    }
}

// ----------------------- Главная функция -----------------------
int main() {
    Logger lg(logging::CONSOLE);
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    if (!torch::cuda::is_available()) {
        lg.warn("CUDA не доступна, обучение на CPU");
    }

    // 1. Загрузка текста
    std::string text = load_text();
    lg.info("Загружен текст: " + std::to_string(text.size()) + " байт");

    UTF8Tokenizer tokenizer(text);
    lg.info("Размер словаря: " + std::to_string(tokenizer.vocab_size));

    std::vector<int> data = tokenizer.encode(text);
    lg.info("Токенов: " + std::to_string(data.size()));

    // 2. Конфигурация модели
    ModelConfig cfg;
    cfg.vocab_size = tokenizer.vocab_size;
    cfg.sequence_length = 64;           // длина контекста
    cfg.hidden_size = 384;              // УВЕЛИЧЕНО для 115 классов
    cfg.attention_heads = 6;            // 384/6 = 64 на голову
    cfg.ffn_hidden = 1536;              // 384 * 4
    cfg.first_transformers = 2;
    cfg.custom_layers = 6;              // УМЕНЬШЕНО для стабильности
    cfg.last_transformers = 2;
    cfg.dropout = 0.1;
    cfg.shared_mlp_hidden = {192};      // промежуточный слой
    cfg.private_mlp_hidden = {192};     // промежуточный слой 192

    lg.info("Параметры модели:");
    lg.info("  hidden_size: " + std::to_string(cfg.hidden_size));
    lg.info("  layers: " + std::to_string(cfg.first_transformers + cfg.custom_layers + cfg.last_transformers));

    ExperimentalLLM model(cfg);
    model->to(device);
    model->train();

    // 3. Параметры обучения
    const int seq_len = cfg.sequence_length;
    const int batch_size = 64;
    const int num_samples = static_cast<int>(data.size()) - seq_len;

    if (num_samples <= 0) {
        lg.error("Текст слишком короткий для длины последовательности " + std::to_string(seq_len));
        return 1;
    }

    lg.info("Обучающих примеров: " + std::to_string(num_samples));

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(3e-4).weight_decay(0.01)
    );

    const int epochs = 15;  // больше эпох
    const int steps_per_epoch = std::min(500, num_samples / batch_size);

    lg.info("Начало обучения (" + std::to_string(epochs) + " эпох по " + std::to_string(steps_per_epoch) + " шагов)...");

    // 4. Обучение
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (int step = 0; step < steps_per_epoch; ++step) {
            std::vector<torch::Tensor> batch_inputs, batch_targets;
            for (int b = 0; b < batch_size; ++b) {
                int start = rand() % num_samples;
                batch_inputs.push_back(
                    torch::tensor(std::vector<int64_t>(data.begin() + start, data.begin() + start + seq_len), torch::kLong));
                batch_targets.push_back(
                    torch::tensor(std::vector<int64_t>(data.begin() + start + 1, data.begin() + start + seq_len + 1), torch::kLong));
            }

            auto inputs = torch::stack(batch_inputs).to(device);
            auto targets = torch::stack(batch_targets).to(device);

            optimizer.zero_grad();
            auto logits = model->forward(inputs);
            auto loss = torch::nn::functional::cross_entropy(
                logits.reshape({-1, cfg.vocab_size}),
                targets.reshape({-1}),
                torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100));

            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();

            total_loss += loss.item<double>();
        }

        double avg_loss = total_loss / steps_per_epoch;
        double ppl = std::exp(avg_loss);
        lg.log("Эпоха " + std::to_string(epoch + 1) + "/" + std::to_string(epochs) +
               " | Loss: " + std::to_string(avg_loss).substr(0, 6) +
               " | Perplexity: " + std::to_string(static_cast<int>(ppl)));
    }

    // 5. Сохранение
    torch::save(model, "experimental_llm.pt");
    lg.info("Модель сохранена в experimental_llm.pt");

    // 6. Демо-генерация
    std::cout << "\n=== Демо-генерация ===\n" << std::endl;
    srand(static_cast<unsigned>(time(nullptr)));

    std::vector<std::string> prompts = {
        "Он посмотрел",
        "В комнате",
        "Она сказала"
    };

    for (const auto& prompt : prompts) {
        std::cout << "Запрос: ";
        generate_streaming(model, tokenizer, prompt, 100, cfg.sequence_length, 0.8, 20);
        std::cout << std::endl;
    }

    // 7. Интерактивный режим
    interactive_mode(model, tokenizer, cfg.sequence_length, 200, 0.8, 20);

    return 0;
}

