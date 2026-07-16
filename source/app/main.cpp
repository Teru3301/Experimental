
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "model/gpt_model.hpp"


// ══════════════════════════════════════════════
// UTF-8 Character Tokenizer
// ══════════════════════════════════════════════

class UTF8Tokenizer
{
    std::unordered_map<std::string, int64_t> token2idx_;
    std::vector<std::string> idx2token_;
    
    static std::string utf8_char(const std::string& str, size_t& pos)
    {
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
    
public:
    int64_t vocab_size() const { return idx2token_.size(); }
    
    UTF8Tokenizer()
    {
        idx2token_ = {"<unk>"};
        token2idx_["<unk>"] = 0;
    }
    
    void fit(const std::string& text)
    {
        for (size_t i = 0; i < text.size(); ) {
            std::string ch = utf8_char(text, i);
            if (token2idx_.find(ch) == token2idx_.end()) {
                token2idx_[ch] = idx2token_.size();
                idx2token_.push_back(ch);
            }
        }
    }
    
    std::vector<int64_t> encode(const std::string& text)
    {
        std::vector<int64_t> tokens;
        for (size_t i = 0; i < text.size(); ) {
            std::string ch = utf8_char(text, i);
            auto it = token2idx_.find(ch);
            tokens.push_back(it != token2idx_.end() ? it->second : 0);
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int64_t>& tokens)
    {
        std::string result;
        for (int64_t t : tokens) {
            if (t >= 0 && t < static_cast<int64_t>(idx2token_.size())) {
                result += idx2token_[t];
            }
        }
        return result;
    }
};


// ══════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════

int main()
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    // Параметры
    std::string train_path = "data/train.txt";
    int64_t d_model = 384;
    int64_t max_seq_len = 256;
    int64_t nhead = 6;
    int64_t num_blocks = 10;
    std::vector<int64_t> ffn_sizes = {1536, 1536};
    double dropout = 0.2;
    
    int64_t batch_size = 32;
    int64_t seq_len = 32;
    int epochs = 10;
    double lr = 1e-4;
    
    int64_t generate_max = 200;
    double temperature = 1.2;
    int64_t top_k = 20;
    
    // ══════════════════════════════════════════
    
    std::cout << "CUDA: " << (torch::cuda::is_available() ? "YES" : "NO") << std::endl;
    
    std::cout << "Loading " << train_path << "..." << std::endl;
    std::ifstream file(train_path, std::ios::binary);
    if (!file.is_open()) { std::cerr << "Cannot open!" << std::endl; return 1; }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    if (text.empty()) { std::cerr << "Empty!" << std::endl; return 1; }
    
    UTF8Tokenizer tokenizer;
    tokenizer.fit(text);
    auto data = tokenizer.encode(text);
    
    int64_t vocab_size = tokenizer.vocab_size();
    std::cout << "Bytes: " << text.size() << std::endl;
    std::cout << "Tokens: " << data.size() << std::endl;
    std::cout << "Vocab: " << vocab_size << std::endl;
    
    auto model = std::make_shared<GPTModel>(
        vocab_size, d_model, max_seq_len, nhead,
        num_blocks, ffn_sizes, dropout);
    
    size_t params = 0;
    for (const auto& p : model->parameters()) params += p.numel();
    std::cout << "Params: " << params / 1e6 << "M" << std::endl;
    
    auto device = model->device();
    auto opt = torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(lr).weight_decay(0.01));
    
    int64_t num_samples = static_cast<int64_t>(data.size()) - seq_len - 1;
    int64_t steps_per_epoch = std::min(500L, num_samples / batch_size);
    
    std::cout << "Training... (" << epochs << " epochs, " << steps_per_epoch << " steps)\n" << std::endl;
    
    model->train();
    
    for (int e = 0; e < epochs; ++e) {
        double total_loss = 0.0;
        
        for (int64_t step = 0; step < steps_per_epoch; ++step) {
            std::vector<int64_t> inputs_flat(batch_size * seq_len);
            std::vector<int64_t> targets_flat(batch_size * seq_len);
            
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t start = std::rand() % num_samples;
                for (int64_t s = 0; s < seq_len; ++s) {
                    inputs_flat[b * seq_len + s] = data[start + s];
                    targets_flat[b * seq_len + s] = data[start + s + 1];
                }
            }
            
            auto inputs = torch::from_blob(inputs_flat.data(), {batch_size, seq_len}, 
                torch::kLong).clone().to(device);
            auto targets = torch::from_blob(targets_flat.data(), {batch_size, seq_len}, 
                torch::kLong).clone().to(device);
            
            opt.zero_grad();
            auto output = model->forward(inputs);
            
            output = output.reshape({-1, vocab_size});
            targets = targets.reshape({-1});
            
            auto loss = torch::nn::functional::cross_entropy(
                output, targets,
                torch::nn::functional::CrossEntropyFuncOptions().ignore_index(0));
            
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            opt.step();
            
            total_loss += loss.item<double>();
        }
        
        double avg_loss = total_loss / steps_per_epoch;
        std::cout << "Epoch " << e + 1 << "/" << epochs
                  << " | Loss: " << avg_loss
                  << " | PPL: " << static_cast<int>(std::exp(avg_loss)) << std::endl;
    }
    
    torch::save(model, "gpt_model.pt");
    std::cout << "\nSaved!\n" << std::endl;
    
    // Интерактив
    model->eval();
    torch::NoGradGuard no_grad;
    
    std::cout << "=== Interactive mode ===" << std::endl;
    std::cout << ":q :t temp :k top_k :n max_tokens\n" << std::endl;
    std::cin.ignore();
    
    while (true) {
        std::cout << "> " << std::flush;
        std::string prompt;
        std::getline(std::cin, prompt);
        
        if (prompt.empty()) continue;
        if (prompt == ":q") break;
        if (prompt.rfind(":t ", 0) == 0) { temperature = std::stod(prompt.substr(3)); std::cout << "[t=" << temperature << "]\n" << std::endl; continue; }
        if (prompt.rfind(":k ", 0) == 0) { top_k = std::stoll(prompt.substr(3)); std::cout << "[k=" << top_k << "]\n" << std::endl; continue; }
        if (prompt.rfind(":n ", 0) == 0) { generate_max = std::stoll(prompt.substr(3)); std::cout << "[n=" << generate_max << "]\n" << std::endl; continue; }
        
        auto tokens = tokenizer.encode(prompt);
        if (tokens.empty()) continue;
        
        auto input = torch::from_blob(tokens.data(), {1, static_cast<int64_t>(tokens.size())},
            torch::kLong).clone().to(device);
        
        auto generated = model->generate(input, generate_max, temperature, top_k);
        auto gen_tokens = generated.index({0, torch::indexing::Slice()});
        
        std::vector<int64_t> new_tokens;
        for (int64_t i = tokens.size(); i < gen_tokens.size(0); ++i) {
            new_tokens.push_back(gen_tokens[i].item<int64_t>());
        }
        
        std::cout << "\n  " << prompt << tokenizer.decode(new_tokens) << "\n" << std::endl;
    }
    
    std::cout << "Bye!" << std::endl;
    return 0;
}

