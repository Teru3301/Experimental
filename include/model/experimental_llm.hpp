
#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>

#include "model/config.hpp"
#include "model/modules/embedding.hpp"
#include "model/modules/transformer_block.hpp"
#include "model/modules/custom_block.hpp"
#include "model/modules/mlp.hpp"
#include "model/modules/layer_norm.hpp"

class ExperimentalLLMImpl : public torch::nn::Module
{
public:
    explicit ExperimentalLLMImpl(ModelConfig config);

    torch::Tensor forward(torch::Tensor input_ids);

private:
    ModelConfig cfg_;

    Embedding embedding_{nullptr};
    std::vector<TransformerBlock> first_blocks_;
    MLP shared_mlp_{nullptr};
    std::vector<CustomBlock> custom_blocks_;
    std::vector<TransformerBlock> last_blocks_;
    LayerNorm final_norm_{nullptr};
    torch::nn::Linear head_{nullptr};
};

TORCH_MODULE(ExperimentalLLM);

