
#pragma once

#include "config.hpp"
#include "math.hpp"
#include <cstdint>
#include <vector>


namespace perception
{

struct TextInput 
{
    std::string text;
    std::string source;
    uint64_t now_time;
};


class AttentionHead
{
private:
    Matrix WQ;
    Matrix WK;
    Matrix WV;

public:
    Tensor forward(const Tensor& embedding);

};


class MultiHeadAttention
{
private:
    std::vector<AttentionHead> heads;
    Matrix WO;

public:
    Tensor forward(const Tensor& embedding);

};


class FeedForward
{
private:
    Matrix W1;
    Matrix W2;

    Vector b1;
    Vector b2;

public:
    Tensor forward(const Tensor& embedding);
};


class LayerNorm
{
private:
    Vector gamma;
    Vector beta;
    float epsilon = 1e-5f;

public:
    Tensor forward(const Tensor& x);
};


class EncoderBlock
{
private:
    MultiHeadAttention attention;
    FeedForward feedForward;
    LayerNorm norm1;
    LayerNorm norm2;

public:
    Tensor forward(const Tensor& embedding);

};


class Text
{
private:
    device executionDevice = device::CPU;
    std::vector<EncoderBlock> blocks;

    std::vector<uint32_t> tokenize(const TextInput& input);
    Tensor embedding(const std::vector<uint32_t>& token_seq);
    Tensor add_pos_embedding(const Tensor& embedding);

public:
    Tensor encode(const TextInput& input);

};

}

