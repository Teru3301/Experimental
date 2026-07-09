
#pragma once

#include "config.hpp"
#include "math.hpp"
#include <cstdint>
#include <vector>


namespace perception
{

struct input 
{
    std::string text;
    std::string source;
    uint64_t now_time;
};


class text
{
private:
    device device = device::CPU;

    class attention
    {
    private:
        Matrix WQ;
        Matrix WK;
        Matrix WV;
        Matrix WO;

    public:
        Matrix forward(const Matrix& embeding);

    };

    Matrix FFN1;
    Matrix FFn2;

    std::vector<uint32_t> tokenize(input input);
    Matrix embeding(const std::vector<uint32_t>& token_seq);
    Matrix add_pos_embeding(const Matrix& embeding);
    Matrix feedforward(const Matrix& meaning);

public:
    Matrix encode(const input& input);

};

}

