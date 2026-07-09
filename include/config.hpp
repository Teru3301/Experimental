
#pragma once

#include <string>


enum class device 
{
    CPU,
    GPU
};


struct Args 
{
    std::string BOT_TOKEN;
    device text_sence;
};



Args parse_args(int argc, char* argv[]);

