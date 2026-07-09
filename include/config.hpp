
#pragma once

#include <string>


enum class Device 
{
    CPU,
    GPU
};


struct Args 
{
    std::string BOT_TOKEN;
    Device text_sence;
};



Args parse_args(int argc, char* argv[]);

