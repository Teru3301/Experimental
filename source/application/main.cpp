
#include "config.hpp"
#include "logger.hpp"


int main(int argc, char* argv[])
{
    Logger lg(logging::CONSOLE);
    lg.debug("The program was launched.");

    return 0;
}

