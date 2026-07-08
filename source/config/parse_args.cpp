
#include "config.hpp"
#include "logger.hpp"

Args parse_args(int argc, char* argv[])
{
    Args args;
    Logger lg(logging::CONSOLE);

    lg.debug(std::to_string(argc));

    if ((argc - 1) % 2 != 0)
    {
        lg.warn("Incorrect number of arguments. The received parameters will be ignored.");
    }
    else 
    {
        for (int i = 1; i < argc; i++)
        {
            std::string_view arg = argv[i];
            if (arg == "-t" || arg == "--token") {
                i++;
                args.BOT_TOKEN = argv[i];
            }
            else 
            {
                lg.error(std::string("Invalid parameter '") + argv[i] + "', all subsequent parameters will be ignored.");
                break;
            }
        }
    }

    if (args.BOT_TOKEN.empty())
    {
        const char* env_token = std::getenv("BOT_TOKEN");
        if (env_token != nullptr) {
            args.BOT_TOKEN = env_token;
            lg.debug("Token loaded from environment variable BOT_TOKEN");
        }
        else {
            lg.critical("Unable to obtain the Telegram bot token.");
            exit(1);
        }
    }

    return args;
}

