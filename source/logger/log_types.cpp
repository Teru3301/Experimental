
#include "logger.hpp"


void Logger::send_to_logger(std::string message, LogLevel level)
{
    switch (this->logging_target)
    {
        case logging::CONSOLE:
            console_log(message, level);
            break;
        case logging::FILE:
            break;
        case logging::CONSOLE_AND_FILE:
            break;
    }
}


void Logger::log(std::string message)
{
    message = "    " + get_prefix() + " " + message;
    send_to_logger(message, LogLevel::LOG);
}

void Logger::info(std::string message)
{
    message = "[+] " + get_prefix() + " " + message;
    send_to_logger(message, LogLevel::INFO);
}

void Logger::warn(std::string message)
{
    message = "[!] " + get_prefix() + " " + message;
    send_to_logger(message, LogLevel::WARN);
}

void Logger::error(std::string message)
{
    message = "[x] " + get_prefix() + " " + message;
    send_to_logger(message, LogLevel::ERROR);
}

void Logger::critical(std::string message)
{
    message = "[#] " + get_prefix() + " " + message;
    send_to_logger(message, LogLevel::CRITICAL);
}

void Logger::debug(std::string message)
{
    message = "[*] " + get_prefix() + " " + message;
    send_to_logger(message, LogLevel::DEBUG);
}

