
#include <string>


enum class logging
{
    CONSOLE,
    FILE,
    CONSOLE_AND_FILE
};


class Logger
{

private:
    enum class LogLevel
    {
        LOG,
        INFO,
        WARN,
        ERROR,
        CRITICAL,
        DEBUG
    };

    std::string path;
    logging logging_target;

    std::string get_prefix();
    std::string get_color(LogLevel level);
    void send_to_logger(std::string message, LogLevel level);
    void console_log(std::string message, LogLevel level);
    void file_log(std::string message);

public:
    Logger(const logging log_target);

    void log(std::string message);
    void info(std::string message);
    void warn(std::string message);
    void critical(std::string message);
    void error(std::string message);
    void debug(std::string message);

};

