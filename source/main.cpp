
#include "config.hpp"
#include "logger.hpp"

#include <tgbot/tgbot.h>


int main(int argc, char* argv[])
{
    Args args = parse_args(argc, argv);
    Logger lg(logging::CONSOLE);

    TgBot::Bot bot(args.BOT_TOKEN);

    bot.getEvents().onCommand("start", [&bot](TgBot::Message::Ptr message) {
        bot.getApi().sendMessage(message->chat->id, "Hi!");
    });
    bot.getEvents().onAnyMessage([&](TgBot::Message::Ptr message) {
        lg.log(std::string("User wrote ") + message->text);
        if (StringTools::startsWith(message->text, "/start")) {
            return;
        }
        bot.getApi().sendMessage(message->chat->id, "Your message is: " + message->text);
    });
    try {
        lg.log(std::string("Bot username: ") + bot.getApi().getMe()->username);
        TgBot::TgLongPoll longPoll(bot);
        while (true) {
            lg.info("Long poll started");
            longPoll.start();
        }
    } catch (TgBot::TgException& e) {
        lg.error(std::string("error: ") + e.what());
    }
    return 0;
}

