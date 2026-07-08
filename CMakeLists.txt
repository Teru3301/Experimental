cmake_minimum_required(VERSION 4.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)

project(app)

file(GLOB_RECURSE APP_SRC "source/*.cpp")
add_executable(bin ${APP_SRC})

include_directories(bin PRIVATE
    include
    libs
)

target_link_libraries(bin
    TgBot
    pthread
    ssl
    crypto
    curl
    z
)

