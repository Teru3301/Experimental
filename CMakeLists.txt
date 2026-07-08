cmake_minimum_required(VERSION 4.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(app)

include_directories(
    include
)

add_executable(bin
    ./source/main.cpp
)

