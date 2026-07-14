cmake_minimum_required(VERSION 4.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(app)


# app

set(CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)

find_package(PkgConfig REQUIRED)
file(GLOB_RECURSE APP_SRC
    CONFIGURE_DEPENDS
    source/logger/*.cpp
    source/model/*.cpp
    source/app/*.cpp
)
add_executable(bin ${APP_SRC})
target_include_directories(bin 
    PRIVATE 
    "${TORCH_INCLUDE_DIRS}"
    include 
    libs
)
target_link_libraries(bin
    PRIVATE
    "${TORCH_LIBRARIES}"
)

add_custom_command(TARGET bin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
)

