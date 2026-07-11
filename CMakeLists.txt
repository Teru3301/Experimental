cmake_minimum_required(VERSION 4.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

find_package(PkgConfig REQUIRED)

project(app)

find_package(IntelSYCL REQUIRED)


# math

file(GLOB_RECURSE MATH_SRC
    CONFIGURE_DEPENDS
    source/math/*.cpp
)

add_library(math 
    STATIC
    ${MATH_SRC}
)

target_include_directories(math 
    PUBLIC
    include
    ${SYCL_INCLUDE_DIR}
)

target_compile_definitions(math 
    PUBLIC 
    USE_SYCL
)

target_link_libraries(math 
    PUBLIC
    ${SYCL_LIBRARY}
)

# model

file(GLOB_RECURSE MODEL_SRC 
    CONFIGURE_DEPENDS
    source/model/*.cpp
)

add_library(model 
    STATIC 
    ${MODEL_SRC}
)

target_include_directories(model 
    PUBLIC
    math
    include
)


# app

file(GLOB_RECURSE APP_SRC
    CONFIGURE_DEPENDS
    source/application/*.cpp
    source/config/*.cpp
    source/logger/*.cpp
)

add_executable(bin ${APP_SRC})


include_directories(bin 
    PRIVATE
    include
    libs
)

target_link_libraries(bin
    model
    TgBot
    pthread
    ssl
    crypto
    curl
    z
)


enable_testing()
add_subdirectory(tests)

