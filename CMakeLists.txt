cmake_minimum_required(VERSION 4.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(app)

include(FetchContent)


# libtorch

set(CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)


# sentencepiece

FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.1
)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sentencepiece)


# app

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
    "${sentencepiece_SOURCE_DIR}/src"
    include 
)
target_link_libraries(bin
    PRIVATE
    "${TORCH_LIBRARIES}"
    sentencepiece
)

add_custom_command(TARGET bin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
)

