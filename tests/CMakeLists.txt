find_package(GTest REQUIRED)

file(GLOB_RECURSE TEST_SRC
    CONFIGURE_DEPENDS
    *.cpp
)

add_executable(math_tests
    ${TEST_SRC}
)


target_link_libraries(math_tests
    PRIVATE
        math
        GTest::gtest_main
)

target_include_directories(math_tests
    PRIVATE
        include
)


include(GoogleTest)

gtest_discover_tests(math_tests)

