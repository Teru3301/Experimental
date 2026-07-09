cd ..
mkdir build
cd build
cmake ..
cmake --build .
ctest --output-on-failure
