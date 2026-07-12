#!/bin/bash

cd ..
mkdir -p build
cd build

cmake .. -DCMAKE_CXX_COMPILER=acpp
cmake --build .

