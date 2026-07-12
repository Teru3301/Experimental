#!/bin/bash

./build.sh

cd ..
cd build

ctest --output-on-failure

