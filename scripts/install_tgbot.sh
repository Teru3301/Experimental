#!/bin/bash

git clone https://github.com/reo7sp/tgbot-cpp

cd tgbot-cpp

cmake .
make -j4
sudo make install

cd ..
rm -rf tgbot-cpp

