#!/bin/bash

if [ -d "build" ]; then
    sudo rm -rf build
fi

cmake -S . -B build

if [ $? -ne 0 ]; then
  echo "CMake configuration failed"
  exit 1
fi

cmake --build build

if [ $? -ne 0 ]; then
  echo "Build failed"
  exit 1
fi

cmake --build build --target storage_test
if [ $? -ne 0 ]; then
  echo "Build for storage_test failed"
  exit 1
fi


cmake --build build --target service_test
if [ $? -ne 0 ]; then
  echo "Build for service_test failed"
  exit 1
fi

cd build
sudo ctest

if [ $? -ne 0 ]; then
  echo "Test failed"
  exit 1
fi
