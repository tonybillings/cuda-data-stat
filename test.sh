#!/bin/bash

check_status() {
    if [ $? -ne 0 ]; then
        echo "$1 failed"
        exit 1
    fi
}

if [ -d "build" ]; then
    sudo rm -rf build
fi

cmake -S . -B build
check_status "CMake configuration"

cmake --build build
check_status "Build"

cmake --build build --target storage_test
check_status "Build for storage_test"

cmake --build build --target service_test
check_status "Build for service_test"

cd build/libcds
sudo ctest
check_status "Test"

cd ../../
cmake --build build --target gui
check_status "Go build"

echo "All steps completed successfully"
