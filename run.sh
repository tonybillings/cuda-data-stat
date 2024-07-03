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

cmake --build build --target gui
check_status "Go build"

if [ ! -f gocds/config.go ]; then
    echo "config.go not found in gocds directory"
    exit 1
fi

cd build
sudo /bin/bash -c "export LD_LIBRARY_PATH=install/lib:$LD_LIBRARY_PATH; ./gui/gocds"
