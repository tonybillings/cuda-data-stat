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

if [ ! -f gocds/cds/config.go ]; then
    echo "config.go not found in gocds/cds directory"
    exit 1
fi

sudo /bin/bash -c "export LD_LIBRARY_PATH=build/install/lib:$LD_LIBRARY_PATH; build/gui/gocds"
