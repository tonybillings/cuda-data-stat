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

cmake --build build --target debug_test
check_status "Build for debug_test"

cmake --build build --target datastats_test
check_status "Build for datastats_test"

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

if [ ! -f gocds/cds/config.go ]; then
    echo "config.go not found in gocds/cds directory"
    exit 1
fi

cd gocds/_test
sudo go get github.com/stretchr/testify/assert
for file in *_test.go; do
    if [ -f "$file" ]; then
        sudo /bin/bash -c "export LD_LIBRARY_PATH=../../build/install/lib:$LD_LIBRARY_PATH; go test -p 1 -v $file"
        check_status "Go test"
    fi
done

echo "All steps completed successfully"
