cmake_minimum_required(VERSION 3.28)

project(CudaDataStat)

add_subdirectory(libcds)

find_path(CXX_INCLUDE_DIR NAMES string
        PATHS
        /usr/include/c++/13
        /usr/include/x86_64-linux-gnu/c++/13
        /usr/local/include/c++/13
        /usr/include/c++/12
        /usr/include/x86_64-linux-gnu/c++/12
        /usr/local/include/c++/12
        /usr/include/c++/11
        /usr/include/x86_64-linux-gnu/c++/11
        /usr/local/include/c++/11
        NO_DEFAULT_PATH
)

find_path(GCC_INCLUDE_DIR NAMES bits/c++config.h
        PATHS
        /usr/include/x86_64-linux-gnu/c++/13
        /usr/local/include/x86_64-linux-gnu/c++/13
        /usr/include/x86_64-linux-gnu/c++/12
        /usr/local/include/x86_64-linux-gnu/c++/12
        /usr/include/x86_64-linux-gnu/c++/11
        /usr/local/include/x86_64-linux-gnu/c++/11
        NO_DEFAULT_PATH
)

configure_file(
        ${CMAKE_SOURCE_DIR}/gocds/cds/config_template.go.in
        ${CMAKE_SOURCE_DIR}/gocds/cds/config.go
        @ONLY
)

set(ENV{CGO_CFLAGS} "-I${CMAKE_BINARY_DIR}/install/include -I${CXX_INCLUDE_DIR} -I${GCC_INCLUDE_DIR}")
set(ENV{CGO_LDFLAGS} "-L${CMAKE_BINARY_DIR}/install/lib")
set(ENV{LD_LIBRARY_PATH} "${CMAKE_BINARY_DIR}/install/lib:$ENV{LD_LIBRARY_PATH}")

add_custom_target(install_headers_and_lib ALL
        DEPENDS cds
        COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --prefix ${CMAKE_BINARY_DIR}/install
        COMMENT "Installing headers and libraries..."
        VERBATIM
)

add_custom_target(go_mod_tidy
        COMMAND go mod tidy
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/gocds
        COMMENT "Running go mod tidy..."
        VERBATIM
)

add_custom_target(gui ALL
        COMMAND ${CMAKE_COMMAND} -E env "CGO_CFLAGS=$ENV{CGO_CFLAGS}" "CGO_LDFLAGS=$ENV{CGO_LDFLAGS}" go build -o ${CMAKE_BINARY_DIR}/gui/gocds .
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/gocds
        COMMENT "Building GUI for Cuda Data Stat..."
        DEPENDS install_headers_and_lib go_mod_tidy
        VERBATIM
)
