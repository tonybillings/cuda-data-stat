cmake_minimum_required(VERSION 3.28)

project(CudaDataStat)

add_subdirectory(libcds)

add_custom_target(go_mod_tidy
        COMMAND go mod tidy
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/gocds
        COMMENT "Running go mod tidy..."
        VERBATIM
)

add_custom_target(gui ALL
        COMMAND go build -o ${CMAKE_BINARY_DIR}/gui/gocds .
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/gocds
        COMMENT "Building GUI for Cuda Data Stat..."
        DEPENDS go_mod_tidy
        VERBATIM
)
