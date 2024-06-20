/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/storage.h"
#include "cds/debug.h"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <cassert>

/*******************************************************************************
 USINGS
*******************************************************************************/

using std::string;

/*******************************************************************************
 PARAMETERS
*******************************************************************************/

namespace {
    const string defaultWorkDir = "/tmp/.cds";
    constexpr size_t ramDiskSizeMb = 10;
}

/*******************************************************************************
 TESTS
*******************************************************************************/

void testCheckDirectory() {
    system(("mkdir -p " + defaultWorkDir).c_str());

    assert(checkDirectory(defaultWorkDir) && "Check directory failed");

    const bool verbose = getVerbose();
    setVerbose(false);
    assert(!checkDirectory(defaultWorkDir + "/nonexistent_dir") && "Nonexistent directory check failed");
    setVerbose(verbose);

    system(("rm -rf " + defaultWorkDir).c_str());
}

void testCreateWorkingDirectory() {
    system(("rm -rf " + defaultWorkDir).c_str());

    assert(createWorkingDirectory(defaultWorkDir) && "Create working directory failed");
    assert(checkDirectory(defaultWorkDir) && "Check created directory failed");

    system(("rm -rf " + defaultWorkDir).c_str());
}

void testDeleteWorkingDirectory() {
    system(("mkdir -p " + defaultWorkDir).c_str());

    assert(deleteWorkingDirectory(defaultWorkDir) && "Delete working directory failed");

    const bool verbose = getVerbose();
    setVerbose(false);
    assert(!checkDirectory(defaultWorkDir) && "Check deleted directory failed");
    setVerbose(verbose);
}

void testCreateRamDisk() {
    system(("mkdir -p " + defaultWorkDir).c_str());

    assert(createRamDisk(defaultWorkDir, ramDiskSizeMb) && "Create RAM disk failed");

    system(("umount " + defaultWorkDir).c_str());
    system(("rm -rf " + defaultWorkDir).c_str());

    const bool verbose = getVerbose();
    setVerbose(false);
    assert(!checkDirectory(defaultWorkDir) && "Check directory failed");
    setVerbose(verbose);
}

void testDeleteRamDisk() {
    system(("mkdir -p " + defaultWorkDir).c_str());
    system(("mount -t ramfs -o size=10m ramfs " + defaultWorkDir).c_str());

    assert(deleteRamDisk(defaultWorkDir) && "Delete RAM disk failed");

    system(("rm -rf " + defaultWorkDir).c_str());

    const bool verbose = getVerbose();
    setVerbose(false);
    assert(!checkDirectory(defaultWorkDir) && "Check directory failed");
    setVerbose(verbose);
}

void testCreateDirectoryStructure() {
    system(("mkdir -p " + defaultWorkDir).c_str());

    assert(createDirectoryStructure(defaultWorkDir) && "Create directory structure failed");
    assert(checkDirectory(defaultWorkDir + "/input") && "Check input directory failed");

    system(("rm -rf " + defaultWorkDir).c_str());
}

void testCreateSparseFile() {
    system(("mkdir -p " + defaultWorkDir).c_str());

    assert(createSparseFile(defaultWorkDir) && "Create sparse file failed");

    struct stat st{};
    assert(stat((defaultWorkDir + "/data").c_str(), &st) == 0 && "Stat data file failed");

    system(("rm -rf " + defaultWorkDir).c_str());
}

void testInitStorage() {
    system(("rm -rf " + defaultWorkDir).c_str());

    assert(InitStorage(defaultWorkDir.c_str(), ramDiskSizeMb) && "Init storage failed");
    assert(checkDirectory(defaultWorkDir) && "Check init directory failed");
    assert(checkDirectory(defaultWorkDir + "/input") && "Check input directory failed");

    struct stat st{};
    assert(stat((defaultWorkDir + "/data").c_str(), &st) == 0 && "Stat data file failed");

    system(("umount " + defaultWorkDir).c_str());
    system(("rm -rf " + defaultWorkDir).c_str());
}

void testCloseStorage() {
    system(("mkdir -p " + defaultWorkDir).c_str());
    system(("mount -t ramfs -o size=10m ramfs " + defaultWorkDir).c_str());

    assert(CloseStorage() && "Close storage failed");

    const bool verbose = getVerbose();
    setVerbose(false);
    assert(!checkDirectory(defaultWorkDir) && "Check directory failed");
    setVerbose(verbose);
}

void testAppendData() {
    const auto dataFilePath = defaultWorkDir + "/data";

    assert(InitStorage(defaultWorkDir.c_str(), ramDiskSizeMb) && "Init storage failed");

    constexpr size_t numFloats = 100;
    const auto floatArray = new float[numFloats];
    for (int i = 0; i < numFloats; i++) {
        floatArray[i] = static_cast<float>(i);
    }
    const auto data = reinterpret_cast<char*>(floatArray);
    constexpr size_t dataSize = numFloats * sizeof(float);

    assert(appendData(dataFilePath, data, dataSize) && "Append data failed");

    FILE* file = fopen(dataFilePath.c_str(), "rb");
    assert(file != nullptr && "Open data file failed");

    const auto readBuffer = new float[numFloats];
    const size_t bytesRead = fread(readBuffer, sizeof(float), numFloats, file);
    assert(bytesRead == numFloats && "Read data file failed");

    for (int i = 0; i < numFloats; i++) {
        assert(readBuffer[i] == static_cast<float>(i) && "Data verification failed");
    }

    fclose(file);
    delete[] floatArray;
    delete[] readBuffer;

    assert(CloseStorage() && "Close storage failed");
}

void testMapAndUnmapData() {
    const auto dataFilePath = defaultWorkDir + "/data";

    assert(InitStorage(defaultWorkDir.c_str(), ramDiskSizeMb) && "Init storage failed");

    constexpr size_t numFloats = 100;
    const auto floatArray = new float[numFloats];
    for (size_t i = 0; i < numFloats; ++i) {
        floatArray[i] = static_cast<float>(i);
    }
    const char* data = reinterpret_cast<char*>(floatArray);
    constexpr size_t dataSize = numFloats * sizeof(float);

    assert(appendData(dataFilePath, data, dataSize) && "Append data for map data failed");

    char* mappedData = nullptr;
    size_t fileSize = 0;
    assert(mapData(dataFilePath, mappedData, fileSize) && "Map data failed");
    assert(mappedData != nullptr && "Mapped data is null");
    assert(fileSize == dataSize && "File size mismatch after mapping data");

    const float* mappedFloatArray = reinterpret_cast<float*>(mappedData);
    for (size_t i = 0; i < numFloats; ++i) {
        assert(mappedFloatArray[i] == static_cast<float>(i) && "Mapped data verification failed");
    }

    assert(unmapData(mappedData, fileSize) && "Unmap data failed");

    delete[] floatArray;
    assert(CloseStorage() && "Close storage failed");
}

int main() {
    setVerbose(true);

    testCheckDirectory();
    testCreateWorkingDirectory();
    testDeleteWorkingDirectory();
    testCreateRamDisk();
    testDeleteRamDisk();
    testCreateDirectoryStructure();
    testCreateSparseFile();
    testInitStorage();
    testCloseStorage();
    testAppendData();
    testMapAndUnmapData();

    printf("All tests completed successfully.\n");
    exit(EXIT_SUCCESS);
}
