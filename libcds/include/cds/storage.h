#ifndef STORAGE_H
#define STORAGE_H

#include <string>

std::string getWorkingDirectory();
void setWorkingDirectory(const std::string& dir);
bool checkDirectory(const std::string& dir);
bool createWorkingDirectory(const std::string& workingDir);
bool deleteWorkingDirectory(const std::string& workingDir);
bool createRamDisk(const std::string& workingDir, size_t sizeMb);
bool deleteRamDisk(const std::string& workingDir);
bool createDirectoryStructure(const std::string& workingDir);
bool createSparseFile(const std::string& workingDir);
bool appendData(const std::string& filePath, const char* data, size_t dataSize);
bool mapData(const std::string& filePath, char*& data, size_t& dataSize);
bool unmapData(char* data, size_t fileSize);

extern "C" {
    bool InitStorage(const char* workingDir, size_t sizeMb);
    bool CloseStorage();
}

#endif // STORAGE_H
