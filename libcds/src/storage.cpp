/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/storage.h"
#include "cds/debug.h"

#include <iostream>
#include <vector>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <mutex>

/*******************************************************************************
 USINGS
*******************************************************************************/

using std::string;
using std::to_string;
using std::vector;
using std::mutex;
using std::lock_guard;

/*******************************************************************************
 STATE
*******************************************************************************/

namespace {
    string workingDirectory;
    mutex workingDirectoryMutex;
}

/*******************************************************************************
 INTERNAL FUNCTIONS
*******************************************************************************/

string getWorkingDirectory() {
    lock_guard lock(workingDirectoryMutex);
    return workingDirectory;
}

void setWorkingDirectory(const string& dir) {
    lock_guard lock(workingDirectoryMutex);
    workingDirectory = dir;
}

bool checkDirectory(const string& dir) {
    struct stat st{};
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        ERROR("'%s' does not exist or is not a directory", dir.c_str());
        return false;
    }
    return true;
}

bool createWorkingDirectory(const string& workingDir) {
    PRINTLN("Creating working directory: %s", workingDir.c_str());
    if (mkdir(workingDir.c_str(), 0755) != 0 && errno != EEXIST) {
        ERROR("unable to create working directory '%s': %s", workingDir.c_str(), strerror(errno));
        return false;
    }
    return true;
}

bool deleteWorkingDirectory(const string& workingDir) {
    if (!checkDirectory(workingDir)) {
        return false;
    }

    PRINTLN("Deleting the working directory...");
    if (rmdir(workingDir.c_str()) != 0) {
        ERROR("unable to delete the working directory '%s': %s", workingDir.c_str(), strerror(errno));
        return false;
    }

    PRINTLN("Successfully deleted the working directory.");
    return true;
}

bool createRamDisk(const string& workingDir, const size_t sizeMb) {
    if (!checkDirectory(workingDir)) {
        return false;
    }

    PRINTLN("Creating RAM disk: %s", workingDir.c_str());
    const string mountOptions = "size=" + to_string(sizeMb) + "m";
    if (mount("ramfs", workingDir.c_str(), "ramfs", 0, mountOptions.c_str()) != 0) {
        ERROR("unable to mount ramfs to '%s': %s", workingDir.c_str(), strerror(errno));
        return false;
    }

    PRINTLN("RAM disk created/mounted successfully.");
    return true;
}

bool deleteRamDisk(const string& workingDir) {
    if (!checkDirectory(workingDir)) {
        return false;
    }

    sync(); // ensure pending actions are completed / device is not busy

    PRINTLN("Deleting the RAM disk...");
    if (umount(workingDir.c_str()) != 0) {
        ERROR("unable to delete the RAM disk '%s': %s", workingDir.c_str(), strerror(errno));
        return false;
    }

    PRINTLN("Successfully deleted the RAM disk.");
    return true;
}

bool createDirectoryStructure(const string& workingDir) {
    if (!checkDirectory(workingDir)) {
        return false;
    }

    PRINTLN("Creating directory structure...");
    vector<string> dirs = { "/input" };
    for (const auto& dir : dirs) {
        string path = workingDir + dir;
        if (mkdir(path.c_str(), 0755) != 0) {
            ERROR("unable to create directory '%s': %s", path.c_str(), strerror(errno));
            return false;
        }
    }

    PRINTLN("Directory structure created successfully.");
    return true;
}

bool createSparseFile(const string& workingDir) {
    if (!checkDirectory(workingDir)) {
        return false;
    }

    const string filePath = workingDir + "/data";
    PRINTLN("Creating data file: %s", filePath.c_str());
    const int fd = open(filePath.c_str(), O_RDWR | O_CREAT, static_cast<mode_t>(0666));
    if (fd == -1) {
        ERROR("unable to create the data file '%s': %s", filePath.c_str(), strerror(errno));
        return false;
    }

    close(fd);
    return true;
}

bool appendData(const string& filePath, const char* data, const size_t dataSize) {
    const int fd = open(filePath.c_str(), O_WRONLY | O_APPEND);
    if (fd == -1) {
        ERROR("unable to open the data file '%s': %s", filePath.c_str(), strerror(errno));
        return false;
    }

    if (const ssize_t written = write(fd, data, dataSize); written == -1) {
        ERROR("unable to write to the data file '%s': %s", filePath.c_str(), strerror(errno));
        close(fd);
        return false;
    } else {
        if (static_cast<size_t>(written) != dataSize) {
            ERROR("unexpected number of bytes written: expected %lu, got %lu", dataSize, written);
            close(fd);
            return false;
        }
    }

    if (close(fd) == -1) {
        ERROR("unexpected issue closing data file '%s': %s", filePath.c_str(), strerror(errno));
        return false;
    }

    return true;
}

bool mapData(const string& filePath, char*& data, size_t& dataSize) {
    const int fd = open(filePath.c_str(), O_RDONLY);
    if (fd == -1) {
        ERROR("unable to open data file: %s", strerror(errno));
        return false;
    }

    struct stat fileInfo{};
    if (fstat(fd, &fileInfo) == -1) {
        ERROR("unable to get file size: %s", strerror(errno));
        close(fd);
        return false;
    }
    dataSize = fileInfo.st_size;

    if (dataSize == 0) {
        close(fd);
        return false;
    }

    data = static_cast<char*>(mmap(nullptr, dataSize, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED) {
        ERROR("unable to map file: %s", strerror(errno));
        close(fd);
        return false;
    }

    close(fd);
    return true;
}

bool unmapData(char* data, const size_t fileSize) {
    if (munmap(data, fileSize) == -1) {
        ERROR("unable to unmap file: %s", strerror(errno));
        return false;
    }
    return true;
}

/*******************************************************************************
 PUBLIC INTERFACE
*******************************************************************************/

extern "C" {
    bool InitStorage(const char* workingDir, const size_t sizeMb) {
        setWorkingDirectory(workingDir);

        if (!createWorkingDirectory(workingDir)) {
            return false;
        }

        if (!createRamDisk(workingDir, sizeMb)) {
            return false;
        }

        if (!createDirectoryStructure(workingDir)) {
            return false;
        }

        if (!createSparseFile(workingDir)) {
            return false;
        }

        return true;
    }

    bool CloseStorage() {
        const string workingDir = getWorkingDirectory();
        if (!checkDirectory(workingDir)) {
            return false;
        }

        if (!deleteRamDisk(workingDir)) {
            return false;
        }

        if (!deleteWorkingDirectory(workingDir)) {
            return false;
        }

        return true;
    }
}
