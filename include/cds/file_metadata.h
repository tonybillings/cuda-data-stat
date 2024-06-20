#ifndef FILE_METADATA_H
#define FILE_METADATA_H

#include <string>
#include <utility>

struct FileMetadata {
    std::string filename;
    size_t fieldCount;
    size_t recordCount;

    FileMetadata(
        std::string name,
        const size_t numberOfFields, const size_t numberOfRecords)
        : filename(std::move(name)),
            fieldCount(numberOfFields), recordCount(numberOfRecords) {}
};

#endif //FILE_METADATA_H
