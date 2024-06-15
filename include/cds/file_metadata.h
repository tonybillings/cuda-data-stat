#ifndef FILE_METADATA_H
#define FILE_METADATA_H

#include <string>
#include <utility>

struct FileMetadata {
    std::string file_name;
    size_t field_count;
    size_t record_count;
    int index_start;
    int index_end;

    FileMetadata(
        std::string name,
        const size_t number_of_fields, const size_t number_of_records,
        const int start_index, const int end_index)
        : file_name(std::move(name)),
            field_count(number_of_fields), record_count(number_of_records),
            index_start(start_index), index_end(end_index) {}
};

#endif //FILE_METADATA_H
