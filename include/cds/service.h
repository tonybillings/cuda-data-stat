#ifndef SERVICE_H
#define SERVICE_H

#include "cds/file_metadata.h"
#include "cds/data_stats.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

bool process_files(const std::string& mount_point, std::vector<char>& data, std::vector<char>& mask,
                   std::unordered_map<std::string, std::unique_ptr<FileMetadata>>& index);

bool analyze_data(const std::vector<char>& data, size_t field_count, DataStats& stats);

bool calculate_stats(const std::vector<char>& data, size_t field_count, size_t record_count, DataStats& stats);

#endif // SERVICE_H
