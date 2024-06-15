#ifndef STORAGE_H
#define STORAGE_H

#include <string>

bool create_ramdisk(const std::string& mount_point, size_t size_mb);
bool create_directory_structure(const std::string& mount_point);
void cleanup(const std::string& mount_point);

#endif // STORAGE_H
