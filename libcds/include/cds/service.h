#ifndef SERVICE_H
#define SERVICE_H

#include "cds/data_stats.h"

#include <vector>
#include <string>

bool isCsvFile(const std::string& filename);
bool processCsvFile(const std::string& filePath, std::vector<double>& data, DataStats& stats);
bool processInputFiles(const std::string& workingDir, DataStats& stats);
bool analyzeData(const std::string& workingDir, DataStats& stats);

#endif // SERVICE_H
