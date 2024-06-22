#ifndef SERVICE_H
#define SERVICE_H

#include "cds/data_stats.h"

#include <vector>
#include <string>

bool isCsvFile(const std::string& filename);
bool processCsvFile(const std::string& filePath, std::vector<float>& data, DataStats& stats);
bool processInputFiles(const std::string& workingDir, DataStats& stats);
bool analyzeData(const std::string& workingDir, DataStats& stats);

extern "C" {
    bool ProcessInputFiles();
    bool AnalyzeData();
    void GetFieldAndRecordCount(int* recordCount, int* fieldCount);
    void GetStats(float* minimums, float* maximums, float* totals, float* means, float* stdDevs,
        float* deltaMinimums, float* deltaMaximums, float* deltaTotals, float* deltaMeans, float* deltaStdDevs);
}

#endif // SERVICE_H
