#ifndef SERVICE_H
#define SERVICE_H

#include "cds/data_stats.h"

#include <vector>
#include <string>

bool isCsvFile(const std::string& filename);
bool processCsvFile(const std::string& filePath, std::vector<double>& data, DataStats& stats);
bool processInputFiles(const std::string& workingDir, DataStats& stats);
bool analyzeData(const std::string& workingDir, DataStats& stats);

extern "C" {
    bool ProcessInputFiles();
    bool AnalyzeData();
    void GetFieldAndRecordCount(int* recordCount, int* fieldCount);
    void GetStats(double* minimums, double* maximums, double* totals, double* means, double* stdDevs,
        double* deltaMinimums, double* deltaMaximums, double* deltaTotals, double* deltaMeans, double* deltaStdDevs);
}

#endif // SERVICE_H
