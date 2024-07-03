#ifndef CDS_H
#define CDS_H

#ifndef __cplusplus
#include <stdbool.h>
#include <stdlib.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

    void EnableVerboseMode(bool enabled);
    const char* GetLastError();

    bool InitStorage(const char* workingDir, size_t sizeMb);
    bool CloseStorage();

    bool ProcessInputFiles();
    bool AnalyzeData();
    void GetFieldAndRecordCount(int* recordCount, int* fieldCount);
    void GetStats(double* minimums, double* maximums, double* totals, double* means, double* stdDevs,
        double* deltaMinimums, double* deltaMaximums, double* deltaTotals, double* deltaMeans, double* deltaStdDevs);

#ifdef __cplusplus
}
#endif

#endif // CDS_H
