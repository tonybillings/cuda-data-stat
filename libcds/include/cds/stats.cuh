#ifndef STATS_CUH
#define STATS_CUH

#include "data_stats.h"

bool calculateStats(const char* data, size_t dataSize, DataStats& stats);
bool calculateStats(const char* data, size_t dataSize, size_t threadsPerBlock, DataStats& stats);

#endif //STATS_CUH
