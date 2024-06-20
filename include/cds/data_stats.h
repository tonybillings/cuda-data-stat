#ifndef DATA_STATS_H
#define DATA_STATS_H

#include <vector>

struct DataStats {
    size_t fieldCount = 0;
    size_t recordCount = 0;
    std::vector<float> minimums;
    std::vector<float> maximums;
    std::vector<float> totals;
    std::vector<float> means;
    std::vector<float> stdDevs;
};

namespace stats {
    DataStats get();
    void set(const DataStats& dataStats);
}

#endif //DATA_STATS_H
