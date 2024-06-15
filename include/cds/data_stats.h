#ifndef DATA_STATS_H
#define DATA_STATS_H

#include <vector>

struct DataStats {
    std::vector<float> totals;
    std::vector<float> means;
    std::vector<float> std_devs;
};

#endif //DATA_STATS_H
