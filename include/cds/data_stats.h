#ifndef DATA_STATS_H
#define DATA_STATS_H

#include <vector>
#include <algorithm>
#include <cmath>

struct DataStats {
    size_t fieldCount = 0;
    size_t recordCount = 0;

    std::vector<float> minimums;
    std::vector<float> maximums;
    std::vector<float> totals;
    std::vector<float> means;
    std::vector<float> stdDevs;

    std::vector<float> deltaMinimums;
    std::vector<float> deltaMaximums;
    std::vector<float> deltaTotals;
    std::vector<float> deltaMeans;
    std::vector<float> deltaStdDevs;

    DataStats()= default;

    explicit DataStats(const size_t fields)
        : fieldCount(fields),
          minimums(fields), maximums(fields), totals(fields), means(fields), stdDevs(fields),
          deltaMinimums(fields), deltaMaximums(fields), deltaTotals(fields), deltaMeans(fields), deltaStdDevs(fields) {}

    bool operator==(const DataStats& other) const {
        auto fuzzyEqual = [](const float x, const float y) {
            return std::fabs(x - y) < 0.0005f;
        };

        return fieldCount == other.fieldCount &&
           recordCount == other.recordCount &&
           std::equal(minimums.begin(), minimums.end(), other.minimums.begin(), other.minimums.end(), fuzzyEqual) &&
           std::equal(maximums.begin(), maximums.end(), other.maximums.begin(), other.maximums.end(), fuzzyEqual) &&
           std::equal(totals.begin(), totals.end(), other.totals.begin(), other.totals.end(), fuzzyEqual) &&
           std::equal(means.begin(), means.end(), other.means.begin(), other.means.end(), fuzzyEqual) &&
           std::equal(stdDevs.begin(), stdDevs.end(), other.stdDevs.begin(), other.stdDevs.end(), fuzzyEqual) &&
           std::equal(deltaMinimums.begin(), deltaMinimums.end(), other.deltaMinimums.begin(), other.deltaMinimums.end(), fuzzyEqual) &&
           std::equal(deltaMaximums.begin(), deltaMaximums.end(), other.deltaMaximums.begin(), other.deltaMaximums.end(), fuzzyEqual) &&
           std::equal(deltaTotals.begin(), deltaTotals.end(), other.deltaTotals.begin(), other.deltaTotals.end(), fuzzyEqual) &&
           std::equal(deltaMeans.begin(), deltaMeans.end(), other.deltaMeans.begin(), other.deltaMeans.end(), fuzzyEqual) &&
           std::equal(deltaStdDevs.begin(), deltaStdDevs.end(), other.deltaStdDevs.begin(), other.deltaStdDevs.end(), fuzzyEqual);
    }

    bool operator!=(const DataStats& other) const {
        return !(*this == other);
    }
};

namespace stats {
    DataStats get();
    void set(const DataStats& dataStats);
    void reset();
}

#endif //DATA_STATS_H
