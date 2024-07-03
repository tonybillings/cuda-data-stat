#ifndef DATA_STATS_H
#define DATA_STATS_H

#include <vector>
#include <algorithm>
#include <cmath>

struct DataStats {
    size_t fieldCount = 0;
    size_t recordCount = 0;

    std::vector<double> minimums;
    std::vector<double> maximums;
    std::vector<double> totals;
    std::vector<double> means;
    std::vector<double> stdDevs;

    std::vector<double> deltaMinimums;
    std::vector<double> deltaMaximums;
    std::vector<double> deltaTotals;
    std::vector<double> deltaMeans;
    std::vector<double> deltaStdDevs;

    DataStats()= default;

    explicit DataStats(const size_t fields)
        : fieldCount(fields),
          minimums(fields), maximums(fields), totals(fields), means(fields), stdDevs(fields),
          deltaMinimums(fields), deltaMaximums(fields), deltaTotals(fields), deltaMeans(fields), deltaStdDevs(fields) {}

    bool operator==(const DataStats& other) const {
        auto fuzzyEqual = [](const double x, const double y) {
            return std::fabs(x - y) < 0.000001;
        };

        auto veryFuzzyEqual = [](const double x, const double y) {
            return std::fabs(x - y) < 0.00005;
        };

        return fieldCount == other.fieldCount &&
            recordCount == other.recordCount &&
            std::equal(minimums.begin(), minimums.end(), other.minimums.begin(), fuzzyEqual) &&
            std::equal(maximums.begin(), maximums.end(), other.maximums.begin(), fuzzyEqual) &&
            std::equal(totals.begin(), totals.end(), other.totals.begin(), veryFuzzyEqual) &&
            std::equal(means.begin(), means.end(), other.means.begin(), fuzzyEqual) &&
            std::equal(stdDevs.begin(), stdDevs.end(), other.stdDevs.begin(), fuzzyEqual) &&
            std::equal(deltaMinimums.begin(), deltaMinimums.end(), other.deltaMinimums.begin(), fuzzyEqual) &&
            std::equal(deltaMaximums.begin(), deltaMaximums.end(), other.deltaMaximums.begin(), fuzzyEqual) &&
            std::equal(deltaTotals.begin(), deltaTotals.end(), other.deltaTotals.begin(), veryFuzzyEqual) &&
            std::equal(deltaMeans.begin(), deltaMeans.end(), other.deltaMeans.begin(), fuzzyEqual) &&
            std::equal(deltaStdDevs.begin(), deltaStdDevs.end(), other.deltaStdDevs.begin(), fuzzyEqual);
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
