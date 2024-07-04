/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/data_stats.h"
#include "cds/debug.h"

#include <cassert>

/*******************************************************************************
 TESTS
*******************************************************************************/

void testDefaultConstructor() {
    PRINTLN("Running testDefaultConstructor()...");

    const DataStats ds{};
    assert(ds.fieldCount == 0);
    assert(ds.recordCount == 0);
    assert(ds.minimums.empty());
    assert(ds.maximums.empty());
    assert(ds.totals.empty());
    assert(ds.means.empty());
    assert(ds.stdDevs.empty());
    assert(ds.deltaMinimums.empty());
    assert(ds.deltaMaximums.empty());
    assert(ds.deltaTotals.empty());
    assert(ds.deltaMeans.empty());
    assert(ds.deltaStdDevs.empty());
}

void testParameterizedConstructor() {
    PRINTLN("Running testParameterizedConstructor()...");

    constexpr size_t fieldCount = 5;
    const DataStats ds(fieldCount);

    assert(ds.fieldCount == fieldCount);
    assert(ds.recordCount == 0);
    assert(ds.minimums.size() == fieldCount);
    assert(ds.maximums.size() == fieldCount);
    assert(ds.totals.size() == fieldCount);
    assert(ds.means.size() == fieldCount);
    assert(ds.stdDevs.size() == fieldCount);
    assert(ds.deltaMinimums.size() == fieldCount);
    assert(ds.deltaMaximums.size() == fieldCount);
    assert(ds.deltaTotals.size() == fieldCount);
    assert(ds.deltaMeans.size() == fieldCount);
    assert(ds.deltaStdDevs.size() == fieldCount);
}

void testEqualityOperator() {
    PRINTLN("Running testEqualityOperator()...");

    constexpr size_t fieldCount = 3;
    DataStats stats1(fieldCount);
    DataStats stats2(fieldCount);
    assert(stats1 == stats2);

    stats1.recordCount = 1;
    assert(stats1 != stats2);

    stats2.recordCount = 1;
    assert(stats1 == stats2);

    stats1.minimums[0] = 1.0;
    assert(stats1 != stats2);

    stats2.minimums[0] = 1.0;
    assert(stats1 == stats2);

    stats1.stdDevs[0] = 0.123456789;
    assert(stats1 != stats2);

    stats2.stdDevs[0] = 0.123456789;
    assert(stats1 == stats2);
}

void testSetAndGet() {
    PRINTLN("Running testSetAndGet()...");

    constexpr size_t fieldCount = 4;
    DataStats stats(fieldCount);
    stats.recordCount = 100;
    for (size_t i = 0; i < fieldCount; ++i) {
        const auto scalar = static_cast<double>(i);
        stats.minimums[i] = scalar * 1.0;
        stats.maximums[i] = scalar * 2.0;
        stats.totals[i] = scalar * 3.0;
        stats.means[i] = scalar * 4.0;
        stats.stdDevs[i] = scalar * 5.0;
        stats.deltaMinimums[i] = scalar * 6.0;
        stats.deltaMaximums[i] = scalar * 7.0;
        stats.deltaTotals[i] = scalar * 8.0;
        stats.deltaMeans[i] = scalar * 9.0;
        stats.deltaStdDevs[i] = scalar * 10.0;
    }

    stats::set(stats);
    const DataStats retrievedStats = stats::get();
    assert(retrievedStats == stats);
}

void testReset() {
    PRINTLN("Running testReset()...");

    constexpr size_t fieldCount = 2;
    DataStats stats(fieldCount);
    stats.recordCount = 10;
    stats.minimums[0] = 1.0;
    stats.maximums[1] = 2.0;
    stats.totals[0] = 3.0;
    stats.means[1] = 4.0;
    stats.stdDevs[0] = 5.0;
    stats.deltaMinimums[0] = 6.0;
    stats.deltaMaximums[1] = 7.0;
    stats.deltaTotals[0] = 8.0;
    stats.deltaMeans[1] = 9.0;
    stats.deltaStdDevs[0] = 10.0;
    stats::set(stats);

    stats::reset();
    const DataStats resetStats = stats::get();
    assert(resetStats.fieldCount == 0);
    assert(resetStats.recordCount == 0);
    assert(resetStats.minimums.empty());
    assert(resetStats.maximums.empty());
    assert(resetStats.totals.empty());
    assert(resetStats.means.empty());
    assert(resetStats.stdDevs.empty());
    assert(resetStats.deltaMinimums.empty());
    assert(resetStats.deltaMaximums.empty());
    assert(resetStats.deltaTotals.empty());
    assert(resetStats.deltaMeans.empty());
    assert(resetStats.deltaStdDevs.empty());
}

/*******************************************************************************
 MAIN
*******************************************************************************/

int main() {
    testDefaultConstructor();
    testParameterizedConstructor();
    testEqualityOperator();
    testSetAndGet();
    testReset();

    printf("All tests completed successfully.\n");
    exit(EXIT_SUCCESS);
}
