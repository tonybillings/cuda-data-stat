/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/storage.h"
#include "cds/service.h"
#include "cds/debug.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <cstdio>
#include <cmath>

/*******************************************************************************
 USINGS
*******************************************************************************/

using std::string;
using std::ofstream;
using std::vector;
using std::to_string;
using std::fabs;

/*******************************************************************************
 PARAMETERS
*******************************************************************************/

namespace {
    const string defaultWorkDir = "/tmp/.cds";
    constexpr size_t ramDiskSizeMb = 10;
}

/*******************************************************************************
 UTILITY FUNCTIONS
*******************************************************************************/

// TODO: add more test datasets

void generateTestFile1(const string& filename) {
    ofstream file(filename);
    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";
    file.close();
    PRINTLN("Test file created: %s", filename.c_str());
}

void generateTestFile2(const string& filename) {
    ofstream file(filename);
    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";

    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";

    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";

    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";

    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";

    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";

    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";

    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";

    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";

    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";

    file.close();
    PRINTLN("Test file created: %s", filename.c_str());
}

void generateTestFiles1(const string& directory, const int count) {
    for (int i = 1; i <= count; i++) {
        generateTestFile1(directory + "/input/test" + to_string(i) + ".csv");
    }
}

void generateTestFiles2(const string& directory, const int count) {
    for (int i = 1; i <= count; i++) {
        generateTestFile2(directory + "/input/test" + to_string(i) + ".csv");
    }
}

/*******************************************************************************
 TESTS
*******************************************************************************/

void testProcessInputFiles() {
    assert(InitStorage(defaultWorkDir.c_str(), ramDiskSizeMb));

    generateTestFiles1(defaultWorkDir, 3);

    assert(ProcessInputFiles());

    int recordCount, fieldCount;
    GetFieldAndRecordCount(&recordCount, &fieldCount);
    assert(recordCount == 9);
    assert(fieldCount == 3);

    assert(CloseStorage());
}

void testAnalyzeData1() {
    assert(InitStorage(defaultWorkDir.c_str(), ramDiskSizeMb));

    generateTestFiles1(defaultWorkDir, 5);
    setWorkingDirectory(defaultWorkDir);

    assert(ProcessInputFiles());
    assert(AnalyzeData());

    int recordCount, fieldCount;
    GetFieldAndRecordCount(&recordCount, &fieldCount);

    assert(recordCount == 15);
    assert(fieldCount == 3);

    const auto minimums = new float[recordCount];
    const auto maximums = new float[recordCount];
    const auto totals = new float[recordCount];
    const auto means = new float[recordCount];
    const auto stdDevs = new float[recordCount];

    GetStats(minimums, maximums,totals, means, stdDevs);

    constexpr float epsilon = 0.0001f;

    assert(fabs(minimums[0] - 1.1f) < epsilon);
    assert(fabs(maximums[0] - 7.7f) < epsilon);
    assert(fabs(totals[0] - 66.0f) < epsilon);
    assert(fabs(means[0] - 4.4f) < epsilon);
    assert(fabs(stdDevs[0] - 2.6944f) < epsilon);

    assert(fabs(minimums[1] - 2.2f) < epsilon);
    assert(fabs(maximums[1] - 8.8f) < epsilon);
    assert(fabs(totals[1] - 82.5f) < epsilon);
    assert(fabs(means[1] - 5.5f) < epsilon);
    assert(fabs(stdDevs[1] - 2.6944f) < epsilon);

    assert(fabs(minimums[2] - 3.3f) < epsilon);
    assert(fabs(maximums[2] - 9.9f) < epsilon);
    assert(fabs(totals[2] - 99.0f) < epsilon);
    assert(fabs(means[2] - 6.6f) < epsilon);
    assert(fabs(stdDevs[2] - 2.6944f) < epsilon);

    assert(CloseStorage());
}

void testAnalyzeData2() {
    assert(InitStorage(defaultWorkDir.c_str(), ramDiskSizeMb));

    generateTestFiles2(defaultWorkDir, 5);
    setWorkingDirectory(defaultWorkDir);

    assert(ProcessInputFiles());
    assert(AnalyzeData());

    int recordCount, fieldCount;
    GetFieldAndRecordCount(&recordCount, &fieldCount);

    assert(recordCount == 150);
    assert(fieldCount == 3);

    const auto minimums = new float[recordCount];
    const auto maximums = new float[recordCount];
    const auto totals = new float[recordCount];
    const auto means = new float[recordCount];
    const auto stdDevs = new float[recordCount];

    GetStats(minimums, maximums,totals, means, stdDevs);

    constexpr float epsilon = 0.0001f;

    assert(fabs(minimums[0] - 1.1f) < epsilon);
    assert(fabs(maximums[0] - 7.7f) < epsilon);
    assert(fabs(totals[0] - 660.0f) < epsilon);
    assert(fabs(means[0] - 4.4f) < epsilon);
    assert(fabs(stdDevs[0] - 2.6944f) < epsilon);

    assert(fabs(minimums[1] - 2.2f) < epsilon);
    assert(fabs(maximums[1] - 8.8f) < epsilon);
    assert(fabs(totals[1] - 825.0f) < epsilon);
    assert(fabs(means[1] - 5.5f) < epsilon);
    assert(fabs(stdDevs[1] - 2.6944f) < epsilon);

    assert(fabs(minimums[2] - 3.3f) < epsilon);
    assert(fabs(maximums[2] - 9.9f) < epsilon);
    assert(fabs(totals[2] - 990.0f) < epsilon);
    assert(fabs(means[2] - 6.6f) < epsilon);
    assert(fabs(stdDevs[2] - 2.6944f) < epsilon);

    assert(CloseStorage());
}

/*******************************************************************************
 MAIN
*******************************************************************************/

int main() {
    setVerbose(true);

    testProcessInputFiles();
    testAnalyzeData1();
    testAnalyzeData2();

    printf("All tests completed successfully.\n");
    exit(EXIT_SUCCESS);
}
