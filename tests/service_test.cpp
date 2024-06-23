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
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>

/*******************************************************************************
 USINGS
*******************************************************************************/

using std::string;
using std::ofstream;
using std::vector;
using std::min;
using std::to_string;
using std::fabs;
using std::cout;
using std::endl;
using std::move;
using std::min_element;
using std::max_element;
using std::accumulate;
using std::this_thread::sleep_for;
using std::chrono::milliseconds;

/*******************************************************************************
 PARAMETERS
*******************************************************************************/

namespace {
    const string defaultWorkDir = "/tmp/.cds";
    constexpr size_t ramDiskSizeMb = 10;
    constexpr bool isVerbose = true;
    constexpr size_t maxFileCount = 10;
    constexpr size_t maxRecordCount = 10;
    constexpr size_t maxFieldCount = 10;
}

/*******************************************************************************
 UTILITY FUNCTIONS
*******************************************************************************/

DataStats generateTestFiles(const std::string& directory, const int fileCount, const int recordCount, const int fieldCount) {
    if (!checkDirectory(directory)) {
        exit(EXIT_FAILURE);
    }

    if (fileCount < 1 || recordCount < 1 || fieldCount < 1) {
        ERROR("invalid parameters for generateTestFiles()");
        exit(EXIT_FAILURE);
    }

    vector<vector<float>> data(fieldCount);
    int totalRecords = 0;

    for (int i = 1; i <= fileCount; i++) {
        string filename = directory + "/input/test_data_" + to_string(i) + ".csv";
        ofstream file(filename);

        for (int j = 0; j < recordCount; j++) {
            const auto x = static_cast<float>(j % 3) * 3.3f;

            for (int k = 0; k < fieldCount; k++) {
                int fieldIndex = k % 3;
                constexpr auto a = 1.1f;
                constexpr auto b = 2.2f;
                constexpr auto c = 3.3f;

                float value = (fieldIndex == 0 ? a : (fieldIndex == 1 ? b : c)) + x;
                file << to_string(value).substr(0, 3);

                if (k < fieldCount - 1) {
                    file << ",";
                }

                data[k].push_back(value);
            }

            file << "\n";
            totalRecords++;
        }

        file.close();
        cout << "Test file created: " << filename << endl;
    }

    DataStats ds(fieldCount);
    ds.recordCount = totalRecords;
    for (int i = 0; i < fieldCount; i++) {
        ds.minimums[i] = *min_element(data[i].begin(), data[i].end());
        ds.maximums[i] = *max_element(data[i].begin(), data[i].end());
        float total = accumulate(data[i].begin(), data[i].end(), 0.0f);
        float mean = total / static_cast<float>(data[i].size());
        ds.totals[i] = total;
        ds.means[i] = mean;

        float sum = 0.0f;
        for (int j = 0; j < totalRecords; j++) {
            sum += powf(data[i][j] - mean, 2);
        }
        ds.stdDevs[i] = sqrtf(sum / static_cast<float>(totalRecords));
    }

    if (totalRecords > 1) {
        for (int i = 0; i < fieldCount; i++) {
            float previous = data[i][0];
            data[i][0] = fabs(data[i][1] - data[i][0]);

            for (int j = 1; j < totalRecords; j++) {
                float current = data[i][j];
                data[i][j] = fabs(current - previous);
                previous = current;
            }
        }

        for (int i = 0; i < fieldCount; i++) {
            ds.deltaMinimums[i] = *min_element(data[i].begin(), data[i].end());
            ds.deltaMaximums[i] = *max_element(data[i].begin(), data[i].end());
            float total = accumulate(data[i].begin(), data[i].end(), 0.0f);
            float mean = total / static_cast<float>(data[i].size());
            ds.deltaTotals[i] = total;
            ds.deltaMeans[i] = mean;

            float sum = 0.0f;
            for (int j = 0; j < totalRecords; j++) {
                sum += powf(data[i][j] - mean, 2);
            }
            ds.deltaStdDevs[i] = sqrtf(sum / static_cast<float>(totalRecords));
        }
    } else {
        for (int i = 0; i < fieldCount; i++) {
            ds.deltaMinimums[i] = data[i][0];
            ds.deltaMaximums[i] = data[i][0];
            ds.deltaMeans[i] = data[i][0];
            ds.deltaTotals[i] = data[i][0];
        }
    }

    return ds;
}

/*******************************************************************************
 TESTS  // TODO: add more test cases
*******************************************************************************/

void testProcessInputFiles() {
    assert(InitStorage(defaultWorkDir.c_str(), ramDiskSizeMb));

    generateTestFiles(defaultWorkDir, 3, 3, 3);

    assert(ProcessInputFiles());

    int recordCount, fieldCount;
    GetFieldAndRecordCount(&recordCount, &fieldCount);
    assert(recordCount == 9);
    assert(fieldCount == 3);

    assert(CloseStorage());
}

void testAnalyzeData(const int fileCount, const int recordCount, const int fieldCount) {
    assert(InitStorage(defaultWorkDir.c_str(), ramDiskSizeMb));

    const DataStats expectedStats = generateTestFiles(defaultWorkDir, fileCount, recordCount, fieldCount);
    setWorkingDirectory(defaultWorkDir);

    assert(ProcessInputFiles());
    assert(AnalyzeData());

    int totalRecordCount, actualFieldCount;
    GetFieldAndRecordCount(&totalRecordCount, &actualFieldCount);

    assert(totalRecordCount == fileCount * recordCount);
    assert(actualFieldCount == fieldCount);

    const DataStats actualStats = stats::get();
    assert(actualStats == expectedStats && "actual DataStats does not equal expected DataStats");

    assert(CloseStorage());
}

/*******************************************************************************
 MAIN
*******************************************************************************/

int main() {
    setVerbose(isVerbose);

    testProcessInputFiles();

    for (int i = 1; i <= maxFileCount; i++) {
        for (int j = 1; j <= maxRecordCount; j++) {
            for (int k = 1; k <= maxFieldCount; k++) {
                testAnalyzeData(i, j, k);
                sleep_for(milliseconds(25));
            }
        }
    }

    printf("All tests completed successfully.\n");
    exit(EXIT_SUCCESS);
}
