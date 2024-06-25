/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/service.h"
#include "cds/storage.h"
#include "cds/stats.cuh"
#include "cds/debug.h"

#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>

/*******************************************************************************
 USINGS
*******************************************************************************/

using std::string;
using std::stringstream;
using std::ifstream;
using std::vector;
using std::invalid_argument;

/*******************************************************************************
 INTERNAL FUNCTIONS
*******************************************************************************/

bool isCsvFile(const string& filename) {
    const size_t fnSize = filename.size();
    constexpr size_t extSize = 4;
    if (const string ext = ".csv"; fnSize >= extSize &&
        filename.compare(fnSize - extSize, extSize, ext) == 0) {
        return true;
    }
    return false;
}

bool processCsvFile(const string& filePath, vector<double>& data, DataStats& stats) {
    ifstream file(filePath);
    if (!file.is_open()) {
        ERROR("unable to open file '%s': %s", filePath.c_str(), strerror(errno));
        return false;
    }

    string line;
    stats.fieldCount = 0;
    stats.recordCount = 0;
    vector<double> fileData;

    while (getline(file, line)) {
        stringstream ss(line);
        string field;
        size_t numFields = 0;

        while (getline(ss, field, ',')) {
            try {
                double value = stod(field);
                fileData.push_back(value);
                numFields++;
            } catch (const invalid_argument&) {
                ERROR("invalid data found in CSV file '%s'", filePath.c_str());
                return false;
            }
        }

        if (stats.fieldCount == 0) {
            stats.fieldCount = numFields;
        } else if (numFields != stats.fieldCount) {
            ERROR("inconsistent row lengths in CSV file '%s'", filePath.c_str());
            return false;
        }

        stats.recordCount++;
    }

    if (fileData.empty()) {
        ERROR("empty file '%s'", filePath.c_str());
        return false;
    }

    data = move(fileData);
    return true;
}

bool processInputFiles(const string& workingDir, DataStats& stats) {
    const string inputDir = workingDir + "/input";
    if (!checkDirectory(inputDir)) {
        return false;
    }

    if (DIR* dir; (dir = opendir(inputDir.c_str())) != nullptr) {
        const string dataFile = workingDir + "/data";
        dirent *ent;
        while ((ent = readdir(dir)) != nullptr) {
            string filename = ent->d_name;
            if (filename == "." || filename == ".." || !isCsvFile(filename)) continue;

            string filePath = inputDir;
            filePath.append("/").append(filename);

            vector<double> fileData;
            DataStats st;
            if (!processCsvFile(filePath, fileData, st)) {
                closedir(dir);
                return false;
            }

            stats.fieldCount = st.fieldCount;
            stats.recordCount += st.recordCount;

            if (!appendData(dataFile, reinterpret_cast<char*>(fileData.data()), fileData.size() * sizeof(double))) {
                closedir(dir);
                return false;
            }
        }

        closedir(dir);
        return true;
    }

    ERROR("unable to open directory '%s': %s", inputDir.c_str(), strerror(errno));
    return false;
}

bool analyzeData(const std::string& workingDir, DataStats& stats) {
    if (!checkDirectory(workingDir)) {
        return false;
    }

    const string dataFile = workingDir + "/data";
    char* data = nullptr;
    size_t dataSize = 0;
    if (!mapData(dataFile, data, dataSize)) {
        return false;
    }

    if (!calculateStats(data, dataSize, stats)) {
        return false;
    }

    if (!unmapData(data, dataSize)) {
        return false;
    }

    return true;
}

/*******************************************************************************
 PUBLIC INTERFACE
*******************************************************************************/

extern "C" {
    bool ProcessInputFiles() {
        const string workingDir = getWorkingDirectory();
        if (!checkDirectory(workingDir)) {
            return false;
        }

        DataStats ds;
        if (!processInputFiles(workingDir, ds)) {
            return false;
        }
        stats::set(ds);

        return true;
    }

    bool AnalyzeData() {
        const string workingDir = getWorkingDirectory();
        if (!checkDirectory(workingDir)) {
            return false;
        }

        DataStats ds = stats::get();
        if (!analyzeData(workingDir, ds)) {
            return false;
        }
        stats::set(ds);

        return true;
    }

    void GetFieldAndRecordCount(int* recordCount, int* fieldCount) {
        const DataStats ds = stats::get();
        *recordCount = static_cast<int>(ds.recordCount);
        *fieldCount = static_cast<int>(ds.fieldCount);
    }

    void GetStats(double* minimums, double* maximums, double* totals, double* means, double* stdDevs,
        double* deltaMinimums, double* deltaMaximums, double* deltaTotals, double* deltaMeans, double* deltaStdDevs) {
        const DataStats ds = stats::get();
        for (int i = 0; i < ds.fieldCount; i++) {
            minimums[i] = ds.minimums[i];
            maximums[i] = ds.maximums[i];
            totals[i] = ds.totals[i];
            means[i] = ds.means[i];
            stdDevs[i] = ds.stdDevs[i];

            deltaMinimums[i] = ds.deltaMinimums[i];
            deltaMaximums[i] = ds.deltaMaximums[i];
            deltaTotals[i] = ds.deltaTotals[i];
            deltaMeans[i] = ds.deltaMeans[i];
            deltaStdDevs[i] = ds.deltaStdDevs[i];
        }
    }
}
