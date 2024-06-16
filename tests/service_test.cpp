/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/storage.h"
#include "cds/service.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

/*******************************************************************************
 USINGS
*******************************************************************************/

using namespace std;

/*******************************************************************************
 UTIL
*******************************************************************************/

void generate_test_file(const string& filename) {
    ofstream file(filename);
    file << "1.1,2.2,3.3\n";
    file << "4.4,5.5,6.6\n";
    file << "7.7,8.8,9.9\n";
    // TODO: improve test datasets
    // file << "1.1,2.2,3.3\n";
    // file << "4.4,5.5,6.6\n";
    // file << "7.7,8.8,9.9\n";
    //
    // file << "1.1,2.2,3.3\n";
    // file << "4.4,5.5,6.6\n";
    // file << "7.7,8.8,9.9\n";
    //
    // file << "1.1,2.2,3.3\n";
    // file << "4.4,5.5,6.6\n";
    // file << "7.7,8.8,9.9\n";
    //
    // file << "1.1,2.2,3.3\n";
    // file << "4.4,5.5,6.6\n";
    // file << "7.7,8.8,9.9\n";
    //
    // file << "1.1,2.2,3.3\n";
    // file << "4.4,5.5,6.6\n";
    // file << "7.7,8.8,9.9\n";
    //
    // file << "1.1,2.2,3.3\n";
    // file << "4.4,5.5,6.6\n";
    // file << "7.7,8.8,9.9\n";
    //
    // file << "1.1,2.2,3.3\n";
    // file << "4.4,5.5,6.6\n";
    // file << "7.7,8.8,9.9\n";
    //
    // file << "1.1,2.2,3.3\n";
    // file << "4.4,5.5,6.6\n";
    // file << "7.7,8.8,9.9\n";
    //
    // file << "1.1,2.2,3.3\n";
    // file << "4.4,5.5,6.6\n";
    // file << "7.7,8.8,9.9\n";

    file.close();
    cout << "Test file created: " << filename << endl;
}

void generate_test_files(const string& directory, const int count) {
    for (int i = 1; i <= count; ++i) {
        generate_test_file(directory + "/test" + to_string(i) + ".csv");
    }
}

/*******************************************************************************
 MAIN
*******************************************************************************/

int main() {
    const string mount_point = "/mnt/ramdisk_test";
    constexpr size_t ramdisk_size_mb = 100 * 1024 * 1024;
    constexpr int test_file_count = 5;

    if (!create_ramdisk(mount_point, ramdisk_size_mb)) {
        cerr << "Failed to create ramdisk." << endl;
        exit(EXIT_FAILURE);
    }

    if (!create_directory_structure(mount_point)) {
        cerr << "Failed to create directory structure." << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Filesystem setup complete." << endl;

    const string add_directory = mount_point + "/add";
    generate_test_files(add_directory, test_file_count);

    vector<char> data;
    vector<char> mask;
    unordered_map<string, unique_ptr<FileMetadata>> index;
    process_files(mount_point, data, mask, index);

    cout << "Index after processing:" << endl;
    size_t num_fields = 0;
    for (const auto&[filename, metadata] : index) {
        num_fields = metadata->field_count;
        cout << "\t- File: " << filename
            << ", Start: " << metadata->index_start << ", End: " << metadata->index_end
            << ", #Fields: " << metadata->field_count << ", #Records: " << metadata->field_count << endl;
    }

    cout << "Mask after processing:" << endl;
    for (const auto& m : mask) {
        cout << static_cast<int>(m) << " ";
    }
    cout << endl;

    cout << "Analyzing data..." << endl;
    if (DataStats stats; !analyze_data(data, num_fields, stats)) {
        cerr << "Failed to analyze data." << endl;
        exit(EXIT_FAILURE);
    } else {
        cout << "Finished analyzing data:" << endl;
        cout << "\tTotals: ";
        for (const auto& t : stats.totals) {
            cout << t << ", ";
        }
        cout << endl;
        cout << "\tAverages: ";
        for (const auto& m : stats.means) {
            cout << m << ", ";
        }
        cout << endl;
        cout << "\tStd Devs: ";
        for (const auto& s : stats.std_devs) {
            cout << s << ", ";
        }
        cout << endl;
    }

    cleanup(mount_point);

    return 0;
}
