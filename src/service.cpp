/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/service.h"
#include "cds/cuda_error_check.h"
#include "cds/file_metadata.h"

#include <cstring>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

/*******************************************************************************
 USINGS
*******************************************************************************/

using namespace std;

/*******************************************************************************
 SERVICE FUNCTIONS
*******************************************************************************/

bool process_csv_file(const string& file_path, vector<float>& data, size_t& field_count, size_t& record_count) {
    ifstream file(file_path);
    string line;
    field_count = 0;
    record_count = 0;
    vector<float> file_data;

    while (getline(file, line)) {
        stringstream ss(line);
        string field;
        size_t num_fields = 0;

        while (getline(ss, field, ',')) {
            try {
                float value = stof(field);
                file_data.push_back(value);
                num_fields++;
            } catch (const invalid_argument&) {
                cerr << "Error: invalid data found in CSV file '" << file_path << "'" << endl;
                return false;
            }
        }

        if (field_count == 0) {
            field_count = num_fields;
        } else if (num_fields != field_count) {
            cerr << "Error: inconsistent row lengths in CSV file '" << file_path << "'" << endl;
            return false;
        }

        record_count++;
    }

    if (file_data.empty()) {
        cerr << "Error: empty file '" << file_path << "'" << endl;
        return false;
    }

    data = move(file_data);
    return true;
}

bool process_add_directory(const string& add_dir, vector<char>& data, vector<char>& mask, unordered_map<string, unique_ptr<FileMetadata>>& index) {
    if (DIR* dir; (dir = opendir(add_dir.c_str())) != nullptr) {
        dirent *ent;
        while ((ent = readdir(dir)) != nullptr) {
            string file_name = ent->d_name;
            if (file_name == "." || file_name == "..") continue;

            // TODO: add support for binary (pre-processed) files

            string file_path = add_dir;
            file_path.append("/").append(file_name);

            vector<float> file_data;
            size_t field_count, record_count;
            process_csv_file(file_path, file_data, field_count, record_count);

            size_t start_pos = data.size();
            data.insert(data.end(), reinterpret_cast<char*>(file_data.data()), reinterpret_cast<char*>(file_data.data() + file_data.size()));
            size_t end_pos = data.size();
            index[file_name] = make_unique<FileMetadata>(file_path, field_count, record_count, start_pos, end_pos);

            mask.insert(mask.end(), record_count, 1);
        }

        closedir(dir);
        return true;
    }

    cerr << "Error: could not open directory" << endl;
    return false;
}

bool process_files(const string& mount_point, vector<char>& data, vector<char>& mask, unordered_map<string, unique_ptr<FileMetadata>>& index) {
    return process_add_directory(mount_point + "/add", data, mask, index);
    // TODO: process other directories, update mask, use mask to determine which data contributes to stats
}

bool analyze_data(const vector<char>& data, const size_t field_count, DataStats& stats) {
    if (const size_t record_count = data.size() / (field_count * sizeof(float)); !calculate_stats(data, field_count, record_count, stats)) {
        cerr << "Error: calculate_stats failed" << endl;
        return false;
    }

    return true;
}
