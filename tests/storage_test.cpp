/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/storage.h"

#include <cstring>
#include <sys/stat.h>
#include <iostream>
#include <string>
#include <sys/mount.h>

/*******************************************************************************
 USINGS
*******************************************************************************/

using namespace std;

/*******************************************************************************
 UTIL
*******************************************************************************/

bool directory_exists(const string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

/*******************************************************************************
 TESTS
*******************************************************************************/

void test_create_ramdisk(const string& mount_point, const size_t size_mb) {
    cout << "Testing create_ramdisk..." << endl;

    if (create_ramdisk(mount_point, size_mb) && directory_exists(mount_point)) {
        cout << "Testing create_ramdisk...PASS." << endl;
    } else {
        cerr << "Testing create_ramdisk...FAIL." << endl;
        exit(EXIT_FAILURE);
    }
}

void test_create_directory_structure(const string& mount_point) {
    cout << "Testing create_directory_structure..." << endl;

    if (create_directory_structure(mount_point) &&
        directory_exists(mount_point + "/inactive") &&
        directory_exists(mount_point + "/active") &&
        directory_exists(mount_point + "/add") &&
        directory_exists(mount_point + "/remove") &&
        directory_exists(mount_point + "/.add") &&
        directory_exists(mount_point + "/.remove"))
    {
        cout << "Testing create_directory_structure...PASS." << endl;
    } else {
        cerr << "Testing create_directory_structure...FAIL." << endl;
        exit(EXIT_FAILURE);
    }
}

void test_cleanup(const string& mount_point) {
    cout << "Testing cleanup..." << endl;

    cleanup(mount_point);
    if (!directory_exists(mount_point)) {
        cout << "Testing cleanup...PASS." << endl;
    } else {
        cerr << "Testing cleanup...FAIL." << endl;
        exit(EXIT_FAILURE);
    }
}

/*******************************************************************************
 MAIN
*******************************************************************************/

int main() {
    const string mount_point = "/mnt/ramdisk_test";
    constexpr size_t size_mb = 100;

    test_create_ramdisk(mount_point, size_mb);
    test_create_directory_structure(mount_point);
    test_cleanup(mount_point);

    return 0;
}
