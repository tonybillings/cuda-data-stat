/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/storage.h"

#include <iostream>
#include <vector>
#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

/*******************************************************************************
 USINGS
*******************************************************************************/

using namespace std;

/*******************************************************************************
 STORAGE FUNCTIONS
*******************************************************************************/

bool create_ramdisk(const string& mount_point, const size_t size_mb) {
    cout << "Preparing mount point..." << endl;

    cout << "Creating mount point directory: " << mount_point << endl;
    if (mkdir(mount_point.c_str(), 0755) != 0 && errno != EEXIST) {
        cerr << "Error: unable to create mount point '" << mount_point << "': " << strerror(errno) << endl;
        return false;
    }

    struct stat st{};
    if (stat(mount_point.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        cerr << "Error: '" << mount_point << "' is not a directory." << endl;
        return false;
    }

    cout << "Mounting ramfs to '" << mount_point << "'..." << endl;
    const string mount_options = "size=" + to_string(size_mb) + "m";
    if (mount("ramfs", mount_point.c_str(), "ramfs", 0, mount_options.c_str()) != 0) {
        cerr << "Error: unable to mount ramfs to '" << mount_point << "': " << strerror(errno) << endl;
        return false;
    }

    cout << "ramfs created and mounted successfully." << endl;
    return true;
}

bool create_directory_structure(const string& mount_point) {
    cout << "Creating directory structure in '" << mount_point << "'..." << endl;

    vector<string> dirs = {"/inactive", "/active", "/add", "/remove", "/.add", "/.remove"};
    for (const auto& dir : dirs) {
        string path = mount_point + dir;
        if (mkdir(path.c_str(), 0755) != 0) {
            cerr << "Error: unable to create directory '" << path << "': " << strerror(errno) << endl;
            return false;
        }
    }

    cout << "Directory structure created successfully." << endl;
    return true;
}

void cleanup(const string& mount_point) {
    cout << "Cleaning up..." << endl;

    if (umount(mount_point.c_str()) != 0) {
        cerr << "Error: failed to unmount ramfs '" << mount_point << "': " << strerror(errno) << endl;
    } else {
        cout << "Unmounted ramfs '" << mount_point << "'" << endl;
    }

    if (rmdir(mount_point.c_str()) != 0) {
        cerr << "Error: failed to delete mount point '" << mount_point << "': " << strerror(errno) << endl;
    } else {
        cout << "Deleted mount point '" << mount_point << "'" << endl;
    }
}
