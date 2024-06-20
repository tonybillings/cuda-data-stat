/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/debug.h"

#include <string>
#include <mutex>

/*******************************************************************************
 USINGS
*******************************************************************************/

using std::string;
using std::mutex;
using std::lock_guard;

/*******************************************************************************
 STATE
*******************************************************************************/

namespace {
    bool verbose = false;
    mutex verboseMutex;

    string lastError;
    mutex errorMutex;
}

/*******************************************************************************
 INTERNAL FUNCTIONS
*******************************************************************************/

bool getVerbose() {
    lock_guard lock(verboseMutex);
    return verbose;
}

void setVerbose(const bool value) {
    lock_guard lock(verboseMutex);
    verbose = value;
}

string getLastError() {
    lock_guard lock(errorMutex);
    return lastError;
}

void setLastError(const string& msg) {
    lock_guard lock(errorMutex);
    lastError = msg;
}

/*******************************************************************************
 PUBLIC INTERFACE
*******************************************************************************/

extern "C" {
    void EnableVerboseMode(const bool enabled) {
        setVerbose(enabled);
    }

    const char* GetLastError() {
        return getLastError().c_str();
    }
}
