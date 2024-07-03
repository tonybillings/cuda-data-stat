/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/debug.h"

#include <string>
#include <cstring>
#include <mutex>
#include <cstdarg>

/*******************************************************************************
 USINGS
*******************************************************************************/

using std::string;
using std::strncpy;
using std::mutex;
using std::lock_guard;

/*******************************************************************************
 STATE
*******************************************************************************/

namespace {
    bool verbose = false;
    mutex verboseMutex;

    string lastError;
    char lastErrorBuffer[1024];
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
    strncpy(lastErrorBuffer, lastError.c_str(), sizeof(lastErrorBuffer) - 1);
    lastErrorBuffer[sizeof(lastErrorBuffer) - 1] = '\0';
}

void setLastErrorFormatted(const char* format, ...) {
    lock_guard lock(errorMutex);
    va_list args;
    va_start(args, format);
    vsnprintf(lastErrorBuffer, sizeof(lastErrorBuffer), format, args);
    va_end(args);
    lastError = lastErrorBuffer;
}

/*******************************************************************************
 PUBLIC INTERFACE
*******************************************************************************/

extern "C" {
    void EnableVerboseMode(const bool enabled) {
        setVerbose(enabled);
    }

    const char* GetLastError() {
        return lastErrorBuffer;
    }
}
