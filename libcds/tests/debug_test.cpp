/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/storage.h"
#include "cds/service.h"
#include "cds/debug.h"
#include "cds/cds.h"

#include <string>
#include <cassert>

/*******************************************************************************
 USINGS
*******************************************************************************/

using std::string;

/*******************************************************************************
 PARAMETERS
*******************************************************************************/

namespace {
    constexpr bool isVerbose = false;
    const string defaultWorkDir = "/tmp/.cds_test";
}

/*******************************************************************************
 TESTS
*******************************************************************************/

void testSetLastError() {
    PRINTLN("\nRunning testSetLastError()...");

    clearLastError();
    assert(getLastError().empty());

    setLastError("Testing 123");
    assert(!getLastError().empty());
    assert(getLastError() == "Testing 123");
}

void testGetLastError() {
    PRINTLN("\nRunning testGetLastError()...");

    clearLastError();
    assert(getLastError().empty());
    assert(!InitStorage("", 0));
    assert(string(GetLastError()).empty() == 0);

    assert(InitStorage(defaultWorkDir.c_str(), 1));
    assert(string(GetLastError()).empty() == 1);

    assert(CloseStorage());
    assert(string(GetLastError()).empty() == 1);
}

/*******************************************************************************
 MAIN
*******************************************************************************/

int main() {
    setVerbose(isVerbose);

    testSetLastError();
    testGetLastError();

    printf("All tests completed successfully.\n");
    exit(EXIT_SUCCESS);
}
