#ifndef DEBUG_H
#define DEBUG_H

#include <cstdio>
#include <string>

bool getVerbose();
void setVerbose(bool value);
std::string getLastError();
void setLastError(const std::string& msg);

extern "C" {
    void EnableVerboseMode(bool enabled);
    const char* GetLastError();
}

#define PRINT(msg, ...) do { if (getVerbose()) printf(msg, ##__VA_ARGS__); } while (0)
#define PRINTLN(msg, ...) do { if (getVerbose()) printf(msg "\n", ##__VA_ARGS__); } while (0)
#define ERROR(msg, ...) do { setLastError(msg); if (getVerbose()) fprintf(stderr,"ERROR: " msg "\n", ##__VA_ARGS__); } while (0)

#endif //DEBUG_H
