/*******************************************************************************
 INCLUDES
*******************************************************************************/

#include "cds/data_stats.h"

#include <mutex>

/*******************************************************************************
 USINGS
*******************************************************************************/

using std::mutex;
using std::lock_guard;

/*******************************************************************************
 STATE
*******************************************************************************/

namespace stats {
    DataStats dataStats_;
    mutex dataStatsMutex;

    DataStats get() {
        lock_guard lock(dataStatsMutex);
        return dataStats_;
    }

    void set(const DataStats& dataStats) {
        lock_guard lock(dataStatsMutex);
        dataStats_ = dataStats;
    }
}
