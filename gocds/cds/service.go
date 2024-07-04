package cds

/*
#include "cds/cds.h"
*/
import "C"

import (
	"unsafe"
)

func ProcessInputFiles() bool {
	return bool(C.ProcessInputFiles())
}

func AnalyzeData() bool {
	return bool(C.AnalyzeData())
}

func GetFieldAndRecordCount() (int, int) {
	var fieldCount, recordCount C.int
	C.GetFieldAndRecordCount(&fieldCount, &recordCount)
	return int(fieldCount), int(recordCount)
}

func GetStats() *DataStats {
	fieldCount, recordCount := GetFieldAndRecordCount()
	if fieldCount == 0 || recordCount == 0 {
		return NewDataStats(uint(fieldCount))
	}

	stats := NewDataStats(uint(fieldCount))
	C.GetStats(
		(*C.double)(unsafe.Pointer(&stats.Minimums[0])),
		(*C.double)(unsafe.Pointer(&stats.Maximums[0])),
		(*C.double)(unsafe.Pointer(&stats.Totals[0])),
		(*C.double)(unsafe.Pointer(&stats.Means[0])),
		(*C.double)(unsafe.Pointer(&stats.StdDevs[0])),
		(*C.double)(unsafe.Pointer(&stats.DeltaMinimums[0])),
		(*C.double)(unsafe.Pointer(&stats.DeltaMaximums[0])),
		(*C.double)(unsafe.Pointer(&stats.DeltaTotals[0])),
		(*C.double)(unsafe.Pointer(&stats.DeltaMeans[0])),
		(*C.double)(unsafe.Pointer(&stats.DeltaStdDevs[0])),
	)

	stats.RecordCount = uint(recordCount)
	return stats
}
