package cds

/*
#include "cds/cds.h"
*/
import "C"

func EnableVerboseMode(enabled bool) {
	C.EnableVerboseMode(C.bool(enabled))
}

func GetLastError() string {
	return C.GoString(C.GetLastError())
}
