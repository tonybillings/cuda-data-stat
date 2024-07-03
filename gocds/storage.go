package main

/*
#include "cds/cds.h"
*/
import "C"

import (
	"unsafe"
)

func InitStorage(workingDir string, sizeMb uint) bool {
	cWorkingDir := C.CString(workingDir)
	defer C.free(unsafe.Pointer(cWorkingDir))

	return bool(C.InitStorage(cWorkingDir, C.size_t(sizeMb)))
}

func CloseStorage() bool {
	return bool(C.CloseStorage())
}
