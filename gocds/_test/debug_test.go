package _test

import (
	"github.com/tonybillings/gocds/_test"
	"github.com/tonybillings/gocds/cds"
	"testing"
)

func TestGetLastError(t *testing.T) {
	if lastError := cds.GetLastError(); lastError != "" {
		t.Error("Expected last error to be empty")
	}

	if cds.InitStorage(_test.WorkingDir, 0) {
		t.Error("Expected InitStorage to fail due to invalid storage size 0")
	}

	if lastError := cds.GetLastError(); lastError == "" {
		t.Error("Expected last error not to be empty")
	} else {
		const msg = "storage size must be greater than zero megabytes"
		if lastError != msg { // warning: fragile!
			t.Errorf("Expected last error to equal '%s', got '%s'", msg, lastError)
		}
	}

	if !cds.InitStorage(_test.WorkingDir, 1) {
		t.Error("Failed to initialize storage")
		t.FailNow()
	}

	if lastError := cds.GetLastError(); lastError != "" {
		t.Error("Expected last error to be empty")
	}

	if !cds.CloseStorage() {
		t.Error("Failed to close storage")
		t.FailNow()
	}

	if lastError := cds.GetLastError(); lastError != "" {
		t.Error("Expected last error to be empty")
	}
}
