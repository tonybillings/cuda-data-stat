package _test

import (
	"github.com/tonybillings/gocds/_test"
	"github.com/tonybillings/gocds/cds"
	"testing"
)

func TestInitCloseStorage(t *testing.T) {
	if cds.InitStorage("", 1024) {
		t.Error("Expected InitStorage to fail due to invalid/missing directory")
	}

	if cds.InitStorage(_test.WorkingDir, 0) {
		t.Error("Expected InitStorage to fail due to invalid storage size 0")
	}

	if !cds.InitStorage(_test.WorkingDir, 1024) {
		t.Error("Failed to initialize storage")
		t.FailNow()
	}

	if !cds.CloseStorage() {
		t.Error("Failed to close storage")
	}
}
