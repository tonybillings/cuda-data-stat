package _test

import (
	"github.com/tonybillings/gocds/cds"
	"testing"
)

func Begin(t *testing.T) {
	if !cds.InitStorage(WorkingDir, 1024) {
		t.Error("Failed to initialize storage")
		t.FailNow()
	}
}

func End(t *testing.T) {
	if !cds.CloseStorage() {
		t.Error("Failed to close storage")
		t.FailNow()
	}
}
