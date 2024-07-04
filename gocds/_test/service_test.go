package _test

import (
	"github.com/stretchr/testify/assert"
	"github.com/tonybillings/gocds/_test"
	"github.com/tonybillings/gocds/cds"
	"testing"
)

func TestProcessInputFiles(t *testing.T) {
	_test.BeginTest(t)
	defer _test.EndTest(t)

	if count := _test.GetInputFileCount(); count != 0 {
		t.Errorf("Expected file count of 0, got %d", count)
		t.FailNow()
	}

	_test.AddTestDataRow()

	if count := _test.GetInputFileCount(); count != 1 {
		t.Errorf("Expected file count of 1, got %d", count)
		t.FailNow()
	}

	if !cds.ProcessInputFiles() {
		t.Error("Failed to process input files")
		t.FailNow()
	}

	if count := _test.GetInputFileCount(); count != 0 {
		t.Errorf("Expected file count of 0, got %d", count)
		t.FailNow()
	}
}

func TestAnalyzeData(t *testing.T) {
	_test.BeginTest(t)
	defer _test.EndTest(t)

	if !cds.AnalyzeData() {
		t.Error("Failed to analyze data with 0 records")
		t.FailNow()
	}

	_test.AddTestDataRow()

	if !cds.ProcessInputFiles() {
		t.Error("Failed to process input files")
		t.FailNow()
	}

	if !cds.AnalyzeData() {
		t.Error("Failed to analyze data with 1 record")
		t.FailNow()
	}
}

func TestGetFieldAndRecordCount(t *testing.T) {
	_test.BeginTest(t)
	defer _test.EndTest(t)

	if fieldCount, recordCount := cds.GetFieldAndRecordCount(); fieldCount != 0 || recordCount != 0 {
		t.Errorf("Unexpected field/record count: expected 0/0, got %d/%d", fieldCount, recordCount)
		t.FailNow()
	}

	_test.AddTestDataRow()

	if !cds.ProcessInputFiles() {
		t.Error("Failed to process input files")
		t.FailNow()
	}

	if !cds.AnalyzeData() {
		t.Error("Failed to analyze data")
		t.FailNow()
	}

	if fieldCount, recordCount := cds.GetFieldAndRecordCount(); fieldCount != 3 || recordCount != 1 {
		t.Errorf("Unexpected field/record count: expected 3/1, got %d/%d", fieldCount, recordCount)
		t.FailNow()
	}

	_test.AddTestDataRow(4)

	if !cds.ProcessInputFiles() {
		t.Error("Failed to process input files")
		t.FailNow()
	}

	if !cds.AnalyzeData() {
		t.Error("Failed to analyze data")
		t.FailNow()
	}

	if fieldCount, recordCount := cds.GetFieldAndRecordCount(); fieldCount != 3 || recordCount != 5 {
		t.Errorf("Unexpected field/record count: expected 3/5, got %d/%d", fieldCount, recordCount)
		t.FailNow()
	}
}

func TestGetStats(t *testing.T) {
	_test.BeginTest(t)
	defer _test.EndTest(t)

	if ds := cds.GetStats(); ds.FieldCount != 0 || ds.RecordCount != 0 {
		t.Errorf("Unexpected field/record count: expected 0/0, got %d/%d", ds.FieldCount, ds.RecordCount)
		t.FailNow()
	}

	_test.AddTestDataRow(333)

	if !cds.ProcessInputFiles() {
		t.Error("Failed to process input files")
		t.FailNow()
	}

	if !cds.AnalyzeData() {
		t.Error("Failed to analyze data")
		t.FailNow()
	}

	if ds := cds.GetStats(); ds.FieldCount != 3 || ds.RecordCount != 333 {
		t.Errorf("Unexpected field/record count: expected 3/333, got %d/%d", ds.FieldCount, ds.RecordCount)
		t.FailNow()
	} else {
		assert.InDelta(t, 1.1, ds.Minimums[0], 0.000001, "unexpected value for Minimums[0]")
		assert.InDelta(t, 2.2, ds.Minimums[1], 0.000001, "unexpected value for Minimums[1]")
		assert.InDelta(t, 3.3, ds.Minimums[2], 0.000001, "unexpected value for Minimums[2]")

		assert.InDelta(t, 7.7, ds.Maximums[0], 0.000001, "unexpected value for Maximums[0]")
		assert.InDelta(t, 8.8, ds.Maximums[1], 0.000001, "unexpected value for Maximums[1]")
		assert.InDelta(t, 9.9, ds.Maximums[2], 0.000001, "unexpected value for Maximums[2]")

		assert.InDelta(t, 1465.2, ds.Totals[0], 0.000001, "unexpected value for Totals[0]")
		assert.InDelta(t, 1831.5, ds.Totals[1], 0.000001, "unexpected value for Totals[1]")
		assert.InDelta(t, 2197.7999999, ds.Totals[2], 0.000001, "unexpected value for Totals[2]")

		assert.InDelta(t, 4.4, ds.Means[0], 0.000001, "unexpected value for Means[0]")
		assert.InDelta(t, 5.5, ds.Means[1], 0.000001, "unexpected value for Means[1]")
		assert.InDelta(t, 6.6, ds.Means[2], 0.000001, "unexpected value for Means[2]")

		assert.InDelta(t, 2.6944387, ds.StdDevs[0], 0.000001, "unexpected value for StdDevs[0]")
		assert.InDelta(t, 2.6944387, ds.StdDevs[1], 0.000001, "unexpected value for StdDevs[1]")
		assert.InDelta(t, 2.6944387, ds.StdDevs[2], 0.000001, "unexpected value for StdDevs[2]")

		assert.InDelta(t, 3.3, ds.DeltaMinimums[0], 0.000001, "unexpected value for DeltaMinimums[0]")
		assert.InDelta(t, 3.3, ds.DeltaMinimums[1], 0.000001, "unexpected value for DeltaMinimums[1]")
		assert.InDelta(t, 3.3, ds.DeltaMinimums[2], 0.000001, "unexpected value for DeltaMinimums[2]")

		assert.InDelta(t, 6.6, ds.DeltaMaximums[0], 0.000001, "unexpected value for DeltaMaximums[0]")
		assert.InDelta(t, 6.6, ds.DeltaMaximums[1], 0.000001, "unexpected value for DeltaMaximums[1]")
		assert.InDelta(t, 6.6, ds.DeltaMaximums[2], 0.000001, "unexpected value for DeltaMaximums[2]")

		assert.InDelta(t, 1461.9, ds.DeltaTotals[0], 0.000001, "unexpected value for DeltaTotals[0]")
		assert.InDelta(t, 1461.9, ds.DeltaTotals[1], 0.000001, "unexpected value for DeltaTotals[1]")
		assert.InDelta(t, 1461.9, ds.DeltaTotals[2], 0.000001, "unexpected value for DeltaTotals[2]")

		assert.InDelta(t, 4.39009, ds.DeltaMeans[0], 0.000001, "unexpected value for DeltaMeans[0]")
		assert.InDelta(t, 4.39009, ds.DeltaMeans[1], 0.000001, "unexpected value for DeltaMeans[1]")
		assert.InDelta(t, 4.39009, ds.DeltaMeans[2], 0.000001, "unexpected value for DeltaMeans[2]")

		assert.InDelta(t, 1.5520956, ds.DeltaStdDevs[0], 0.000001, "unexpected value for DeltaStdDevs[0]")
		assert.InDelta(t, 1.5520956, ds.DeltaStdDevs[1], 0.000001, "unexpected value for DeltaStdDevs[1]")
		assert.InDelta(t, 1.5520956, ds.DeltaStdDevs[2], 0.000001, "unexpected value for DeltaStdDevs[2]")
	}
}
