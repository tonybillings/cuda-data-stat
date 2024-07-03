package main

import (
	"fmt"
	"strings"
)

type DataStats struct {
	FieldCount    uint
	RecordCount   uint
	Minimums      []float64
	Maximums      []float64
	Totals        []float64
	Means         []float64
	StdDevs       []float64
	DeltaMinimums []float64
	DeltaMaximums []float64
	DeltaTotals   []float64
	DeltaMeans    []float64
	DeltaStdDevs  []float64
}

func NewDataStats(fields uint) *DataStats {
	return &DataStats{
		FieldCount:    fields,
		Minimums:      make([]float64, fields),
		Maximums:      make([]float64, fields),
		Totals:        make([]float64, fields),
		Means:         make([]float64, fields),
		StdDevs:       make([]float64, fields),
		DeltaMinimums: make([]float64, fields),
		DeltaMaximums: make([]float64, fields),
		DeltaTotals:   make([]float64, fields),
		DeltaMeans:    make([]float64, fields),
		DeltaStdDevs:  make([]float64, fields),
	}
}

func (ds *DataStats) String() string {
	var sb strings.Builder

	sb.WriteString("FieldCount: ")
	sb.WriteString(fmt.Sprint(ds.FieldCount))
	sb.WriteString("\n")

	sb.WriteString("RecordCount: ")
	sb.WriteString(fmt.Sprint(ds.RecordCount))
	sb.WriteString("\n")

	sb.WriteString("Minimums: ")
	sb.WriteString(fmt.Sprint(ds.Minimums))
	sb.WriteString("\n")

	sb.WriteString("Maximums: ")
	sb.WriteString(fmt.Sprint(ds.Maximums))
	sb.WriteString("\n")

	sb.WriteString("Totals: ")
	sb.WriteString(fmt.Sprint(ds.Totals))
	sb.WriteString("\n")

	sb.WriteString("Means: ")
	sb.WriteString(fmt.Sprint(ds.Means))
	sb.WriteString("\n")

	sb.WriteString("Standard Deviations: ")
	sb.WriteString(fmt.Sprint(ds.StdDevs))
	sb.WriteString("\n")

	sb.WriteString("Delta Minimums: ")
	sb.WriteString(fmt.Sprint(ds.DeltaMinimums))
	sb.WriteString("\n")

	sb.WriteString("Delta Maximums: ")
	sb.WriteString(fmt.Sprint(ds.DeltaMaximums))
	sb.WriteString("\n")

	sb.WriteString("Delta Totals: ")
	sb.WriteString(fmt.Sprint(ds.DeltaTotals))
	sb.WriteString("\n")

	sb.WriteString("Delta Means: ")
	sb.WriteString(fmt.Sprint(ds.DeltaMeans))
	sb.WriteString("\n")

	sb.WriteString("Delta Standard Deviations: ")
	sb.WriteString(fmt.Sprint(ds.DeltaStdDevs))

	return sb.String()
}
