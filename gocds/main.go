package main

import (
	"fmt"
	"os"
)

// TODO: move this test code where it belongs, begin implementing GUI
func main() {
	if InitStorage("/tmp/.cds", 1024) {
		fmt.Println("Storage initialized successfully.")
	} else {
		fmt.Println("Failed to initialize storage.")
		fmt.Println("Error:", GetLastError())
		return
	}

	addTestData()
	
	if ProcessInputFiles() {
		fmt.Println("Input files processed successfully.")
	} else {
		fmt.Println("Failed to process input files.")
		fmt.Println("Error:", GetLastError())
	}

	if AnalyzeData() {
		fmt.Println("Data analyzed successfully.")

		stats := GetStats()
		if stats != nil {
			fmt.Println(stats.String())
		} else {
			fmt.Println("Failed to get stats.")
			fmt.Println("Error:", GetLastError())
		}
	} else {
		fmt.Println("Failed to analyze data.")
		fmt.Println("Error:", GetLastError())
	}

	if CloseStorage() {
		fmt.Println("Storage closed successfully.")
	} else {
		fmt.Println("Failed to close storage.")
		fmt.Println("Error:", GetLastError())
	}
}

func addTestData() {
	f, err := os.OpenFile("/tmp/.cds/input/testdata.csv", os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		fmt.Println("Failed to open file:", err)
		return
	}
	defer func(f *os.File) {
		e := f.Close()
		if e != nil {
			panic(e)
		}
	}(f)

	data := "1,2,3\n4,5,6\n7,8,9\n"

	if _, err = f.WriteString(data); err != nil {
		fmt.Println("Failed to write to file:", err)
		return
	}
}
