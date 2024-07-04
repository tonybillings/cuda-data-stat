package _test

import (
	"fmt"
	"os"
	"path"
)

func AddTestData(data string) {
	dataFile, err := os.OpenFile(fmt.Sprintf("%s/input/testdata.csv", WorkingDir), os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		fmt.Println("Failed to open file:", err)
		return
	}
	defer func(f *os.File) {
		e := f.Close()
		if e != nil {
			panic(e)
		}
	}(dataFile)

	if _, err = dataFile.WriteString(data); err != nil {
		fmt.Println("Failed to write to file:", err)
		return
	}
}

func AddTestDataRow(count ...uint) {
	c := uint(1)
	if len(count) > 0 {
		c = count[0]
	}

	rows := []string{"1.1,2.2,3.3\n", "4.4,5.5,6.6\n", "7.7,8.8,9.9\n"}
	for i := uint(0); i < c; i++ {
		data := rows[i%3]
		AddTestData(data)
	}
}

func GetInputFileCount() int {
	files, err := os.ReadDir(path.Join(WorkingDir, "input"))
	if err != nil {
		return -1
	}
	return len(files)
}
