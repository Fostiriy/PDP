CXX = g++
CXXFLAGS = -std=c++20 -mavx2 -mavx512vl
TARGET = lab2.exe
SRC = lab_2.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean
