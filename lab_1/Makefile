CXX = g++
CXXFLAGS = -std=c++20 -fopenmp
TARGET = lab1.exe
SRC = lab_1.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
