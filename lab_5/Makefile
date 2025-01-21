CXX = g++
CXXFLAGS = -std=c++20 -pthread
TARGET = lab5.exe
SRC = lab_5.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
