CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

TARGETS = engine data_gen

all: $(TARGETS)

engine: engine.cpp common.h
	$(CXX) $(CXXFLAGS) engine.cpp -o engine

data_gen: data_gen.cpp common.h
	$(CXX) $(CXXFLAGS) data_gen.cpp -o data_gen

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
