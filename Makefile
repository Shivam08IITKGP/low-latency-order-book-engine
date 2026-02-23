CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native -mtune=native -fno-omit-frame-pointer -Wall -Wextra -pthread

# Source files for the modular order book
ORDERBOOK_SRCS = main.cpp timing.cpp cpu_utils.cpp messages.cpp order_book.cpp threads.cpp latency.cpp
ORDERBOOK_OBJS = $(ORDERBOOK_SRCS:.cpp=.o)

TARGETS = orderbook data_gen

all: $(TARGETS)

orderbook: $(ORDERBOOK_OBJS)
	$(CXX) $(CXXFLAGS) $(ORDERBOOK_OBJS) -o orderbook

data_gen: data_gen.cpp common.h
	$(CXX) $(CXXFLAGS) data_gen.cpp -o data_gen

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
