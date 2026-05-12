CXX := g++
CXXFLAGS := -std=c++17 -O2
TARGET := serial_naive

SRCS_BASE := src/parser.cpp src/utils.cpp src/pretty_printer.cpp src/printer.cpp
SRCS_NAIVE := benchmarks/serial_naive.cpp $(SRCS_BASE)
SRCS_SMARTSUM := benchmarks/serial_smartsum.cpp $(SRCS_BASE)
OBJS_BASE := $(SRCS_BASE:.cpp=.o)
OBJS_NAIVE := $(SRCS_NAIVE:.cpp=.o)
OBJS_SMARTSUM := $(SRCS_SMARTSUM:.cpp=.o)

all: $(TARGET) serial_smartsum

$(TARGET): $(OBJS_NAIVE)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_NAIVE)

serial_smartsum: $(OBJS_SMARTSUM)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_SMARTSUM)

%.o: %.cpp include/parser.hpp include/utils.hpp include/pretty_printer.hpp include/printer.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) serial_smartsum $(OBJS_BASE) $(OBJS_NAIVE) $(OBJS_SMARTSUM)
