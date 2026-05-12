CXX := g++
CXXFLAGS := -std=c++17 -O2
TARGET := serial_naive

SRCS_BASE := src/parser.cpp src/utils.cpp src/pretty_printer.cpp src/printer.cpp
SRCS_NAIVE := benchmarks/serial_naive.cpp $(SRCS_BASE)
OBJS_BASE := $(SRCS_BASE:.cpp=.o)
OBJS_NAIVE := $(SRCS_NAIVE:.cpp=.o)


all: $(TARGET)

$(TARGET): $(OBJS_NAIVE)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_NAIVE)

%.o: %.cpp include/parser.hpp include/utils.hpp include/pretty_printer.hpp include/printer.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS_BASE) $(OBJS_NAIVE)
