# ─── Compilers ───────────────────────────────────────────────
NVCC     := nvcc
CXX      := g++
NVCCFLAGS := -std=c++17 -O2
CXXFLAGS  := -std=c++17 -O2

# ─── Directories ─────────────────────────────────────────────
SRC_DIR   := src
BENCH_DIR := benchmarks
INC_DIR   := include
BUILD_DIR := build

# ─── Common CPP sources (shared library code) ───────────────
COMMON_CPP := $(wildcard $(SRC_DIR)/*.cpp)
COMMON_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(COMMON_CPP))

# ─── CUDA targets (each .cu in src/ → one binary) ───────────
CU_SRCS   := $(wildcard $(SRC_DIR)/*.cu)
CU_TARGETS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(CU_SRCS))

# ─── Benchmark targets (each .cpp in benchmarks/ → one binary)
BENCH_SRCS    := $(wildcard $(BENCH_DIR)/*.cpp)
BENCH_TARGETS := $(patsubst $(BENCH_DIR)/%.cpp,$(BUILD_DIR)/%,$(BENCH_SRCS))

# ─── All targets ─────────────────────────────────────────────
ALL_TARGETS := $(CU_TARGETS) $(BENCH_TARGETS)

.PHONY: all clean

all: $(ALL_TARGETS)

# ─── Common .cpp → .o ───────────────────────────────────────
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

# ─── CUDA binary (link .cu + common objects) ─────────────────
$(BUILD_DIR)/%: $(SRC_DIR)/%.cu $(COMMON_OBJ) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR) -o $@ $< $(COMMON_OBJ)

# ─── Benchmark binary (link .cpp + common objects) ───────────
$(BUILD_DIR)/%: $(BENCH_DIR)/%.cpp $(COMMON_OBJ) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -o $@ $< $(COMMON_OBJ)

# ─── Create build directory ─────────────────────────────────
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
