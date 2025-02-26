CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++
CXXFLAGS = -std=c++17 -Wall
NVCCFLAGS = -arch=sm_50 -std=c++17 -Xcompiler -Wall -I.
LDFLAGS = -lcudart

GOL_IMPL ?= /dev/null
GOL_IMPL_DIR = $(dir $(GOL_IMPL))

BIN_DIR = $(GOL_IMPL_DIR)/bin
OBJ_DIR = $(BIN_DIR)/obj
EXECUTABLE = $(BIN_DIR)/game_of_life

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ_DIR)/main.o $(OBJ_DIR)/gol.o
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(OBJ_DIR)/main.o $(OBJ_DIR)/gol.o $(LDFLAGS) -L$(CUDA_PATH)/lib64

$(OBJ_DIR)/main.o: main.cpp gol.cuh
	mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c main.cpp -o $(OBJ_DIR)/main.o

$(OBJ_DIR)/gol.o: $(GOL_IMPL) gol.cuh
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $(GOL_IMPL) -o $(OBJ_DIR)/gol.o

clean:
	rm -rf $(BIN_DIR)
