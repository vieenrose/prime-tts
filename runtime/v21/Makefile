# v2.1 runtime build.
#   make host   - CPU build (x86 dev box; parity against the python oracle and the goldens)
#   make nano   - aarch64 + sm_53 build (Jetson Nano: conv/convT/matmul on the GPU)
#   make gen    - regenerate the graph tables from spec/v21_layers.json
CC      ?= gcc
NVCC    ?= /usr/local/cuda/bin/nvcc
PY      ?= .venv/bin/python
CFLAGS  ?= -O2 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
CUF     ?= -O3 -arch=sm_53 -std=c++11
CUDA    ?= /usr/local/cuda
GEN      = runtime/v21/gen/pt_graph.c
HDR      = runtime/v21/src/pt_v21.h runtime/v21/gen/pt_graph_tab.h
SRC      = runtime/v21/src/pt_v21_host.c runtime/v21/src/main_v21.c
INC      = -Iruntime/v21/src -Iruntime/v21/gen

host: build/pt_v21_host

build/pt_v21_host: $(SRC) $(GEN) $(HDR) runtime/v21/src/pt_v21_host.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) $(SRC) $(GEN) -lm -o $@

# The CUDA build defines PT_V21_CUDA so pt_v21_host.c drops its own copies of the
# pt_v21_conv/convT/matmul dispatchers; pt_v21_cuda.cu provides them and forwards to the CPU
# kernels when ctx->use_cuda is 0, so --cuda and the CPU path share one binary.
nano: build/pt_v21_nano

build/pt_v21_nano: $(SRC) $(GEN) runtime/v21/src/pt_v21_cuda.cu $(HDR)
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -DPT_V21_CUDA -c runtime/v21/src/pt_v21_host.c -o build/pt_v21_host_cu.o
	$(CC) $(CFLAGS) $(INC) -c runtime/v21/src/main_v21.c -o build/main_v21_cu.o
	$(CC) $(CFLAGS) $(INC) -c $(GEN) -o build/pt_graph_cu.o
	$(NVCC) $(CUF) $(INC) -c runtime/v21/src/pt_v21_cuda.cu -o build/pt_v21_cuda.o
	$(NVCC) $(CUF) -o $@ build/pt_v21_cuda.o build/pt_v21_host_cu.o build/pt_graph_cu.o \
	  build/main_v21_cu.o -L$(CUDA)/lib64 -lcudart -lm

gen:
	$(PY) tools/gen_v21_graph.py

clean:
	rm -f build/pt_v21_host build/pt_v21_nano build/*.o
.PHONY: host nano gen clean
