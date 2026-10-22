# v2.1 runtime build.
#   make host   - CPU build (x86 dev box; parity checks against the python oracle)
#   make nano   - aarch64 + sm_53 build (run on the Jetson Nano)
#   make gen    - regenerate the graph tables from spec/v21_layers.json
CC      ?= gcc
NVCC    ?= /usr/local/cuda/bin/nvcc
PY      ?= .venv/bin/python
CFLAGS  ?= -O2 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
GEN      = runtime/v21/gen/pt_graph.c
SRC      = runtime/v21/src/pt_v21_host.c runtime/v21/src/main_v21.c
INC      = -Iruntime/v21/src -Iruntime/v21/gen

host: build/pt_v21_host

build/pt_v21_host: $(SRC) $(GEN) runtime/v21/src/pt_v21.h runtime/v21/gen/pt_graph_tab.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) $(SRC) $(GEN) -lm -o $@

nano: build/pt_v21_nano

build/pt_v21_nano: $(SRC) $(GEN) runtime/v21/src/pt_v21_cuda.cu runtime/v21/src/pt_v21.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -DPT_WITH_CUDA -c runtime/v21/src/pt_v21_host.c -o build/pt_v21_host.o
	$(CC) $(CFLAGS) $(INC) -DPT_WITH_CUDA -c runtime/v21/src/main_v21.c -o build/main_v21.o
	$(CC) $(CFLAGS) $(INC) -DPT_WITH_CUDA -c $(GEN) -o build/pt_graph.o
	$(NVCC) -O3 -arch=sm_53 -std=c++11 runtime/v21/src/pt_v21_cuda.cu -c -o build/pt_v21_cuda.o
	$(CC) build/pt_v21_host.o build/main_v21.o build/pt_graph.o build/pt_v21_cuda.o -lm -o $@

gen:
	$(PY) tools/gen_v21_graph.py

clean:
	rm -f build/pt_v21_host build/pt_v21_nano build/*.o
.PHONY: host nano gen clean
