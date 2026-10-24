# v2.1 runtime build.
#   make host   - CPU build (x86 dev box; parity against the python oracle and the goldens)
#   make nano   - aarch64 + sm_53 build (Jetson Nano: conv/convT/matmul on the GPU)
#   make gen    - regenerate the graph tables from spec/v21_layers.json
CC      ?= gcc
NVCC    ?= /usr/local/cuda/bin/nvcc
PY      ?= .venv/bin/python
# -O3: gcc 8 only auto-vectorizes at -O3 (-ftree-vectorize); the interpreter's elementwise fast
# paths (bin_op, un_op, slice/pad/transpose rows) rely on it for NEON on the A57. No fast-math:
# fp semantics are unchanged and float reductions are not reassociated.
CFLAGS  ?= -O3 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
CUF     ?= -O3 -arch=sm_53 -std=c++11
CUDA    ?= /usr/local/cuda
GEN      = runtime/v21/gen/pt_graph.c
HDR      = runtime/v21/src/pt_v21.h runtime/v21/src/pt_cpucap.h runtime/v21/gen/pt_graph_tab.h runtime/v21/Makefile   # flag changes rebuild
# Fingerprint of the effective toolchain flags: an env-provided CFLAGS/CUF change must also
# rebuild (a stale binary would silently benchmark the wrong code). Content changes only when
# the flags do, so steady-state builds stay no-ops.
BFLAGS   := build/flags.mk
BFORCE: ;
$(BFLAGS): BFORCE
	@mkdir -p build
	@echo "CC=$(CC) CFLAGS=$(CFLAGS) NVCC=$(NVCC) CUF=$(CUF) CUDA=$(CUDA)" > $@.new
	@if cmp -s $@.new $@ 2>/dev/null; then rm -f $@.new; else mv $@.new $@; fi
HDR      += $(BFLAGS)
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

build/pt_v21_nano: $(SRC) $(GEN) runtime/v21/src/pt_v21_cuda.cu $(HDR) build/pt_sass_cubin.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -DPT_V21_CUDA -c runtime/v21/src/pt_v21_host.c -o build/pt_v21_host_cu.o
	$(CC) $(CFLAGS) $(INC) -c runtime/v21/src/main_v21.c -o build/main_v21_cu.o
	$(CC) $(CFLAGS) $(INC) -c $(GEN) -o build/pt_graph_cu.o
	$(NVCC) $(CUF) $(INC) -Ibuild -DPT_V21_SASS_EMBED -c runtime/v21/src/pt_v21_cuda.cu -o build/pt_v21_cuda.o
	$(NVCC) $(CUF) -o $@ build/pt_v21_cuda.o build/pt_v21_host_cu.o build/pt_graph_cu.o \
	  build/main_v21_cu.o -L$(CUDA)/lib64 -lcudart -lcuda -lm

# Hand-assembled kernels (runtime/v21/sass): compile a template cubin of pt_v21_cuda.cu, assemble
# each .masm source into its function's .text with masm.py (encodings learned from this very
# toolchain's cuobjdump listing; no third-party assembler), embed with bin2c. Loaded at run time
# through the driver API behind PT_V21_SASS=1 (default off: the nvcc-compiled kernels run).
PY3     ?= python3
SASSDIR  = runtime/v21/sass
SASS_IG8 = _Z18pt_conv_kernel_ig8ILi1EEvPKfliiiiPflliS1_S1_lliifflj
build/pt_sass_cubin.h: runtime/v21/src/pt_v21_cuda.cu $(SASSDIR)/masm.py $(SASSDIR)/conv_ig8.masm $(HDR)
	@mkdir -p build
	$(NVCC) $(CUF) $(INC) -cubin runtime/v21/src/pt_v21_cuda.cu -o build/sass_tmpl.cubin
	$(CUDA)/bin/cuobjdump -sass build/sass_tmpl.cubin > build/sass_tmpl.sass
	$(PY3) $(SASSDIR)/masm.py patch build/sass_tmpl.cubin build/sass_tmpl.sass $(SASS_IG8) \
	  $(SASSDIR)/conv_ig8.masm build/sass.cubin
	$(CUDA)/bin/bin2c -c --name pt_sass_cubin build/sass.cubin > $@

# M3 multi-stream driver: N concurrent requests on one GPU (needs CUDA on the Nano)
par: build/pt_v21_par

build/pt_v21_par: $(SRC) runtime/v21/src/main_v21_par.c $(GEN) runtime/v21/src/pt_v21_cuda.cu $(HDR) build/pt_sass_cubin.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -DPT_V21_CUDA -c runtime/v21/src/pt_v21_host.c -o build/pt_v21_host_par.o
	$(CC) $(CFLAGS) $(INC) -c runtime/v21/gen/pt_graph.c -o build/pt_graph_par.o
	$(NVCC) $(CUF) $(INC) -Ibuild -DPT_V21_SASS_EMBED -c runtime/v21/src/pt_v21_cuda.cu -o build/pt_v21_cuda_par.o
	$(NVCC) $(CUF) $(INC) -o $@ build/pt_v21_cuda_par.o build/pt_v21_host_par.o build/pt_graph_par.o \
	  runtime/v21/src/main_v21_par.c -L$(CUDA)/lib64 -lcudart -lcuda -lpthread -lm

# Persistent server: warm contexts, OpenAI-shaped POST /v1/audio/speech (see main_v21_serve.c)
serve: build/pt_v21_serve

build/pt_v21_serve: build/pt_v21_par runtime/v21/src/main_v21_serve.c
	$(NVCC) $(CUF) $(INC) -o $@ build/pt_v21_cuda_par.o build/pt_v21_host_par.o build/pt_graph_par.o \
	  runtime/v21/src/main_v21_serve.c -L$(CUDA)/lib64 -lcudart -lcuda -lpthread -lm

gen:
	$(PY) tools/gen_v21_graph.py

clean:
	rm -f build/pt_v21_host build/pt_v21_nano build/pt_v21_par build/pt_v21_serve build/*.o
.PHONY: host nano par serve gen clean BFORCE
