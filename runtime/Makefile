# PrimeTTS v1b_8k CPU runtime.
#   native (Nano):  make -C runtime                 -> build/aarch64/{test_kernels,test_e2e,bench_cpu}
#   x86 host:       make -C runtime                 -> build/x86_64/...   (portable C kernels)
#   cross (qemu):   make -C runtime CC=aarch64-linux-gnu-gcc ARCH=aarch64
ARCH ?= $(shell uname -m)
CC ?= gcc
ROOT := ..
OUT := $(ROOT)/build/$(ARCH)
CFLAGS ?= -O3 -g
CFLAGS += -Wall -Wno-comment -std=gnu11 -Iinclude -fno-math-errno -pthread
ifeq ($(ARCH),aarch64)
  CFLAGS += -mcpu=cortex-a57
  ASM := kern/k4x16.S kern/k8x12.S kern/act.S kern/dwln.S
endif
SRC := src/weights.c src/kern_ref.c src/conv.c src/npy.c src/graph_v1b8k.c src/sched.c
OBJ := $(patsubst %.c,$(OUT)/obj/%.o,$(SRC)) $(patsubst %.S,$(OUT)/obj/%.o,$(ASM))
LIBS := -lm -lpthread

all: $(OUT)/libpt.a $(OUT)/test_kernels $(OUT)/test_e2e $(OUT)/bench_cpu

$(OUT)/obj/%.o: %.c include/pt.h include/pt_v1b8k.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
$(OUT)/obj/%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) -c $< -o $@
$(OUT)/libpt.a: $(OBJ)
	ar rcs $@ $^
$(OUT)/test_kernels: $(ROOT)/tests/test_kernels.c $(OUT)/libpt.a
	$(CC) $(CFLAGS) $< $(OUT)/libpt.a -o $@ $(LIBS)
$(OUT)/test_e2e: $(ROOT)/tests/test_e2e.c $(OUT)/libpt.a
	$(CC) $(CFLAGS) $< $(OUT)/libpt.a -o $@ $(LIBS)
$(OUT)/bench_cpu: $(ROOT)/bench/cpu/bench_cpu.c $(OUT)/libpt.a
	$(CC) $(CFLAGS) $< $(OUT)/libpt.a -o $@ $(LIBS)
clean:
	rm -rf $(OUT)
.PHONY: all clean
