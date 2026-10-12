# Standalone tools for the bit-exact ALF GPU work
#
#   make bench            -> alf_bench   (needs nvcc + NVML, i.e. a CUDA toolkit with the driver installed)
#   make test_core        -> test_alf_core (needs libvvenc built in ../../build/<BUILD>, CPU only)
#
# Variables: CUDA_ARCH (default native), NVCC, BUILD (cmake build dir name, default base)

NVCC      ?= nvcc
CUDA_ARCH ?= native
BUILD     ?= base
ROOT      := ../..

bench: alf_bench
CL := $(ROOT)/source/Lib/CommonLib
alf_bench: alf_bench.cu $(CL)/AlfGpuHost.cpp $(CL)/AlfGpuCore.h $(CL)/AlfGpuTiles.h $(CL)/AlfGpu.h $(CL)/cuda/AlfGpuKernels.cuh $(CL)/cuda/AlfGpuCuda.cu
	$(NVCC) -O3 -std=c++14 -arch=$(CUDA_ARCH) -lineinfo -DVVENC_CUDA_ALF=1 -o $@ alf_bench.cu $(CL)/AlfGpuHost.cpp $(CL)/cuda/AlfGpuCuda.cu -lnvidia-ml

test_core: test_alf_core
test_alf_core: test_alf_core.cpp $(ROOT)/source/Lib/CommonLib/AlfGpuCore.h
	g++ -O2 -std=gnu++14 -flto=auto -DTARGET_SIMD_X86=1 -DHAVE_PTHREADS \
	    -I$(ROOT)/include -I$(ROOT)/build/$(BUILD) -I$(ROOT)/source/Lib -I$(ROOT)/source/Lib/CommonLib -I$(ROOT)/source/Lib/CommonLib/x86 \
	    test_alf_core.cpp $(ROOT)/lib/release-static/libvvenc.a -lpthread -o $@

clean:
	rm -f alf_bench test_alf_core

.PHONY: bench test_core clean
