# Build the GPU-LBM review benchmark.
#   make                 -> ./lbmbench for RTX 50-series (sm_120, CUDA >= 12.8)
#   make ARCH=native     -> build for the GPU in this machine
#   make FAST=1          -> D3Q19 only (halves compile time)
NVCC  ?= nvcc
ARCH  ?= sm_120
FLAGS  = -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo
ifeq ($(ARCH),native)
  ARCHFLAG = -arch=native
else
  ARCHFLAG = -arch=$(ARCH)
endif
ifeq ($(FAST),1)
  FLAGS += -DONLY_D3Q19
endif

lbmbench: lbmbench.cu
	$(NVCC) $(FLAGS) $(ARCHFLAG) -Xptxas -v $< -o $@ 2> ptxas_report.txt || (cat ptxas_report.txt; exit 1)
	@grep -c "Compiling entry" ptxas_report.txt | xargs echo "kernels compiled:"
	@grep "spill" ptxas_report.txt | grep -v " 0 bytes spill stores, 0 bytes spill loads" | head -5 || true

clean:
	rm -f lbmbench ptxas_report.txt
.PHONY: clean
