CXX        = nvcc -O3

#-arch=sm_60

SRC_GPU = main-gpu.cu
SRC_CPU = main-omp.c
HELPERS = ../helper.h goldenSeq.h parallelPlan.h spatial-only-opt.h
KERNELS = kernels.cu.h
EXEC_GPU= mmm-gpu
EXEC_CPU= mmm-omp

default: compile_cpu run_cpu compile_gpu run_gpu

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<


compile_gpu: $(EXEC_GPU)
compile_cpu: $(EXEC_CPU)

$(EXEC_GPU): $(SRC_GPU) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(EXEC_GPU) $(SRC_GPU)

$(EXEC_CPU): $(SRC_CPU) $(HELPERS)
	g++ -O3 -fopenmp -o $(EXEC_CPU) $(SRC_CPU)


run_gpu: $(EXEC_GPU)
	./$(EXEC_GPU) 2048 4096 2048

run_cpu: $(EXEC_CPU)
	./$(EXEC_CPU) 2048 1024 2048

clean:
	rm -f $(EXEC_CPU) $(EXEC_GPU)

