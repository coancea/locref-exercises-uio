CXX        = nvcc -O3

SRC_GPU = main-gpu.cu
SRC_CPU = main-omp.c
HELPERS = ../helper.h parallelPlan.h goldenSeq.h
KERNELS = kernels.cu.h
EXEC_GPU = histo-gpu
EXEC_CPU = histo-cpu

default: compile_cpu run_cpu compile_gpu run_gpu

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile_cpu: $(EXEC_CPU)
compile_gpu: $(EXEC_GPU)

$(EXEC_GPU): $(SRC_GPU) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(EXEC_GPU) $(SRC_GPU)

$(EXEC_CPU): $(SRC_CPU) $(HELPERS)
	g++ -O3 -fopenmp -o $(EXEC_CPU) $(SRC_CPU)

run_gpu: $(EXEC_GPU)
	./$(EXEC_GPU) 536870912 41943040

run_cpu: $(EXEC_CPU)
	./$(EXEC_CPU) 536870912 134217728

clean:
	rm -f $(EXEC_GPU) $(EXEC_CPU)

