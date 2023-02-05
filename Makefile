NVCC=nvcc
NVCCFLAGS=-std=c++17 -Xcompiler='-fopenmp'
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-I./src/cutf/include -lcusolver -lcublas -lcurand -I./src/matfile/include

TARGET=pseudo-inv-pair

$(TARGET):src/main.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS)
  
clean:
	rm -f $(TARGET)
