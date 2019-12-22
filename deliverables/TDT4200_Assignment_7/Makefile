FLAGS := -arch=sm_61 -rdc=true

.PHONY: clean

task1b: libs/bitmap.c libs/bitmap.c task1b.cu
	nvcc libs/bitmap.c task1b.cu

task1e: libs/bitmap.c libs/bitmap.c task1e.cu
	nvcc libs/bitmap.c task1e.cu

task2b: libs/bitmap.c libs/bitmap.c task2b.cu
	nvcc $(FLAGS) libs/bitmap.c task2b.cu

serial: libs/bitmap.c libs/bitmap.c serial.cu
	nvcc libs/bitmap.c serial.cu

clean:
	rm -Rf *.o
	rm -Rf main

# end
