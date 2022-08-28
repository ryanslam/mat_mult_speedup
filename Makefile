CC = g++

default: matmult_omp

matmult_omp: $(SRC)
	export OMP_PLACES=cores; export OMP_PROC_BIND=true; ${CC} -O3 -Wall -Wextra -fopenmp -o $@ matmult_omp.cpp

clean:
	-rm -f matmult_omp