Parallelized Matrix Multiplication:
    - Files/directories:
        * /log_files
        * matmult_omp.cpp
        * Makefile
        * run_mult.sh

    - How to run:
        * ./matmult_omp <nrows> <mcols> <mcols2> -t <num_threads>
            * Example: ./matmult_omp 9000 2500 3750 -t 20

    - Makefile:
        * Compiles and creates an executable for matmult_omp.cpp.
        * make clean: removes the executable.

    - run_mult.sh:
        * Script logged the results from my computations to a txt file.
    
    - /log_files:
        * Directory contains experiment results from runs on the different parameters.
