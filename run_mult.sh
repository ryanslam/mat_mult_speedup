echo "$1 Threads" >> log_file_tiling.txt
export OMP_PLACES=cores; export OMP_PROC_BIND=true; ./mat_mult_omp 1000 1000 1000 -t $1 >> log_file_tiling.txt 
export OMP_PLACES=cores; export OMP_PROC_BIND=true; ./mat_mult_omp 1000 2000 5000 -t $1 >> log_file_tiling.txt 
export OMP_PLACES=cores; export OMP_PROC_BIND=true; ./mat_mult_omp 9000 2500 3750 -t $1 >> log_file_tiling.txt
echo "" >> log_file_tiling.txt