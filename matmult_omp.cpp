#include <stdlib.h>
#include <iostream>
#include <vector>       /* Used to contain a 2D contiguous array */
#include <iomanip>      /* Included for setting print precision */
#include <omp.h>        /* omp */
#include <time.h>       /* time */
#include <cstring>      /* strcasecmp */

#define CACHE_SIZE 128

using namespace std;

// Zeros inputted matrix:
void clear_matrix(vector<double> &mat, int nrows, int mcols){
    for(int i = 0; i < nrows; i++){
        for(int j = 0; j < mcols; j++){
            mat[(i*mcols) + j] = 0;
        }
    }
}

// Populates the matrices.
void fill_matrix(vector<double> &mat, int nrows, int mcols) {
    // We use for instead of collapse because there is a dependency upon previous loops.
    for(int i = 0; i < nrows; i++){
        for(int j = 0; j < mcols; j++)
            mat[(i*mcols) + j] = (double)(rand() % 1000) / 353.0;
    }
} 

// Transposes the matrix (Can only use if its a square matrix multiplication).
// If it is however, the transposition of matrix B prior to multiplication increase speed.
void transpose_matrix(vector<double> &mat, vector<double> &mat2, int nrows, int mcols){
    #pragma omp parallel for
    for(int i = 0; i < nrows; i++){
        for(int j = 0; j < mcols; j++){  
            mat2[j*mcols + i] = mat[i*nrows + j];
        }
    }
}

// Unparallelized matrix multiplication example provided from assignment.
void matrix_mult_example(vector<double> &A, vector<double> &B, vector<double> &C,
                            int nrows, int mcols, int mcols2){
    double sum;
    for(int i = 0; i < nrows; i++){
            for(int j = 0; j < mcols2; j++){
                sum = 0.0;
                for(int k = 0; k < mcols; k++){
                    sum += A[i*mcols+k] * B[k*mcols2+j];
                }
                C[i*mcols2+j] = sum;
            }
        }
}

// Matrix multiplication without tiling.
void matrix_mult_standard(vector<double> &A, vector<double> &B, vector<double> &C,
                            int nrows, int mcols, int mcols2){
    double sum;

    // If matrix B is a square matrix, we can transpose.
    if(mcols == mcols2){
        // Transpose matrix B in order to optimize efficiency.
        vector<double>B_T(mcols*mcols2);
        transpose_matrix(A, B_T, mcols, mcols2);

        // Compute the dot product row-wise because of transposed matrix.
        #pragma omp parallel for reduction(+:sum)
        for(int i = 0; i < nrows; i++){
            for(int j = 0; j < mcols2; j++){
                sum = 0.0;
                for(int k = 0; k < mcols; k++){
                    sum += A[i*mcols+k] * B_T[j*mcols2+k];
                }   
                C[i*mcols2+j] = sum;
            }
        }
    }
    else{
        // Compute a regular dot product with no transpose of matricies.
        #pragma omp parallel for reduction(+:sum)
        for(int i = 0; i < nrows; i++){
            for(int j = 0; j < mcols2; j++){
                sum = 0.0;
                for(int k = 0; k < mcols; k++){
                    sum += A[i*mcols+k] * B[k*mcols2+j];
                }
                C[i*mcols2+j] = sum;
            }
        }
    }
}

// Matrix multiplication with tiling implemented.
void matrix_mult_tiling(vector<double> &A, vector<double> &B, vector<double> &C,
                            int nrows, int mcols, int mcols2){
    
    double sum;                         // Maintains the sum of the dotproducts.
    int i_0, j_0, k_0;                  // Tile creation indicies.
    int i, j, k;                        // Matrix multiplication indicies.
    int comp_i, comp_j, comp_k;

    #pragma omp parallel private(i_0, j_0, k_0, i, j, k) 
    {
        // Collapse the tile for loops.
        #pragma omp collapse(3)
        // Creating the tiles
        for(i_0 = 0; i_0 < nrows; i_0+=CACHE_SIZE){
            for(j_0 = 0; j_0 < mcols; j_0+=CACHE_SIZE){
                for(k_0 = 0; k_0 < mcols2; k_0+=CACHE_SIZE){
                    // Compute tile sizes in the last loop so the above loops can be collapsed.
                    if(i_0 + CACHE_SIZE < nrows)
                        comp_i = i_0 + CACHE_SIZE - 1;
                    else
                        comp_i = nrows - 1;
                    
                    if(j_0 + CACHE_SIZE < mcols)
                        comp_j = j_0 + CACHE_SIZE - 1;
                    else
                        comp_j = mcols - 1;
                    
                    if(k_0 + CACHE_SIZE < mcols2)
                        comp_k = k_0 + CACHE_SIZE - 1;
                    else
                        comp_k = mcols2 - 1;

                    // Computes the dot products of matrices A and B, and stores the result in matrix C.
                    #pragma omp for reduction(+:sum)
                    for(i = i_0; i <= comp_i; i++) {
                        // Precompute indicies whene first available to lower computation time.
                        int a_mult = mcols * i;
                        int c_mult = mcols2 * i;
                        for(int k=k_0; k <= comp_k; k++) {
                            int c_ind = c_mult + k;
                            for(int j = j_0; j <= comp_j; j++) {                  
                                sum += A[a_mult+j] * B[mcols2*j+k];
                            }   
                            C[c_ind] = sum;
                            // Reset the sum for next dot product index.
                            sum = 0.0;
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    // Input arguments for matrix A and B dimensions.
    int nrows, mcols, mcols2;

    // Starting and ending time for omp computations.
    double start;
    double end;

    // Dimensions of the computed matrices.
    nrows = atoi(argv[1]);
    mcols = atoi(argv[2]);
    mcols2 = atoi(argv[3]);

    // Declare number of threads to be utilized.
    if(strcasecmp(argv[4], "-t") == 0){
        int nthreads = atoi(argv[5]);
        omp_set_num_threads(nthreads);
    }
    // Initializes vectors of the correct size according to arguments.
    vector<double> A(nrows*mcols);
    vector<double> B(mcols*mcols2);
    vector<double> C(nrows*mcols2);

    // Sets a random seed for the population of values.
    srand(time(NULL));

    cout << "\nDimensions: " << nrows << "x" << mcols <<"x" << mcols2 <<endl;
    // Fills the matrices with random floating point values.
    start = omp_get_wtime();
    fill_matrix(A, nrows, mcols);
    fill_matrix(B, mcols, mcols2);
    end = omp_get_wtime();
    cout << "Time to fill both matrices: " << end - start << " seconds" << endl;
    
    // Default non parallelized mat mult.
    // matrix_mult_example(A, B, C, nrows, mcols, mcols2);

    // Zeroes the result matrix C.
    clear_matrix(C, nrows, mcols2);
    start = omp_get_wtime();
    matrix_mult_standard(A, B, C, nrows, mcols, mcols2);
    end = omp_get_wtime();
    cout << "\n\tTime to perform parallelized matrix multiplication: " << end - start << " seconds" << endl;

    // Zeroes the result matrix C.
    clear_matrix(C, nrows, mcols2);
    start = omp_get_wtime();
    matrix_mult_tiling(A, B, C, nrows, mcols, mcols2);
    end = omp_get_wtime();
    cout << "\n\tTime to perform parallelized tiled matrix multiplication: " << end - start << " seconds" << endl;

    // Prints out entire matrix C.
    // for(int i = 0; i < nrows; i++){
    //     for(int j = 0; j < mcols2; j++){
    //         cout << setprecision(3) << C[i*mcols2 + j] << "\t";
    //     }
    //     cout << endl;
    // }

    return 0;
}