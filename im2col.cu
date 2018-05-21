#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Feature maps dimensionality descriptions and assumptions:
//             : Height          : Width           : Channels : Number                :
// INPUT   / A | H               | W               | C        | --------------------- |
// KERNELS / F | P = K           | Q = K           | R = C    | D = number of kernels |
// OUTPUT  / B | L = H * (K - 1) | M = W * (K - 1) | N = D    | --------------------- |
// [!] K must be odd number.
// [!] Data layout for INPUT/OUTPUT: C x H x W.
// [!] Data layout for KERNELS: D x R(=C) x P(=K) x Q(=K)

// Turn on/off debug mode
#define DEBUG

#ifdef DEBUG
    #define LOG(...) printf(__VA_ARGS__); fflush(stdout);
#else
    #define LOG(...) ;
#endif

const int H = 128, W = 64, C = 32, K = 3, D = 1; 
 
__global__ 
void im2col(char *A, int *B) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    A[idx] = B[idx];
}
 
int main()
{
    // CONSTS AND VARIABLES

    // Input/kernel/output counts and sizes
    const int countA = H*W*C;
    const size_t sizeA = countA*sizeof(double);
    LOG("[i] INPUT PARAMS: %d height, %d width, %d channels, %d elems, %d bytes\n", H, W, C, countA, sizeA);

    const int radiusF = (K - 1) / 2;
    const int countF = K*K*C;
    const int countFs = D*K*K*C;
    const size_t sizeF = countF*sizeof(double);
    const size_t sizeFs = countFs*sizeof(double);
    LOG("[i] FILTER PARAMS: %d radius, %d elems, %d bytes\n", radiusF, countF, sizeF);
    LOG("[i] FILTERS PARAMS: %d elems, %d bytes\n", countFs, sizeFs);
    
    const int L = H - (K - 1);
    const int M = W - (K - 1);
    LOG("[i] OUTPUT PARAMS: %d height, %d width, %d channels\n", L, M, D);
    
    const int countAc = countF*(L*M);
    const size_t sizeAc = countAc*sizeof(double);
    LOG("[i] INPUT IN COL PARAMS: %d elems, %d bytes\n", countAc, sizeAc);

    
    // PREPARE DATA

    // Generate input data
    double *matA = (double *)malloc(sizeA);
    for (int i = 0; i < countA; i++) {
        matA[i] = i;
    }
    LOG("  [!] FINISHED GENERATING INPUT\n");

    // Calculate im2col result
    double *matAc = (double *)malloc(sizeAc);
    // For each spatial position in output...
    for (int m = 0; m < M; m++) {
        int w = m + radiusF;
        for (int l = 0; l < L; l++) {
            int h = l + radiusF;
            // For each kernel weight...
            for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                    for (int r = 0; r < C; r++) {
                        matAc[(r + C * (p + K* q)) + countF * (l + L * m)] = matA[r + C * ((h + p) + H * (w + q))]; 
                        // LOG("matAc[%3d x %3d] <- matA[%3d x %3d x %3d]\n", (r + C * (p + K* q)), (l + L * m), (h + p), (w + q), r);
                    }
                }
            }
        }
    }
    LOG("  [!] FINISHED CALCULATING im2col RESULT\n");


    // Alloc memory and copy data to device
    // int *devA, *devAc;
    
    // cudaMalloc((void**)&devA, sizeA); 
    // cudaMalloc((void**)&devAc, sizeAc); 

    // cudaMemcpy(devA, matA, sizeA, cudaMemcpyHostToDevice); 
    
    // dim3 dimBlock(blocksize, 1);
    // dim3 dimGrid(1, 1);
    // copy<<<dimGrid, dimBlock>>>(ad, bd);
    // cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost); 
    // cudaFree(ad);
    // cudaFree(bd);
    
    // printf("%s\n", a);

    // CLEAN UP
    free(matA);
    free(matAc);
    
    return EXIT_SUCCESS;
}