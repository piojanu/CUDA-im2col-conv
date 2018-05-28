#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

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
#define TESTON

#ifdef DEBUG
    #define LOG(...) printf(__VA_ARGS__); fflush(stdout);
#else
    #define LOG(...) ;
#endif

const unsigned int H = 512, W = 256, C = 128, K = 3, D = 1; 
const unsigned int BLOCK_SIZE = 256;

// HOST FUNCTION
// Takes matrix A [double *matA] and transforms it
// into column representation [double *matAc]
void im2colOnHost(double *matA, double *matAc, int radiusF, int countLR, int L, int M, int K, int C)
{
    // For each spatial position in output...
    for (int m = 0; m < M; m++) {
        int w = m + radiusF;
        for (int l = 0; l < L; l++) {
            int h = l + radiusF;

            // Progress..
            LOG("\r[i] Calculation on CPU %3d%%...", ((m * L + l) * 100 / (M * L)));

            // For each kernel weight...
            for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                    for (int r = 0; r < C; r++) {
                        matAc[(l + L * m) + countLR * (r + C * (p + K * q))] = matA[r + C * ((h + op) + H * (w + oq))]; 
                        // LOG("matAc[%3d x %3d] <- matA[%3d x %3d x %3d]\n", (r + C * (p + K* q)), (l + L * m), (h + op), (w + oq), r);
                    }
                }
            }
        }
    }
    LOG("\n");
}
 
// DEVICE KERNEL
// Takes matrix A [double *matA] and transforms it
// into column representation [double *matAc] on GPU
__global__ 
void im2colOnDevice(unsigned int n, double *matAc, double *matA, int radiusF, int countLR, int L, int M, int K, int C)
{
    // Using grid-stride loop if too big problem size.
    // https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) 
    {
        int m = (idx / C) / L;
        int l = (idx / C) % L;
        int r = idx % C;
        
        // For each spatial position in output...
        if (m < M) {
            int w = m + radiusF;
            if (l < L) {
                int h = l + radiusF;
                // For each kernel weight...
                for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                    for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                        if (r < C) {
                            matAc[(l + L * m) + countLR * (r + C * (p + K * q))] = matA[r + C * ((h + op) + H * (w + oq))]; 
                        }
                    }
                }
            }
        }
    }
}
 
// DEVICE KERNEL
// Takes matrix A [double *matA] and transforms it
// into column representation [double *matAc] on GPU
__global__ 
void col2imOnDevice(unsigned int n, double *matA, double *matAc, int radiusF, int countLR, int L, int M, int K, int C)
{
    // Using grid-stride loop if too big problem size.
    // https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) 
    {
        int m = (idx / C) / L;
        int l = (idx / C) % L;
        int r = idx % C;
    
        // For each spatial position in output...
        if (m < M) {
            int w = m + radiusF;
            if (l < L) {
                int h = l + radiusF;
                // For each kernel weight...
                for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                    for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                        if (r < C) {
                            matA[r + C * ((h + op) + H * (w + oq))] = matAc[(l + L * m) + countLR * (r + C * (p + K * q))]; 
                        }
                    }
                }
            }
        }
    }
}

int main()
{
    // CONSTS AND VARIABLES

    // For kernel execution time tracking
    clock_t start, end;

    // Input/kernel/output counts and sizes
    const unsigned int countA = H*W*C;
    const size_t sizeA = countA*sizeof(double);
    LOG("[i] INPUT PARAMS: %u height, %u width, %u channels, %u elems, %u bytes\n", H, W, C, countA, sizeA);

    const unsigned int radiusF = (K - 1) / 2;
    const unsigned int countF = K*K*C;
    const unsigned int countFs = D*K*K*C;
    const size_t sizeF = countF*sizeof(double);
    const size_t sizeFs = countFs*sizeof(double);
    LOG("[i] FILTER PARAMS: %u radius, %u elems, %u bytes\n", radiusF, countF, sizeF);
    LOG("[i] FILTERS PARAMS: %u elems, %u bytes\n", countFs, sizeFs);
    
    const unsigned int L = H - (K - 1);
    const unsigned int M = W - (K - 1);
    LOG("[i] OUTPUT PARAMS: %u height, %u width, %u channels\n", L, M, D);
    
    const unsigned int countLR = L * M;
    const unsigned int countAc = countF * countLR;
    const size_t sizeAc = countAc*sizeof(double);
    LOG("[i] INPUT IN COL PARAMS: %u elems, %u bytes\n", countAc, sizeAc);

    
    // PREPARE DATA

    // Generate input data
    double *matA = (double *)malloc(sizeA);
    for (int i = 0; i < countA; i++) {
        matA[i] = i;
    }
    LOG("  [!] FINISHED GENERATING INPUT\n");

    // Calculate im2col result
    double *matAc = (double *)malloc(sizeAc);
    im2colOnHost(matA, matAc, radiusF, countLR, L, M, K, C);
    LOG("  [!] FINISHED CALCULATING im2col RESULT ON CPU\n");


    // Alloc memory and copy data to device
    double *devA, *devAc, *retAc;
    
    cudaMalloc((void**)&devA, sizeA); 
    cudaMalloc((void**)&devAc, sizeAc); 
    retAc = (double *)malloc(sizeAc);

    cudaMemcpy(devA, matA, sizeA, cudaMemcpyHostToDevice); 
    
    // Run im2col computation on device and copy results
    const unsigned int KERNELS_NUM = L * M * C;
    const unsigned int GRID_SIZE = (KERNELS_NUM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    start = clock();
    im2colOnDevice<<<GRID_SIZE, BLOCK_SIZE>>>(KERNELS_NUM, devAc, devA, radiusF, countLR, L, M, K, C);
    end = clock();
    LOG("  [!] FINISHED CALCULATING im2col ON DEVICE in %.3fms\n", ((double)(end - start)) * 1000 / CLOCKS_PER_SEC);
    
    cudaMemcpy(retAc, devAc, sizeAc, cudaMemcpyDeviceToHost);

#ifdef TESTON
    // Compare results
    int success = 1;
    for (int i = 0; i < countAc; i++) {
        if (retAc[i] != matAc[i]) {
            success = 0;
            printf("TEST FAILED: im2col device kernel...\n");
            break;
        }
    }

    if (success) {
        printf("TEST PASSED: im2col device kernel!\n");
    }
#endif

    // Allocate memory for return value
    double *retA;
    retA = (double *)malloc(sizeA);
    cudaMemset(devA, 0, sizeA); 
    
    // Run col2im computation on device and copy results
    start = clock();
    col2imOnDevice<<<GRID_SIZE, BLOCK_SIZE>>>(KERNELS_NUM, devA, devAc, radiusF, countLR, L, M, K, C);
    end = clock();
    LOG("  [!] FINISHED CALCULATING col2im ON DEVICE in %.3fms\n", ((double)(end - start)) * 1000 / CLOCKS_PER_SEC);
    
    cudaMemcpy(retA, devA, sizeA, cudaMemcpyDeviceToHost);

#ifdef TESTON
    // Compare results
    success = 1;
    for (int i = 0; i < countA; i++) {
        if (retA[i] != matA[i]) {
            success = 0;
            printf("TEST FAILED: col2im device kernel...\n");
            break;
        }
    }

    if (success) {
        printf("TEST PASSED: col2im device kernel!\n");
    }
#endif

    // CLEAN UP
    cudaFree(devA);
    cudaFree(devAc);
    
    free(matA);
    free(matAc);
    free(retA);
    free(retAc);
    
    return EXIT_SUCCESS;
}