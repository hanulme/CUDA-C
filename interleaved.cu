
#include <stdio.h>
#include <sys/time.h>

double seconds(){
   struct timeval tp;
   gettimeofday(&tp, NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n) { 
 // set thread ID
 unsigned int tid = threadIdx.x;
 unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
 // convert global data pointer to the local pointer of this block
 int *idata = g_idata + blockIdx.x * blockDim.x;
 // boundary check
 if(idx >= n) return;
 // in-place reduction in global memory
 for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (tid < stride) {
   idata[tid] += idata[tid + stride];
  }
 __syncthreads();
 }
 // write result for this block to global mem
 if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv) {
 // set up device
 int dev = 0;
 cudaDeviceProp deviceProp;
 cudaGetDeviceProperties(&deviceProp, dev);
 printf("%s starting reduction at ", argv[0]);
 printf("device %d: %s ", dev, deviceProp.name);
 cudaSetDevice(dev);
 bool bResult = false;
 // initialization
 int size = 1<<24; // total number of elements to reduce
 printf(" with array size %d ", size);
 // execution configuration
 int blocksize = 512; // initial block size
  
 if(argc > 1) {
  blocksize = atoi(argv[1]); // block size from command line argument
 }
  
 dim3 block (blocksize,1);
 dim3 grid ((size+block.x-1)/block.x,1);
 printf("grid %d block %d\n",grid.x, block.x);
 // allocate host memory
 size_t bytes = size * sizeof(int);
 int *h_idata = (int *) malloc(bytes);
 int *h_odata = (int *) malloc(grid.x*sizeof(int));
 int *tmp = (int *) malloc(bytes);
  
 // initialize the array
 for (int i = 0; i < size; i++) {
  // mask off high 2 bytes to force max number to 255
  h_idata[i] = (int)(rand() & 0xFF);
 }
 memcpy (tmp, h_idata, bytes);
 size_t iStart,iElaps;
 int gpu_sum = 0;
 // allocate device memory
 int *d_idata = NULL;
 int *d_odata = NULL;
 cudaMalloc((void **) &d_idata, bytes);
 cudaMalloc((void **) &d_odata, grid.x*sizeof(int));
  
 gpu_sum = 0;
 for (int i=0; i<grid.x; i++) gpu_sum += h_odata[i];
 printf("gpu Warmup elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
 iElaps,gpu_sum,grid.x,block.x);
  
 // kernel 1: reduceNeighbored
 cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
 cudaDeviceSynchronize();
  
 iStart = seconds ();
 reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
 cudaDeviceSynchronize();
 iElaps = seconds () - iStart;
  
 cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
  
 gpu_sum = 0;
 for (int i=0; i<grid.x; i++) gpu_sum += h_odata[i];
  
 printf("gpu Neighbored elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
 iElaps,gpu_sum,grid.x,block.x);
 cudaDeviceSynchronize();
  
 iElaps = seconds() - iStart;
 cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost);
 gpu_sum = 0;
 for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
  
 printf("gpu Cmptnroll elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
 iElaps,gpu_sum,grid.x/8,block.x);
  
 /// free host memory
 free(h_idata);
 free(h_odata);
 // free device memory
 cudaFree(d_idata);
 cudaFree(d_odata);
 // reset device
 cudaDeviceReset();
 return EXIT_SUCCESS;
}
