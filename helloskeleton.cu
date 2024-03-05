#include <stdio.h>

__global__ void hellofromgpu(int *in, int *out){
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

   out[idx] = in[idx];
   printf("[%d]: in:%d, out:%d\n", idx, in[idx], out[idx]);
}


int main(){
   int m;
   int dev=0;
   cudaDeviceProp deviceProp;

   int *h_idata;   // host(CPU) memory for input
   int *d_idata;   // device(GPU) memory for input

   int *h_odata;   // host(CPU) memory for output
   int *d_odata;   // device(GPU) memory for output

   cudaGetDeviceProperties(&deviceProp, dev);
   printf("starting at device %d: %s\n", dev, deviceProp.name);

   cudaSetDevice(dev);

   h_idata = (int*)malloc(100*sizeof(int));   // allocate memory for input (CPU)
   cudaMalloc((void **)&d_idata, 100*sizeof(int));   // allocate memory for input (GPU)

   h_odata = (int*)malloc(100*sizeof(int));   // allocate memory for output (CPU)
   cudaMalloc((void **)&d_odata, 100*sizeof(int));   // allocate memory for output (GPU)

   for(m=0; m<100; m++){
      h_idata[m] = m;
   }

   cudaMemcpy(d_idata, h_idata, 100*sizeof(int), cudaMemcpyHostToDevice);
   cudaDeviceSynchronize();

   hellofromgpu<<<1, 100>>>(d_idata, d_odata);

   cudaMemcpy(h_odata, d_odata, 100*sizeof(int), cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();

   free(h_idata);
   cudaFree(d_idata);

   free(h_odata);
   cudaFree(d_odata);

   cudaDeviceReset();

   return EXIT_SUCCESS;
}
