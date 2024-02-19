
#include <stdio.h>

__global__ void hellofromgpu(){
   printf("hello world from gpu!\n");
}

int main(){
   printf("hello wolrd from cpu!\n");

   hellofromgpu<<<1, 10>>>();
   cudaDeviceReset();

   return 0;
}



/usr/local/cuda/bin/nvcc (-arch sm_82) hello.cu -o hello
