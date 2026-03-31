#include "metrics.h"

#include <cuda_runtime.h>

float mse(uchar4* golden, uchar4* buffer, int size){
    float acum = 0;
    for(int i = 0; i < size; i++){
        acum += length2(golden[i] - buffer[i]);
    }

    acum /= size;

    return acum;
}

float snr(uchar4* golden, uchar4* buffer, int size){
    float num = 0;
    float denom = 0;

    for(int i = 0; i < size; i++){
        num += length2(golden[i]);
        denom += length2(golden[i] - buffer[i]);
    }     

    return num/denom;
}

float psnr(uchar4* golden, uchar4* buffer, int size){
    return (float) log10(255*255/mse(golden, buffer, size));
}