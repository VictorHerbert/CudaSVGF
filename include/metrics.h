#include "utils.h"
#include "image.h"
#include "gbuffer.h"

float mse(uchar4* golden, uchar4* buffer, int2 shape);
float mseCuda(uchar4* golden, uchar4* buffer, int2 shape);
//__global__ float mseCuda(uchar4* golden, uchar4* buffer, int2 shape);

float psnr(uchar4* golden, uchar4* buffer, int2 shape);
float psnrCuda(uchar4* golden, uchar4* buffer, int2 shape);
//__global__ float psnrKernel(uchar4* golden, uchar4* buffer, int2 shape);

float snr(uchar4* golden, uchar4* buffer, int2 shape);
float snrCuda(uchar4* golden, uchar4* buffer, int2 shape);
//__global__ float snrKernel(uchar4* golden, uchar4* buffer, int2 shape);