#include "utils.h"
#include "image.h"
#include "gbuffer.h"

float mse(uchar4* golden, uchar4* buffer, int size);
float snr(uchar4* golden, uchar4* buffer, int shape);
float psnr(uchar4* golden, uchar4* buffer, int shape);