#include "metrics.h"

float mse(uchar4* golden, uchar4* buffer, int2 shape){
    float acum = 0;
    for(int i = 0; i < totalSize(shape); i++)
            acum += (golden[i].x - buffer[i].x)*(golden[i].x - buffer[i].x);
    
    acum /= totalSize(shape);
    return acum;
}