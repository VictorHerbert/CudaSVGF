#include "filter.cuh"

#include "math_utils.h"
#include <cuda_runtime.h>


void atrousFilterCudaAprox(GBuffer<uchar4> frame, int depth, FilterParams params, cudaStream_t stream){
    for(int i = 0; i < depth; i++){
        dim3 blockShape = DEFAULT_BLOCK_SIZE;
        dim3 gridShape((frame.shape.x + blockShape.x-1) / blockShape.x, (frame.shape.y + blockShape.y-1) / blockShape.y);

        atrousFilterCudaKernelAprox<<<gridShape, blockShape, 0, stream>>>(
            getLevelBuffer(i, depth, frame),
            getLevelBuffer(i+1, depth, frame),
            i, frame, params
        );
    }
}

KERNEL void atrousFilterCudaKernelAprox(const uchar4* in, uchar4* out, int level, GBuffer<uchar4> frame, FilterParams params){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int2 framePos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};

    if(framePos.x >= frame.shape.x || framePos.y >= frame.shape.y)
        return;

    float3 acum = {0, 0, 0};

    float norm = 0;
    int2 dPos;

    int memIdx = posToIndex(framePos, frame.shape);

    float var = variance(framePos, in, frame.shape, level);

    float refLum   = luminance(make_float3(in[memIdx]));
    float3 refNormal   = parseNormal(frame.normal[memIdx]);
    float3 refAlbedo   = make_float3(frame.albedo[memIdx]);

    for(dPos.x = -ATROUS_RADIUS; dPos.x <= ATROUS_RADIUS; dPos.x++){
        for(dPos.y = -ATROUS_RADIUS; dPos.y <= ATROUS_RADIUS; dPos.y++){
            int2 nFramePos = framePos + dPos * (1<<level);
            int nMemIdx = posToIndex(nFramePos, frame.shape);

            if(!inRange(nFramePos, frame.shape))
                continue;
            
            float3 nRender = make_float3(in[nMemIdx]);
            float dRender = abs(refLum - luminance(nRender));

            float3 nAlbedo = make_float3(frame.albedo[nMemIdx]);
            float dAlbedo = length(refAlbedo - nAlbedo);

            float3 nNormal = parseNormal(frame.normal[nMemIdx]);
            float dNormal = max(dot(refNormal, nNormal), 0.0);

            float wCross = __expf(
                -dRender/var
                -min(dAlbedo/(441.672956*params.sigmaAlbedo), 2.0)
                -min(dNormal/params.sigmaNormal, 2.0)
            );

            float wWavelet = waveletCoef(dPos);

            float w = wWavelet*wCross;
            w = 1;
            
            acum += w*nRender;
            norm += w;
        }
    }
    acum /= norm;

    out[memIdx] = make_uchar4(acum);
}