#include "filter.cuh"

#include "image.h"
#include "utils.h"

#include "extended_math.h"

#include <regex>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>

CUDA_CPU_FUNC void atrousFilterPixelU4(int2 pos, const uchar4* in, uchar4* out, int level, GBuffer<uchar4> frame, FilterParams params){
    const float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};

    float3 acum = {0, 0, 0};
    float norm = 0;
    int2 dPos;

    int memIdx = flattenIndex(pos, frame.shape);

    float3 refRender   = make_float3(in[memIdx]);
    float3 refNormal   = make_float3(frame.normal[memIdx]);
    float3 refAlbedo   = make_float3(frame.albedo[memIdx]);

    for(dPos.x = -ATROUS_RADIUS; dPos.x <= ATROUS_RADIUS; dPos.x++){
        for(dPos.y = -ATROUS_RADIUS; dPos.y <= ATROUS_RADIUS; dPos.y++){

            int2 nFramePos = pos + dPos * (1<<level);
            int nMemIdx = flattenIndex(nFramePos, frame.shape);

            if(!inRange(nFramePos, frame.shape))
                continue;

            float3 nRender = make_float3(frame.render[flattenIndex(nFramePos, frame.shape)]);
            float dRender = length(refRender - nRender);
            float wRender = exp(-dRender/params.sigmaRender);

            float3 nAlbedo = make_float3(frame.albedo[nMemIdx]);
            float dAlbedo = length(refAlbedo - nAlbedo);
            float wAlbedo = exp(-dAlbedo/params.sigmaAlbedo);

            float3 nNormal = make_float3(frame.normal[nMemIdx]);
            float dNormal = dot(refNormal, nNormal);
            float wNormal = exp(-dNormal/params.sigmaNormal);

            float dSpace = length(make_float2(dPos));
            float wSpace = exp(-dSpace/params.sigmaSpace);

            float wWavelet = waveletSpline[abs(dPos.x)]*waveletSpline[abs(dPos.y)];

            float w = wWavelet*wRender*wAlbedo*wNormal*wSpace;
            acum += w*nRender;
            norm += w;
        }
    }
    acum /= norm;

    out[flattenIndex(pos, frame.shape)] = make_type<uchar4>(acum);
}


KERNEL void atrousFilterCudaKernelU4(const uchar4* in, uchar4* out, int level, GBuffer<uchar4> frame, FilterParams params){
    int2 framePos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};

    if(framePos.x >= frame.shape.x || framePos.y >= frame.shape.y)
        return;

    atrousFilterPixelU4(framePos, in, out, level, frame, params);
}

void atrousFilterCudaU4(GBuffer<uchar4> frame, int depth, FilterParams params, cudaStream_t stream){
    for(int i = 0; i < depth; i++){
        dim3 blockShape;
        dim3 gridShape((frame.shape.x + blockShape.x-1) / blockShape.x, (frame.shape.y + blockShape.y-1) / blockShape.y);

        atrousFilterCudaKernelU4<<<gridShape, blockShape, 0, stream>>>(
            (i == 0) ? frame.render : frame.buffer[i%2],
            (i == depth-1) ? frame.denoised : frame.buffer[(i+1)%2],
            i, frame, params
        );
    }
}
