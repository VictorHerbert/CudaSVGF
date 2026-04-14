#include "filter.cuh"

#include "image.h"
#include "cuda_utils.h"

#include "math_utils.h"

#include <regex>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>

__inline__ CUDA_FUNC void saveTile(uchar4* tile, int2 tileShape, const uchar4* img, int2 imgShape, int2 tileStartPos){
    int tidBlock = threadIdx.x + blockDim.x*threadIdx.y;
    int threadsPerBlock = blockDim.x * blockDim.y;
    int cacheSize = totalSize(tileShape);

    for(int offset = 0; tidBlock + offset < cacheSize; offset += threadsPerBlock){
        int cacheIdx = tidBlock + offset;
        int2 cachePos = indexToPos(cacheIdx, tileShape);
        int2 imgPos = tileStartPos + cachePos;
        int imgIdx = posToIndex(imgPos, imgShape);

        if(inRange(imgPos, imgShape))
            tile[cacheIdx] = img[imgIdx];
    }

    __syncthreads();
}

void atrousFilterCudaTile(GBuffer<uchar4> frame, int depth, FilterParams params, cudaStream_t stream){
    for(int i = 0; i < depth; i++){
        dim3 blockShape;
        dim3 gridShape((frame.shape.x + blockShape.x-1) / blockShape.x, (frame.shape.y + blockShape.y-1) / blockShape.y);

        atrousFilterCudaKernelTile<<<gridShape, blockShape, 0, stream>>>(
            (i == 0) ? frame.render : frame.buffer[i%2],
            (i == depth-1) ? frame.denoised : frame.buffer[(i+1)%2],
            i, frame, params
        );
    }
}

KERNEL void atrousFilterCudaKernelTile(const uchar4* in, uchar4* out, int level, GBuffer<uchar4> frame, FilterParams params){
    int2 framePos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
    extern __shared__ uchar4 sharedMem[];

    atrousFilterPixelTile(framePos, sharedMem, in, out, level, frame, params);
}

CUDA_FUNC void atrousFilterPixelTile(int2 pos, uchar4* tile, const uchar4* in, uchar4* out, int level, GBuffer<uchar4> frame, FilterParams params){
    const float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};

    float3 acum = {0, 0, 0};
    float norm = 0;
    int2 dPos;

    int memIdx = posToIndex(pos, frame.shape);

    float d[ATROUS_DIM][ATROUS_DIM];
        
    int2 tileShape = {blockDim.x + 2*ATROUS_RADIUS, blockDim.y + 2*ATROUS_RADIUS};
    int2 tileStart = {blockIdx.x * blockDim.x - ATROUS_RADIUS*(1<<level), blockIdx.y * blockDim.y - ATROUS_RADIUS*(1<<level)};

    float3 refAlbedo   = make_float3(frame.albedo[memIdx]);

    saveTile(tile, tileShape, frame.albedo, frame.shape, tileStart);

    for(dPos.x = -ATROUS_RADIUS; dPos.x <= ATROUS_RADIUS; dPos.x++){
        for(dPos.y = -ATROUS_RADIUS; dPos.y <= ATROUS_RADIUS; dPos.y++){
            int2 nFramePos = pos + dPos * (1<<level);
            int nMemIdx = posToIndex(nFramePos - tileStart, tileShape);

            if(!inRange(nFramePos, frame.shape))
                continue;

            float3 nAlbedo = make_float3(tile[nMemIdx]);
            float dAlbedo = length(refAlbedo - nAlbedo);
            d[dPos.x + ATROUS_RADIUS][dPos.y + ATROUS_RADIUS] = dAlbedo/params.sigmaAlbedo;
        }
    }

    float3 refNormal   = parseNormal(frame.normal[memIdx]);
    saveTile(tile, tileShape, frame.normal, frame.shape, tileStart);

    for(dPos.x = -ATROUS_RADIUS; dPos.x <= ATROUS_RADIUS; dPos.x++){
        for(dPos.y = -ATROUS_RADIUS; dPos.y <= ATROUS_RADIUS; dPos.y++){
            int2 nFramePos = pos + dPos * (1<<level);
            int nMemIdx = posToIndex(nFramePos - tileStart, tileShape);

            if(!inRange(nFramePos, frame.shape))
                continue;

            float3 nNormal = parseNormal(tile[nMemIdx]);
            float dNormal = 1-dot(refNormal, nNormal);
            d[dPos.x + ATROUS_RADIUS][dPos.y + ATROUS_RADIUS] += dNormal/params.sigmaNormal;
        }
    }

    float3 refRender   = make_float3(in[memIdx]);
    saveTile(tile, tileShape, in, frame.shape, tileStart);

    for(dPos.x = -ATROUS_RADIUS; dPos.x <= ATROUS_RADIUS; dPos.x++){
        for(dPos.y = -ATROUS_RADIUS; dPos.y <= ATROUS_RADIUS; dPos.y++){
            int2 nFramePos = pos + dPos * (1<<level);
            int nMemIdx = posToIndex(nFramePos - tileStart, tileShape);

            if(!inRange(nFramePos, frame.shape))
                continue;

            float dSum = d[dPos.x + ATROUS_RADIUS][dPos.y + ATROUS_RADIUS];

            float3 nRender = make_float3(tile[nMemIdx]);
            float dRender = length(refRender - nRender);
            float dSpace = length(make_float2(dPos));

            dSum += dRender/params.sigmaRender+dSpace/params.sigmaSpace;

            float wCross = exp(-dSum);

            float wWavelet = waveletSpline[abs(dPos.x)]*waveletSpline[abs(dPos.y)];

            float w = wWavelet*wCross;
            acum += w*nRender;
            norm += w;
        }
    }

    if(pos.x >= frame.shape.x || pos.y >= frame.shape.y) return;

    acum /= norm;
    out[posToIndex(pos, frame.shape)] = make_type<uchar4>(acum);
}