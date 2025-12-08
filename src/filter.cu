#include "filter.cuh"

#include "image.h"
#include "utils.h"

#include "extended_math.h"

//#include <math.h>
#include <regex>
#include <iostream>
#include <assert.h>



__constant__ float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};

#define CHANNEL_COUNT 3

template<typename... Args>
__device__ void printfSingle(const char * str, Args... args) {
    if(blockIdx.x == 0 && blockIdx.y == 0)
        if(threadIdx.x == 0 && threadIdx.y == 0)
            printf(str, args...);
}

template<typename... Args>
__device__ void printfBlock(const char * str, Args... args) {
    if(blockIdx.x == 0 && blockIdx.y == 1)        
        printf(str, args...);
}

template<typename... Args>
__device__ void printfThread(const char * str, Args... args) {
    if(threadIdx.x == 0 && threadIdx.y == 0)        
        printf(str, args...);
}

CUDA_CPU_FUNC int haloSize(int radius, int level){
    return radius*(1<<level);
}

CUDA_CPU_FUNC int lineTileSize(int radius, int blockDimX, int level){
    return blockDimX + 2*haloSize(radius, level);
}

CUDA_CPU_FUNC int lineTileArea(int radius, int2 blockDim, int level){
    return blockDim.y * lineTileSize(radius, blockDim.x, level);
}

CUDA_CPU_FUNC int lineTileBytes(int radius, int2 blockDim, int level){
    return 4*CHANNEL_COUNT*lineTileArea(radius, blockDim, level);
}

CUDA_FUNC void dilatedFilterBase(uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params){
    int2 framePos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    int2 blockPos   = {blockIdx.x, blockIdx.y};
    int2 blockShape = {blockDim.x, blockDim.y};
    int2 threadPos  = {threadIdx.x, threadIdx.y};

    if(framePos.x >= frame.shape.x || framePos.y >= frame.shape.y)
        return;

    float3 acum = {0, 0, 0};
    float norm = 0;
    int2 dPos;

    float3 refRender, refAlbedo, refNormal;

    refRender   = make_float3(in[flattenIndex(framePos, frame.shape)]);
    refNormal   = make_float3(frame.normal[flattenIndex(framePos, frame.shape)]);
    refAlbedo   = make_float3(frame.albedo[flattenIndex(framePos, frame.shape)]);

    for(dPos.y = -params.radius; dPos.y <= params.radius; dPos.y++){
        for(dPos.x = -params.radius; dPos.x <= params.radius; dPos.x++){

            int2 nFramePos = framePos + dPos * (1<<level);
            int nMemIdx = flattenIndex(nFramePos, frame.shape);

            if(!inRange(nFramePos, frame.shape))
                continue;

            float dSum = 0;

            float3 nRender = make_float3(frame.render[flattenIndex(nFramePos, frame.shape)]);

            float dSpace = length(make_float2(dPos));
            dSum += dSpace/params.sigmaSpace;

            float dRender = length(refRender - nRender);
            dSum += dRender/params.sigmaSpace;

            float3 nAlbedo = make_float3(frame.albedo[nMemIdx]);
            float dAlbedo = length(refAlbedo - nAlbedo);
            dSum += dAlbedo/params.sigmaSpace;

            float3 nNormal = make_float3(frame.normal[nMemIdx]);
            float dNormal = length(refNormal - nNormal);
            dSum += dNormal/params.sigmaSpace;

            float w = exp(-dSum);
            acum += w*nRender;
            norm += w;
        }
    }
    acum /= norm;

    // TODO make sure it load an int instead of 4x load char
    out[flattenIndex(framePos, frame.shape)] = make_uchar4(acum);

    __syncthreads();
}

CUDA_FUNC void dilatedFilter2(uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params){
    int2 framePos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    int2 blockPos   = {blockIdx.x, blockIdx.y};
    int2 blockShape = {blockDim.x, blockDim.y};
    int2 threadPos  = {threadIdx.x, threadIdx.y};

    if(framePos.x >= frame.shape.x || framePos.y >= frame.shape.y)
        return;

    float3 acum = {0, 0, 0};
    float norm = 0;
    int2 dPos;

    float3 refRender, refAlbedo, refNormal;

    refAlbedo   = make_float3(frame.albedo[flattenIndex(framePos, frame.shape)]);

    float dSum = 0;

    for(dPos.y = -params.radius; dPos.y <= params.radius; dPos.y++){
        for(dPos.x = -params.radius; dPos.x <= params.radius; dPos.x++){

            int2 nFramePos = framePos + dPos * (1<<level);
            int nMemIdx = flattenIndex(nFramePos, frame.shape);

            if(!inRange(nFramePos, frame.shape))
                continue;

            float3 nAlbedo = make_float3(frame.albedo[nMemIdx]);
            float dAlbedo = length(refAlbedo - nAlbedo);
            dSum += dAlbedo/params.sigmaSpace;
        }
    }

    refNormal   = make_float3(frame.normal[flattenIndex(framePos, frame.shape)]);

    for(dPos.y = -params.radius; dPos.y <= params.radius; dPos.y++){
        for(dPos.x = -params.radius; dPos.x <= params.radius; dPos.x++){

            int2 nFramePos = framePos + dPos * (1<<level);
            int nMemIdx = flattenIndex(nFramePos, frame.shape);

            if(!inRange(nFramePos, frame.shape))
                continue;

            float3 nNormal = make_float3(frame.normal[nMemIdx]);
            float dNormal = length(refNormal - nNormal);
            dSum += dNormal/params.sigmaSpace;
        }
    }

    refRender   = make_float3(in[flattenIndex(framePos, frame.shape)]);

    for(dPos.y = -params.radius; dPos.y <= params.radius; dPos.y++){
        for(dPos.x = -params.radius; dPos.x <= params.radius; dPos.x++){

            int2 nFramePos = framePos + dPos * (1<<level);
            int nMemIdx = flattenIndex(nFramePos, frame.shape);

            if(!inRange(nFramePos, frame.shape))
                continue;

            float3 nRender = make_float3(in[flattenIndex(nFramePos, frame.shape)]);

            float dSpace = length(make_float2(dPos));
            dSum += dSpace/params.sigmaSpace;

            float dRender = length(refRender - nRender);
            dSum += dRender/params.sigmaSpace;

            float w = exp(-dSum);
            acum += w*nRender;
            norm += w;
        }
    }


    acum /= norm;

    // TODO make sure it load an int instead of 4x load char
    out[flattenIndex(framePos, frame.shape)] = make_uchar4(acum);

    __syncthreads();
}

CUDA_FUNC int tileLine(uchar4* tile, uchar4* input, int size){
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.y*blockDim.x;

    for(int threadOffset = 0; threadOffset < size; threadOffset += threadsPerBlock){
        if(threadId + threadOffset < size){
            //printf("Tid %3d | Wid %2d | Input %p | Idx %3d | ADDR %p\n", threadId, threadId%32, input, threadId + threadOffset, &input[threadId + threadOffset]);
            tile[threadId + threadOffset] = input[threadId + threadOffset];
        }
    }
    __syncthreads();
}

CUDA_FUNC int tileLine(uchar4* tile, uchar4* start, uchar4* end){
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.y*blockDim.x;

    for(int threadOffset = 0; &start[threadOffset] < end; threadOffset += threadsPerBlock){
        if(&start[threadId + threadOffset] < end){
            //printf("Tid %3d | Wid %2d | Input %p | Idx %3d | ADDR %p\n", threadId, threadId%32, input, threadId + threadOffset, &input[threadId + threadOffset]);
            tile[threadId + threadOffset] = start[threadId + threadOffset];
        }
    }
    __syncthreads();
}

CUDA_FUNC int tileLine(uchar4* tile, uchar4* input, int2 shape, int radius, int line, int level){
    assert(inRange(line, shape.y));

    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.y*blockDim.x;

    int halo = radius * (1<<level);

    int tileStart = blockIdx.x * blockDim.x - halo;
    int tileEnd = (blockIdx.x+1) * blockDim.x + halo;

    uchar4 *start = input + tileStart + line * shape.x;
    uchar4 *end = input + tileEnd + line * shape.x;
    
    for(int threadOffset = 0; start + threadOffset < end; threadOffset += threadsPerBlock){
        if((start + threadId + threadOffset < end) && (tileStart >= 0) && (tileEnd < shape.x)){
            /*printfBlock(
                "Tid %3d | Wid %2d | start %p | Idx %3d | ADDR %p\n",
                threadId, threadId%32,
                start, threadId + threadOffset, &start[threadId + threadOffset]);
            */
            tile[threadId + threadOffset] = start[threadId + threadOffset];
        }
    }
    __syncthreads();
}

CUDA_FUNC void dilatedFilterLineTile(uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params){
    assert(blockDim.y == 1);

    int2 framePos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    int2 blockPos   = {blockIdx.x, blockIdx.y};
    int2 blockShape = {blockDim.x, blockDim.y};
    int2 threadPos  = {threadIdx.x, threadIdx.y};

    int halo        = haloSize(params.radius, level);
    int2 haloShape  = {halo, halo};
    int2 tileStartPos   = blockPos * blockShape - haloShape;
    int2 tileEndPos     = (blockPos + make_int2(1,1)) * blockShape + haloShape;

    float3 acum = {0, 0, 0};
    float norm = 0;
    int2 dPos;

    float3 refRender, refAlbedo, refNormal;

    if(inRange(framePos, frame.shape)){
        //int mem;
        //mem = reinterpret_cast<int&>(in[flattenIndex(framePos, frame.shape)]);
        //refRender   = make_float3(reinterpret_cast<uchar4&>(mem));
        refRender   = make_float3(in[flattenIndex(framePos, frame.shape)]);
        refNormal   = make_float3(frame.normal[flattenIndex(framePos, frame.shape)]);
        refAlbedo   = make_float3(frame.albedo[flattenIndex(framePos, frame.shape)]);
    }

    for(dPos.y = -params.radius; dPos.y <= params.radius; dPos.y++){
        __syncthreads();

        int line = framePos.y + dPos.y;

        if(inRange(line, frame.shape.y)){
            tileLine(frame.renderTile, in,              frame.shape, params.radius, line, level);
            tileLine(frame.albedoTile, frame.albedo,    frame.shape, params.radius, line, level);
            tileLine(frame.normalTile, frame.normal,    frame.shape, params.radius, line, level);
        }


        for(dPos.x = -params.radius; dPos.x <= params.radius; dPos.x++){
            int2 nFramePos = framePos + dPos * (1<<level);
            int nMemIdx = flattenIndex(nFramePos, frame.shape);

            if(!inRange(nFramePos, frame.shape))
                continue;

            if(framePos.x >= frame.shape.x || framePos.y >= frame.shape.y)
                continue;

            float dSum = 0;

            float3 nRender = make_float3(frame.renderTile[threadIdx.x]);

            float dSpace = length(make_float2(dPos));
            dSum += dSpace/params.sigmaSpace;

            float dRender = length(refRender - nRender);
            dSum += dRender/params.sigmaSpace;

            float3 nAlbedo = make_float3(frame.albedoTile[threadIdx.x]);
            float dAlbedo = length(refAlbedo - nAlbedo);
            dSum += dAlbedo/params.sigmaSpace;

            float3 nNormal = make_float3(frame.normalTile[threadIdx.x]);
            float dNormal = length(refNormal - nNormal);
            dSum += dNormal/params.sigmaSpace;

            float w = exp(-dSum);
            acum += w*nRender;
            norm += w;
        }
    }
    acum /= norm;

    // TODO make sure it load an int instead of 4x load char
    // Make sure threads get the value before
    if(inRange(framePos, frame.shape)){
        reinterpret_cast<int&>(out[flattenIndex(framePos, frame.shape)]) = 
            *reinterpret_cast<const int*>(&make_uchar4(acum));

        //out[flattenIndex(framePos, frame.shape)] = make_uchar4(acum);
    }
    //else printf("%d %d\n", framePos);

    
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    /*printfBlock("Tid %3d | Wid %2d | Input %p | Idx %3d | ADDR %p\n",
        threadId,
        threadId%32,
        out,
        flattenIndex(framePos, frame.shape),
        &out[flattenIndex(framePos, frame.shape)]);*/

    //printfSingle("in %p | out %p | render %p | albedo %p | normal %p\n", in, out, frame.render, frame.albedo, frame.normal);
}