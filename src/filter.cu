#include "filter.cuh"

#include "image.h"
#include "utils.h"

#include <math.h>
#include <regex>
#include <iostream>

__constant__ float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};

__device__ void printMemInfo(uchar4* ptr, int2 grid = {0,0}) {
    unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = threadId % 32;

    uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    unsigned long long int wordAddress = static_cast<unsigned int>(address >> 2);
    unsigned long long int bank = wordAddress % 32;

    if(blockIdx.x == grid.x && blockIdx.y == grid.y){
        printf("Tid %3u | Lane %2u | Addr 0x%llx | WordAddr 0x%x | Bank %2u\n",
           threadId,
           lane,
           static_cast<unsigned long long>(address),
           wordAddress,
           bank);
    }
}

CUDA_CPU_FUNC int tileByteCount(int radius, int level, int2 blockShape){
    int halo = radius * (1<<level);
    return 4*(2*halo + blockShape.x)*(2*halo + blockShape.y);
}

CUDA_FUNC void cacheTile(uchar4* tile, const uchar4* in, const int2 frameShape, const int2 start, const int2 end){
    int2 gridPos    = { blockIdx.x, blockIdx.y };
    int2 blockShape = { blockDim.x, blockDim.y };
    int2 blockPos   = { threadIdx.x, threadIdx.y };
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int laneId = threadId%32;
    int warpId = threadId/32;

    int2 tileShape = end - start;
    int startLine = start.x/32;
    int linesPerRow = (end.x + 1)/32 - startLine;
    int warpsPerBlock = totalSize(blockShape)/32;
    int tileLineCount = linesPerRow * (end.y - start.y);


    /*for (int idx = threadId; idx < totalTileSize; idx += blockShape.x * blockShape.y) {
        int2 tilePos = { idx % tileShape.x, idx / tileSize.x };

        int2 framePos = gridPos * blockShape + tilePos - halo;

        if (inRange(framePos, shape)){
            int frameIdx = flattenIndex(framePos, shape);
            tile[idx] = in[frameIdx];
        }
    }*/

    for(int warpOffset = 0; warpOffset < tileLineCount; warpOffset += warpsPerBlock){
        int nWarpId = warpId + warpOffset;
        int2 posInFrame = {
            startLine*32 + 32*(warpId%linesPerRow) + laneId,
            nWarpId/linesPerRow + start.y
        };

        int2 posInTile = posInFrame - start;

        reinterpret_cast<int&>(tile[flattenIndex(posInTile, tileShape)]) =
            reinterpret_cast<const int&>(in[flattenIndex(posInFrame, frameShape)]);
    }

    __syncthreads();
}

KERNEL void filterKernel(GBuffer frame, FilterParams params){
    extern __shared__ uchar4 renderTile[];

    for(int level = 0; level < params.depth; level++){
        uchar4* in = (level == 0) ? frame.render : frame.buffer[level%2];
        uchar4* out = (level == (params.depth - 1)) ? frame.denoised : frame.buffer[(level+1)%2];
        singleLevelFilter(in, out, level, renderTile, frame, params);
    }
}

CUDA_FUNC void singleLevelFilter(uchar4* in, uchar4* out, int level, uchar4* tile, const GBuffer frame, const FilterParams params){
    int2 posInFrame = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    int2 gridPos    = {blockIdx.x, blockIdx.y};
    int2 blockShape = {blockDim.x, blockDim.y};
    int2 blockPos   = {threadIdx.x, threadIdx.y};
    int2 halo       = {params.radius, params.radius};

    /*if(params.cacheTile){
        int2 startPos = gridPos * blockShape;
        int2 endPos = startPos + blockShape;
        cacheTile(tile, in, frame.shape, startPos, endPos);
    }*/

    if(posInFrame.x >= frame.shape.x || posInFrame.y >= frame.shape.y)
        return;

    float3 acum = {0, 0, 0};
    float norm = 0;
    int2 dPos;

    float3 refRender = make_float3(in[flattenIndex(posInFrame, frame.shape)]);

    for(dPos.x = -params.radius; dPos.x <= params.radius; dPos.x++){
        for(dPos.y = -params.radius; dPos.y <= params.radius; dPos.y++){
            int2 nPosInFrame = posInFrame + dPos * (1<<level);

            if(!inRange(nPosInFrame, frame.shape))
                continue;
            
            float dSum = 0;
            float3 nRender = make_float3(in[flattenIndex(nPosInFrame, frame.shape)]);

            if(params.type & FilterParams::SPATIAL){
                float dSpace = length(make_float2(dPos));
                dSum += dSpace/params.sigmaSpace;
            }
            if(params.type & FilterParams::RENDER){
                float dRender = length(refRender - nRender);
                dSum += dRender/params.sigmaSpace;
            }
            if(params.type & FilterParams::ALBEDO){
                float3 refAlbedo = make_float3(frame.albedo[flattenIndex(posInFrame, frame.shape)]);
                float3 nAlbedo = make_float3(frame.albedo[flattenIndex(nPosInFrame, frame.shape)]);
                float dAlbedo = length(refAlbedo - nAlbedo);
                dSum += dAlbedo/params.sigmaSpace;
            }
            if(params.type & FilterParams::NORMAL){
                float3 refNormal = make_float3(frame.normal[flattenIndex(posInFrame, frame.shape)]);
                float3 nNormal = make_float3(frame.normal[flattenIndex(nPosInFrame, frame.shape)]);
                float dAlbedo = length(refNormal - nNormal);
                dSum += dAlbedo/params.sigmaSpace;
            }

            float w = exp(-dSum);
            acum += w*nRender;
            norm += w;
        }
    }
    acum /= norm;    

    // TODO make sure it load an int instead of 4x load char
    out[flattenIndex(posInFrame, frame.shape)] = make_uchar4(acum);

    __syncthreads();
}