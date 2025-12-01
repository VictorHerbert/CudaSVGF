#include "filter.cuh"

#include "image.h"
#include "utils.h"

#include <math.h>
#include <regex>
#include <iostream>

__constant__ float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};

CUDA_CPU_FUNC int haloSize(int radius, int level){
    return radius*(1<<level);
}

CUDA_CPU_FUNC int tileByteCount(int radius, int level, int2 blockShape){
    int halo = radius * (1<<level);
    return 4*(2*halo + blockShape.x)*(2*halo + blockShape.y);
}

CUDA_CPU_FUNC int tileBytes(FilterParams params, int2 blockShape){
    return ((params.tile & FilterParams::NORMAL) + (params.tile & FilterParams::ALBEDO))*
        totalSize(blockShape)*4;
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
    extern __shared__ uchar4 sharedMem[];
    uchar4* sharedMemPtr = sharedMem;


    int2 framePos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    int2 blockPos   = {blockIdx.x, blockIdx.y};
    int2 blockShape = {blockDim.x, blockDim.y};
    int2 threadPos  = {threadIdx.x, threadIdx.y};

    if(params.tile & FilterParams::NORMAL){
        frame.normalTile = sharedMemPtr;
        frame.normalTile[flattenIndex(threadPos, blockShape)] =
            frame.normal[flattenIndex(framePos, frame.shape)];
        __syncthreads();
        sharedMemPtr += 4*totalSize(blockShape);
    }

    if(params.tile & FilterParams::ALBEDO){
        frame.albedoTile = sharedMemPtr;
        frame.albedoTile[flattenIndex(threadPos, blockShape)] =
            frame.albedo[flattenIndex(framePos, frame.shape)];
        __syncthreads();
        sharedMemPtr += 4*totalSize(blockShape);
    }

    for(int level = 0; level < params.depth; level++){
        uchar4* in = (level == 0) ? frame.render : frame.buffer[level%2];
        uchar4* out = (level == (params.depth - 1)) ? frame.denoised : frame.buffer[(level+1)%2];
        singleLevelFilter(in, out, level, frame, params);
    }
}

KERNEL void filterKernelBase(GBuffer frame, FilterParams params){
    for(int level = 0; level < params.depth; level++){
        uchar4* in = (level == 0) ? frame.render : frame.buffer[level%2];
        uchar4* out = (level == (params.depth - 1)) ? frame.denoised : frame.buffer[(level+1)%2];
        singleLevelFilterBase(in, out, level, frame, params);
    }
}


CUDA_FUNC void singleLevelFilterBase(uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params){
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

    if(framePos.x >= frame.shape.x || framePos.y >= frame.shape.y)
        return;

    float3 acum = {0, 0, 0};
    float norm = 0;
    int2 dPos;

    float3 refRender, refAlbedo, refNormal;

    refRender   = make_float3(in[flattenIndex(framePos, frame.shape)]);
    refNormal   = make_float3(frame.normal[flattenIndex(framePos, frame.shape)]);
    refAlbedo   = make_float3(frame.albedo[flattenIndex(framePos, frame.shape)]);

    for(dPos.x = -params.radius; dPos.x <= params.radius; dPos.x++){
        for(dPos.y = -params.radius; dPos.y <= params.radius; dPos.y++){
            int2 nPosInFrame = framePos + dPos * (1<<level);
            int nMemIdx = flattenIndex(nPosInFrame, frame.shape);

            if(!inRange(nPosInFrame, frame.shape))
                continue;

            float dSum = 0;

            float3 nRender = make_float3(frame.render[flattenIndex(nPosInFrame, frame.shape)]);

            if(params.type & FilterParams::SPATIAL){
                float dSpace = length(make_float2(dPos));
                dSum += dSpace/params.sigmaSpace;
            }
            if(params.type & FilterParams::RENDER){
                float dRender = length(refRender - nRender);
                dSum += dRender/params.sigmaSpace;
            }
            if(params.type & FilterParams::ALBEDO){
                float3 nAlbedo = make_float3(frame.albedo[nMemIdx]);
                float dAlbedo = length(refAlbedo - nAlbedo);
                dSum += dAlbedo/params.sigmaSpace;
            }
            if(params.type & FilterParams::NORMAL){
                float3 nNormal = make_float3(frame.normal[nMemIdx]);
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
    out[flattenIndex(framePos, frame.shape)] = make_uchar4(acum);

    __syncthreads();
}

CUDA_FUNC void singleLevelFilter(uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params){
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

    if(framePos.x >= frame.shape.x || framePos.y >= frame.shape.y)
        return;

    float3 acum = {0, 0, 0};
    float norm = 0;
    int2 dPos;

    float3 refRender, refAlbedo, refNormal;

    refRender   = make_float3(in[flattenIndex(framePos, frame.shape)]);
    refNormal   = make_float3(frame.normal[flattenIndex(framePos, frame.shape)]);
    refAlbedo   = make_float3(frame.albedo[flattenIndex(framePos, frame.shape)]);

    for(dPos.x = -params.radius; dPos.x <= params.radius; dPos.x++){
        for(dPos.y = -params.radius; dPos.y <= params.radius; dPos.y++){
            int2 nPosInFrame = framePos + dPos * (1<<level);
            int2 nPosInTile = threadPos + dPos * (1<<level);
            int nMemIdx = flattenIndex(nPosInFrame, frame.shape);
            int nTileIdx = flattenIndex(nPosInTile, blockShape);

            if(!inRange(nPosInFrame, frame.shape))
                continue;

            float dSum = 0;

            float3 nRender = make_float3(frame.render[flattenIndex(nPosInFrame, frame.shape)]);

            if(params.type & FilterParams::SPATIAL){
                float dSpace = length(make_float2(dPos));
                dSum += dSpace/params.sigmaSpace;
            }
            if(params.type & FilterParams::RENDER){
                float dRender = length(refRender - nRender);
                dSum += dRender/params.sigmaSpace;
            }
            if(params.type & FilterParams::ALBEDO){
                uchar4 mem = (params.tile & FilterParams::ALBEDO & inRange(nPosInTile, blockShape)) ?
                    frame.albedoTile[nTileIdx] :
                    frame.albedo[nMemIdx];

                float3 nAlbedo = make_float3(mem);
                float dAlbedo = length(refAlbedo - nAlbedo);
                dSum += dAlbedo/params.sigmaSpace;
            }
            if(params.type & FilterParams::NORMAL){
                uchar4 mem = (params.tile & FilterParams::NORMAL & inRange(nPosInTile, blockShape)) ?
                    frame.normalTile[nTileIdx] :
                    frame.normal[nMemIdx];

                float3 nNormal = make_float3(mem);
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
    out[flattenIndex(framePos, frame.shape)] = make_uchar4(acum);

    __syncthreads();
}
