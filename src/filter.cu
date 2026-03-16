#include "filter.cuh"

#include "image.h"
#include "utils.h"

#include "extended_math.h"

//#include <math.h>
#include <regex>
#include <iostream>
#include <assert.h>

__constant__ float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};


void atrousFilterCpu(GBuffer frame, const FilterParams params){
    for(int i = 0; i < params.depth; i++){
        atrousFilterPassCpu(
            (i == 0) ? frame.render : frame.buffer[i%2],
            (i == params.depth-1) ? frame.denoised : frame.buffer[(i+1)%2],
            i, frame, params
        );
    }
}

void atrousFilterPassCpu(uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params){
    int2 framePos;
    
    for(framePos.x = 0; framePos.x < frame.shape.x; framePos.x++){
        for(framePos.y = 0; framePos.y < frame.shape.y; framePos.y++){
            float3 acum = {0, 0, 0};
            float norm = 0;
            int2 dPos;
                
            float3 refRender   = make_float3(in[flattenIndex(framePos, frame.shape)]);
            float3 refNormal   = make_float3(frame.normal[flattenIndex(framePos, frame.shape)]);
            float3 refAlbedo   = make_float3(frame.albedo[flattenIndex(framePos, frame.shape)]);

            //printf("%x ", in[flattenIndex(framePos, frame.shape)]);

            for(dPos.x = -params.radius; dPos.x <= params.radius; dPos.x++){
                for(dPos.y = -params.radius; dPos.y <= params.radius; dPos.y++){

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
                    w = 1;
                    acum += w*nRender;
                    norm += w;
                }
            }
            acum /= norm;
            
            out[flattenIndex(framePos, frame.shape)] = make_uchar4(acum);
            out[flattenIndex(framePos, frame.shape)].w = 255;
        }
        //printf("\n");
    }
}

CUDA_CPU_FUNC void atrousFilterPixel(int2 pos, uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params){

}


KERNEL void atrousFilterCudaBase(GBuffer frame, const FilterParams params){
    for(int i = 0; i < params.depth; i++){
        atrousFilterPassCudaBase(
            (i == 0) ? frame.render : frame.buffer[i%2],
            (i == params.depth-1) ? frame.denoised : frame.buffer[(i+1)%2],
            i, frame, params
        );
    }
}

CUDA_FUNC void atrousFilterPassCudaBase(uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params){
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
    
    for(dPos.x = -params.radius; dPos.x <= params.radius; dPos.x++){
        for(dPos.y = -params.radius; dPos.y <= params.radius; dPos.y++){

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
    out[flattenIndex(framePos, frame.shape)] = in[flattenIndex(framePos, frame.shape)];

    __syncthreads();
}