#include "filter.cuh"

#include "math_utils.h"
#include <cuda_runtime.h>
#include <assert.h>

template <typename T>
float variance(int2 pos, const T* in, int2 shape, int level){
    float mean = 0.0f;
    float meanSqr = 0.0f;
    float varCount = 0.0f;

    int2 dPos;

    for(dPos.x = -ATROUS_RADIUS; dPos.x <= ATROUS_RADIUS; dPos.x++){
        for(dPos.y = -ATROUS_RADIUS; dPos.y <= ATROUS_RADIUS; dPos.y++){
            int2 nPos = pos + dPos * (1 << level);
            int nMemIdx = posToIndex(nPos, shape);

            if(!inRange(nPos, shape))
                continue;

            float lum = luminance(make_float3(in[nMemIdx]));

            mean += lum;
            meanSqr += lum * lum;
            varCount += 1.0f;
        }
    }

    mean /= varCount;
    meanSqr /= varCount;

    return meanSqr - mean * mean;
}

template <FilterEngine engine, FilterType type, typename T>
void atrousFilter(GBuffer<T> frame, int depth, FilterParams params, cudaStream_t stream){
    for(int i = 0; i < depth; i++){
        auto curr = getLevelBuffer(i, depth, frame);
        auto next = getLevelBuffer(i+1, depth, frame);

        if constexpr(engine == CUDA){
            dim3 blockShape = DEFAULT_BLOCK_SIZE;
            dim3 gridShape((frame.shape.x + blockShape.x-1) / blockShape.x, (frame.shape.y + blockShape.y-1) / blockShape.y);

            int2 tileShape = {blockShape.x + 2*ATROUS_RADIUS*(1<<depth), blockShape.y + 2*ATROUS_RADIUS*(1<<depth)};
            int tileSize = totalSize(tileShape);
            int tileBytes = tileSize*sizeof(T);
            if constexpr(type == BASE)
                tileBytes = 0;

            assert(tileBytes < 48*1024);

            atrousFilterCudaPass<type><<<gridShape, blockShape, tileBytes, stream>>>(curr, next, i, frame, params);
            //CUDA_KERNEL_ERROR_CHECK();
        }
        else if constexpr (engine == CPU)
            atrousFilterCpuPass<type>(curr, next,i, frame, params);
    }
}

template <FilterType type, typename T>
void atrousFilterCpuPass(const T* in, T* out, int level, GBuffer<T> frame, FilterParams params){
    int2 pos;
    for(pos.x = 0; pos.x < frame.shape.x; pos.x++)
        for(pos.y = 0; pos.y < frame.shape.y; pos.y++)
            atrousFilterPixel<CPU, type, T>(pos, in, out, nullptr, level, frame);

}

template <FilterType type, typename T> KERNEL
void atrousFilterCudaPass(const T* in, T* out, int level, GBuffer<T> frame, FilterParams params){
    extern __shared__ uchar1 shared[];
    int2 pos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};

    atrousFilterPixel<CUDA, type, T>(pos, in, out, (T*) shared, level, frame);
}

template <FilterEngine engine, FilterType type, typename T> CUDA_CPU_FUNC
void atrousFilterPixel(int2 pos, const T* in, T* out, T* tile, int level, GBuffer<T> frame, FilterParams params){
    float3 acum = {0, 0, 0};
    float norm = 0;
    int2 dPos;

    float var;

#ifdef __CUDA_ARCH__
        int2 tileStart = {blockIdx.x * blockDim.x - ATROUS_RADIUS*(1<<level), blockIdx.y * blockDim.y - ATROUS_RADIUS*(1<<level)};
        int2 tileEnd = {(blockIdx.x +1)*blockDim.x + ATROUS_RADIUS*(1<<level), (blockIdx.y +1)* blockDim.y + ATROUS_RADIUS*(1<<level)};

        tileStart = max(tileStart, make_int2(0,0));
        tileEnd = min(tileEnd, make_int2(frame.shape.x, frame.shape.y));

        int2 tileShape = tileEnd - tileStart;

        if constexpr(type != BASE){
            if constexpr (type == TILE)
                saveTile(tile, tileShape, in, frame.shape, tileStart);
            else if constexpr (type == ALIGNED)
                saveTileAligned(tile, tileShape, in, frame.shape, tileStart);

            if(pos.x >= frame.shape.x || pos.y >= frame.shape.y)
                return;
            
                var = variance(pos-tileStart, tile, tileShape, level);
        }
        else {
            if(pos.x >= frame.shape.x || pos.y >= frame.shape.y)
                return;

            var = variance(pos, in, frame.shape, level);
        }

#else
        var = variance(pos, in, frame.shape, level);
#endif

    int memIdx = posToIndex(pos, frame.shape);

    float refLum    = luminance(make_float3(in[memIdx]));
    float3 refNormal   = parseNormal(frame.normal[memIdx]);
    float3 refAlbedo   = make_float3(frame.albedo[memIdx]);

    for(dPos.x = -ATROUS_RADIUS; dPos.x <= ATROUS_RADIUS; dPos.x++){
        for(dPos.y = -ATROUS_RADIUS; dPos.y <= ATROUS_RADIUS; dPos.y++){
            int2 nPos = pos + dPos * (1<<level);
            int nMemIdx = posToIndex(nPos, frame.shape);

            if(!inRange(nPos, frame.shape))
                continue;

            float3 nRender;

#ifdef __CUDA_ARCH__
            int nTileMemIdx = posToIndex(nPos-tileStart, tileShape);

            if constexpr(type != BASE)
                nRender = make_float3(tile[nTileMemIdx]);
            else
                nRender = make_float3(in[nMemIdx]);
#else
            nRender = make_float3(in[nMemIdx]);
#endif
            float dRender = abs(refLum - luminance(nRender));

            float3 nAlbedo = make_float3(frame.albedo[nMemIdx]);
            float dAlbedo = length(refAlbedo - nAlbedo);

            float3 nNormal = parseNormal(frame.normal[nMemIdx]);
            float dNormal = max(dot(refNormal, nNormal), 0.0);

            float expArg =
                -dRender/var
                -min(dAlbedo/params.sigmaAlbedo, 2.0f)
                -min(dNormal/params.sigmaNormal, 2.0f);

            float wCross = exp(expArg);

            float wWavelet = waveletCoef(dPos);

            float w = wWavelet*wCross;

            acum += w*nRender;
            norm += w;
        }
    }

    acum /= norm;

    out[memIdx] = make_type<T>(acum);
}

INLINE CUDA_FUNC
void saveTile(uchar4* tile, int2 tileShape, const uchar4* img, int2 imgShape, int2 tileStartPos){
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


INLINE CUDA_FUNC
void saveTileAligned(uchar4* tile, int2 tileShape, const uchar4* img, int2 imgShape, int2 tileStartPos){
    int tidBlock = threadIdx.x + blockDim.x*threadIdx.y;
    int threadsPerBlock = blockDim.x * blockDim.y;
    int warpPerBlock = threadsPerBlock/32;
    int cacheSize = totalSize(tileShape);
    int2 tileEndPos = tileStartPos + tileShape;

    int tIdInWarp = tidBlock%32;
    int warpIdx = tidBlock/32;
    int startX = tileStartPos.x & ~31;
    int endX = (tileEndPos.x + 31) & ~31;
    int warpsPerLine = (endX - startX)/32;
    int warpsPerTile = warpsPerLine*tileShape.y;

    for(int warpOffset = warpIdx; warpOffset < warpsPerTile; warpOffset += warpPerBlock){
        int2 tilePos = {warpOffset%warpsPerLine + tIdInWarp, warpOffset/warpsPerLine};
        int tileIdx = posToIndex(tilePos, tileShape);
        int2 imgPos = tileStartPos + tilePos;
        int imgIdx = posToIndex(imgPos, imgShape);

        if(inRange(imgPos, imgShape) && tileStartPos.x <= tilePos.x && tileStartPos.x < endX)
            tile[tileIdx] = img[imgIdx];
    }
    __syncthreads();
}


INSTANTIATE_ATROUS_CPU_CFG(uchar3);
INSTANTIATE_ATROUS_CUDA_CFG(BASE, uchar3);
INSTANTIATE_ATROUS_CUDA_CFG(BASE, uchar4);
INSTANTIATE_ATROUS_CUDA_CFG(TILE, uchar4);
INSTANTIATE_ATROUS_CUDA_CFG(ALIGNED, uchar4);