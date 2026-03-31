__inline__ CUDA_FUNC void saveTile(uchar4* tile, int2 tileShape, uchar4* img, int2 imgShape, int2 tileStartPos){
    int tidBlock = threadIdx.x + blockDim.x*threadIdx.y;
    int threadsPerBlock = blockDim.x * blockDim.y;
    int cacheSize = totalSize(tileShape);

    for(int offset = 0; tidBlock + offset < cacheSize; offset += threadsPerBlock){
        int cacheIdx = tidBlock + offset;
        int2 cachePos = indexToPos(cacheIdx, tileShape);
        int2 imgPos = tileStartPos + cachePos;
        int imgIdx = flattenIndex(imgPos, imgShape);

        if(inRange(imgPos, imgShape))
            tile[cacheIdx] = img[imgIdx];
    }

    __syncthreads();
}

__inline__ CUDA_FUNC void saveTileAligned(uchar4* tile, int2 tileShape, uchar4* img, int2 imgShape, int2 tileStartPos){
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
        int tileIdx = flattenIndex(tilePos, tileShape);
        int2 imgPos = tileStartPos + tilePos;
        int imgIdx = flattenIndex(imgPos, imgShape);

        if(inRange(imgPos, imgShape) && tileStartPos.x <= tilePos.x && tileStartPos.x < endX)
            tile[tileIdx] = img[imgIdx];
    }
    __syncthreads();
}

__inline__ CUDA_FUNC void saveTile1D(uchar4* tile, int tileShape, uchar4* img, int imgShape, int tileStartPos){
    saveTileAligned(tile, {tileShape, 1}, img, {imgShape,1}, {tileStartPos, 1});
}
