#include "video.cuh"

#include <regex>
#include <string>

void videoFilterCpu(
    std::string filepath,
    std::string outputPath,
    int2 shape,
    int frameCount,
    FilterParams params
){        
    CpuGBuffer buffer;
    buffer.shape = shape;
    CpuVector<uchar4> denoised(totalSize(shape));
    CpuVector<uchar4> bufferVec(2*totalSize(shape));
    buffer.denoised = denoised.data();
    buffer.buffer[0] = bufferVec.data();
    buffer.buffer[1] = bufferVec.data() + totalSize(shape);

    for(int i = 1; i < frameCount; i++){
        printf("%d frame\n", i);
        auto framePath = std::regex_replace(filepath, std::regex("\\$i\\$"), std::to_string(i));

        Image renderImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "render"), 4);
        Image normalImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "normal"), 4);
        Image albImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "albedo"), 4);

        buffer.render = (uchar4*) renderImg.data;
        buffer.normal = (uchar4*) normalImg.data;
        buffer.albedo = (uchar4*) albImg.data;

        atrousFilterCpu(buffer, params);

        Image::save(
            std::regex_replace(outputPath, std::regex("\\$i\\$"), std::to_string(i)),
            (byte*) buffer.denoised,
            {buffer.shape.x, buffer.shape.y, 4}
        );
    }
}

void videoFilterCuda(
    std::string filepath,
    std::string outputPath,
    int2 shape,
    int frameCount,
    FilterParams params,
    int maxStreamCount
){
    int size = totalSize(shape);
    int cacheUsage = 0;

    dim3 defaultBlockSize(32,8);
    dim3 defaultGridSize((shape.x + defaultBlockSize.x-1) / defaultBlockSize.x, (shape.y + defaultBlockSize.y-1) / defaultBlockSize.y);
    int2 defaultBlockShape = {defaultBlockSize.x, defaultBlockSize.y};

    int streamCount = min(maxStreamCount, frameCount);

    cudaStream_t streams[MAX_STREAM];
    CudaGBuffer frames[MAX_STREAM];
    CpuVector<uchar4> denoised[MAX_STREAM];

    for(int i = 0; i < streamCount; i++){
        cudaStreamCreate(&streams[i]);
        denoised[i].resize(size);
        frames[i].resize(shape);
    }

    for(int i = 0; i < frameCount + streamCount; i++){
        int streamIdx = i%streamCount;
        cudaStream_t stream = streams[streamIdx];
        Image renderImg, normalImg, albImg;

        if(i < frameCount){
            auto framePath = std::regex_replace(filepath, std::regex("\\$i\\$"), std::to_string(i+1));

            renderImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "render"), 4);
            normalImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "normal"), 4);
            albImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "albedo"), 4);
        }

        if(i >= streamCount){
            auto frameOutputPath = std::regex_replace(outputPath, std::regex("\\$i\\$"), std::to_string(i+1-streamCount));
            frameOutputPath = std::regex_replace(frameOutputPath, std::regex("\\$channel\\$"), "output");

            cudaStreamSynchronize(stream);

            Image::save(
                frameOutputPath,
                (byte*) denoised[streamIdx].data(),
                {shape.x, shape.y, 4}
            );
        }

        if(i < frameCount){
            frames[streamIdx].renderVec.copyFromAsync((uchar4*) renderImg.data, size, stream);
            frames[streamIdx].normalVec.copyFromAsync((uchar4*) normalImg.data, size, stream);
            frames[streamIdx].albedoVec.copyFromAsync((uchar4*) albImg.data, size, stream);

            atrousFilterCudaBase<<<defaultGridSize, defaultBlockSize, cacheUsage, stream>>>(frames[streamIdx], params);
            CUDA_ERROR_CHECK(cudaGetLastError());

            frames[streamIdx].denoisedVec.copyToAsync(denoised[streamIdx].data(), stream);
        }
    }

    for(int i = 0; i < streamCount; i++)
        cudaStreamDestroy(streams[i]);
}