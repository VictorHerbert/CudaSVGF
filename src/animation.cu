#include "animation.cuh"

#include <regex>
#include <string>


#include <sstream>
#include <iomanip>
#include <string>

void animationFilterCpu(
    std::string filepath,
    std::string outputPath,
    int2 shape,
    int depth,
    int frameCount,
    FilterParams params
){        
    CpuGBuffer<uchar4> buffer;
    buffer.shape = shape;
    CpuVector<uchar4> denoised(totalSize(shape));
    CpuVector<uchar4> bufferVec(2*totalSize(shape));
    buffer.denoised = denoised.data();
    buffer.buffer = bufferVec.data();

    for(int i = 1; i < frameCount; i++){
        auto framePath = std::regex_replace(filepath, std::regex("\\$i\\$"), std::to_string(i));

        Image renderImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "render"), 3);
        Image normalImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "normal"), 3);
        Image albImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "albedo"), 3);

        buffer.render = (uchar4*) renderImg.data;
        buffer.normal = (uchar4*) normalImg.data;
        buffer.albedo = (uchar4*) albImg.data;


        //atrousFilterCpu(buffer, depth, params);

        Image::save(
            std::regex_replace(outputPath, std::regex("\\$i\\$"), std::to_string(i)),
            (byte*) buffer.denoised,
            {buffer.shape.x, buffer.shape.y, 4}
        );
    }
}

std::string paddedNumber(int number, int pad){
    std::ostringstream oss;
    oss << std::setw(pad) << std::setfill('0') << number;
    return oss.str();
}

void animationFilterCuda(
    std::string filepath,
    int2 shape,
    int frameCount,
    int depth,
    FilterParams params,
    int maxStreamCount
){
    int size = totalSize(shape);
    int streamCount = min(maxStreamCount, frameCount);

    CpuVector<cudaStream_t> streams(streamCount);
    CpuVector<cudaEvent_t> events(streamCount);
    
    CpuVector<CudaGBuffer<uchar4>> frames(streamCount);
    CpuVector<CpuVector<uchar4>> denoised(streamCount);

    cudaStream_t stream_write;
    cudaStreamCreate(&stream_write);

    for(int i = 0; i < streamCount; i++){
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        denoised[i].resize(size);
        frames[i].resize(shape);
    }

    for(int i = 0; i < frameCount + streamCount; i++){
        int streamIdx = i%streamCount;
        cudaStream_t stream = streams[streamIdx];
        cudaEvent_t event = events[streamIdx];
        Image renderImg, normalImg, albImg;

        if(i < frameCount){
            renderImg = Image(filepath + "Render" + paddedNumber(i+1, 4) + ".png", 4);
            normalImg = Image(filepath + "Normal" + paddedNumber(i+1, 4) + ".png", 4);
            albImg    = Image(filepath + "Albedo" + paddedNumber(i+1, 4) + ".png", 4);

            frames[streamIdx].renderVec.copyFromAsync((uchar4*) renderImg.data, size, stream);
            frames[streamIdx].normalVec.copyFromAsync((uchar4*) normalImg.data, size, stream);
            frames[streamIdx].albedoVec.copyFromAsync((uchar4*) albImg.data, size, stream);

            atrousFilter<CUDA, BASE>(frames[streamIdx], depth, params, stream);
        }

        if(i >= streamCount){
            cudaEventSynchronize(event);

            Image::save(
                filepath + "Denoised" + paddedNumber(i+1-streamCount, 4) + ".png",
                (byte*) denoised[streamIdx].data(),
                {shape.x, shape.y, 4}
            );
        }

        if(i < frameCount){
            frames[streamIdx].denoisedVec.copyToAsync(denoised[streamIdx].data(), stream);
            cudaEventRecord(event, stream);
        }
    }

    for(int i = 0; i < streamCount; i++){
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }
}