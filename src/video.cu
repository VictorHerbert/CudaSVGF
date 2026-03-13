#include "video.cuh"

#include <regex>
#include <string>

void videoFilterCpu(std::string filepath, const FilterParams params){
    std::string framePath, outputPath;
    int2 shape = {1920, 1080};

    int frameCount = 10;

    CpuGBuffer buffer;
    buffer.shape = shape;
    CpuVector<uchar4> denoised(totalSize(shape));
    CpuVector<uchar4> bufferVec(2*totalSize(shape));
    buffer.denoised = denoised.data();
    buffer.buffer[0] = bufferVec.data();
    buffer.buffer[1] = bufferVec.data() + totalSize(shape);
    
    for(int i = 1; i < frameCount; i++){
        framePath = std::regex_replace(filepath, std::regex("\\$i\\$"), std::to_string(i));
        
        Image renderImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "render"), 4);
        Image normalImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "normal"), 4);
        Image albImg = Image(std::regex_replace(framePath, std::regex("\\$channel\\$"), "albedo"), 4);

        buffer.render = (uchar4*) renderImg.data;
        buffer.normal = (uchar4*) normalImg.data;
        buffer.albedo = (uchar4*) albImg.data;
        
        atrousFilterCpu(buffer, params);
        
        Image::save(
            std::regex_replace(framePath, std::regex("\\$channel\\$"), "output"), 
            (byte*) buffer.denoised,
            {buffer.shape.x, buffer.shape.y, 4}
        );
    }

    
    
}