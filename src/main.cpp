#include "test.h"

#include "primitives.h"
#include "video.cuh"

#include <cstring>
#include <stdio.h>

int main(int argc, char* argv[]) {
    if(strcmp(argv[1], "-t") == 0){
        test();
    }
    else {
        int2 shape;
        int depth;
        int frameCount;
        FilterParams params;
        std::string filepath;
        
        filepath = argv[1];

        shape.x = std::stoi(argv[2]);
        shape.y = std::stoi(argv[3]);
        depth = std::stoi(argv[4]);
        frameCount = std::stoi(argv[5]);

        params.sigmaSpace   = std::stof(argv[6]);
        params.sigmaRender  = std::stof(argv[7]);
        params.sigmaAlbedo  = std::stof(argv[8]);
        params.sigmaNormal  = std::stof(argv[9]);

        videoFilterCuda(filepath, shape, depth, frameCount, params);
    }
    return 0;
}
