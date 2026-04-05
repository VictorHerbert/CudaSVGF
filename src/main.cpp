#include "test.h"

#include "animation.cuh"

#include <cstring>
#include <stdio.h>

int main(int argc, char* argv[]) {    
    if(argc == 2 && strcmp(argv[1], "-t") == 0){
        test();
    }
    else if(argc == 10) {
        int2 shape;
        int depth;
        int frameCount;
        FilterParams params;
        std::string filepath;

        std::cout << "Called with:\n";
        std::cout << "filepath      = " << argv[1] << "\n";
        std::cout << "shape.x       = " << argv[2] << "\n";
        std::cout << "shape.y       = " << argv[3] << "\n";
        std::cout << "depth         = " << argv[4] << "\n";
        std::cout << "frameCount    = " << argv[5] << "\n";
        std::cout << "sigmaSpace    = " << argv[6] << "\n";
        std::cout << "sigmaRender   = " << argv[7] << "\n";
        std::cout << "sigmaAlbedo   = " << argv[8] << "\n";
        std::cout << "sigmaNormal   = " << argv[9] << "\n";
        
        try {
            filepath = argv[1];

            shape.x = std::stoi(argv[2]);
            shape.y = std::stoi(argv[3]);
            depth = std::stoi(argv[4]);
            frameCount = std::stoi(argv[5]);

            params.sigmaSpace   = std::stof(argv[6]);
            params.sigmaRender  = std::stof(argv[7]);
            params.sigmaAlbedo  = std::stof(argv[8]);
            params.sigmaNormal  = std::stof(argv[9]);
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
        }

        try {
            videoFilterCuda(filepath, shape,frameCount, depth, params);
            std::cout << "Output at " << filepath << "DenoisedXXXX.png\n";
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }
    else{
        printf("Not enough arguments. Expected 9, got %d\n", argc-1);
        printf("Usage\n %s filepath shape.x shape.y depth frameCount sigmaSpace sigmaRenger sigmaAlbedo sigmaNormal\n",
            argv[0]
        );

        return 1;
    }

    return 0;
}
