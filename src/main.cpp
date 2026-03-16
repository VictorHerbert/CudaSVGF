#include "test.h"

#include <cstring>
#include <stdio.h>

int main(int argc, char* argv[]) {
    if(strcmp(argv[1], "-t") == 0){
        test();
    }
    else {
        printf("Not Implemented");
    }
    return 0;
}
