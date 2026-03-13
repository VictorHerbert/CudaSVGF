#include "test.h"

#include <cstring>
#include <stdio.h>

int main(int argc, char* argv[]) {
    if(strcmp(argv[1], "-t") == 0){
        test(".*");
    }
    else {
        printf("Usage:\n\t%s test_pattern\n\n", argv[0]);
        printf("Example:\n\t%s filter_*\n\n", argv[0]);
        printf("Available tests:\n", argv[0]);
        printTestFuncs();
    }
    return 0;
}
