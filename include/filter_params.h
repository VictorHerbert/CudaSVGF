#ifndef FILTER_PARAMS_H
#define FILTER_PARAMS_H

struct FilterParams {
    int depth = 2;
    int radius = 2;

    float sigmaSpace = .1;
    float sigmaColor = .1;
    float sigmaAlbedo = .1;
    float sigmaNormal = .1;
};

#endif