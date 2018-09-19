/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
* This sample implements a separable convolution filter
* of a 2D image with an arbitrary kernel.
*/

#include "Filter.h"


int main(int argc, char **argv)
{
    // start logs
    printf("[%s] - Starting...\n", argv[0]);

    double* image;
    double* result;

    Filter::reserveMemory(image, 16, 16);
    Filter::reserveMemory(result, 16, 16);

    double kernel[5*5] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    Filter::showData(kernel, 5, 5);

    Filter::generateData(image, 16, 16);
    Filter::showData(image, 16, 16);

    Filter filter(kernel);
    filter.convolution(image, result, 16, 16, 1);
    Filter::showData(result, 16, 16);

    Filter::deleteMemory(image, 16, 16);
    Filter::deleteMemory(result, 16, 16);

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
