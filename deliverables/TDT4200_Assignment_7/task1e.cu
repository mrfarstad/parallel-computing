#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
#include "libs/bitmap.h"
}
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define ERROR_EXIT -1

#define cudaErrorCheck(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %s %d\n", cudaGetErrorName(code),
            cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

#define BLOCKX 32
#define BLOCKY 32

// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5
// If you apply another filter, remember not only to exchange
// the filter but also the filterFactor and the correct dimension.

// int const sobelYFilter[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
// float const sobelYFilterFactor = (float)1.0;
//
// int const sobelXFilter[] = {-1, -0, -1, -2, 0, -2, -1, 0, -1, 0};
// float const sobelXFilterFactor = (float)1.0;

int const laplacian1Filter[] = {-1, -4, -1, -4, 20, -4, -1, -4, -1};
int const laplacian1FilterDim = 3;
float const laplacian1FilterFactor = (float)1.0;

// int const laplacian2Filter[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
// float const laplacian2FilterFactor = (float)1.0;
//
// int const laplacian3Filter[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
// float const laplacian3FilterFactor = (float)1.0;
//
//// Bonus Filter:
//
// int const gaussianFilter[] = {1,  4, 6, 4,  1,  4,  16, 24, 16, 4, 6, 24, 36,
//                              24, 6, 4, 16, 24, 16, 4,  1,  4,  6, 4, 1};
//
// float const gaussianFilterFactor = (float)1.0 / 256.0;

/********** SUBTASK 1e: Implement kernel using shared memory **********/
__global__ void deviceSharedMemoryApplyFilter(unsigned char *out,
                                              unsigned char *in,
                                              unsigned int width,
                                              unsigned int height, int *filter,
                                              unsigned int filterDim,
                                              float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);

  // Global pixel indices
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary checks to make sure reads and writes are inside allocated memory
  if (x < width && y < height) {

    // Blockwise shared memory
    __shared__ unsigned char workingData[BLOCKX][BLOCKY];

    // Copy data from global memory to shared memory
    workingData[threadIdx.y][threadIdx.x] = in[y * width + x];

    // Sync all threads within block before executing using shared memory
    __syncthreads();

    int aggregate = 0;
    for (unsigned int ky = 0; ky < filterDim; ky++) {
      int nky = filterDim - 1 - ky;
      for (unsigned int kx = 0; kx < filterDim; kx++) {
        int nkx = filterDim - 1 - kx;

        // Indices within block for applying filter
        int yy = threadIdx.y + (ky - filterCenter);
        int xx = threadIdx.x + (kx - filterCenter);

        // If indices within block, then access shared memory
        if (xx >= 0 && xx < BLOCKX && yy >= 0 && yy < BLOCKY)
          aggregate += workingData[yy][xx] * filter[nky * filterDim + nkx];

        // If indices outside the block (indices accessing the halo), then
        // access global memory instead
        else {
          // Change to global indices for applying filter on halo data
          yy = y + (ky - filterCenter);
          xx = x + (kx - filterCenter);

          // Boundary checks to make sure reads are inside allocated memory
          if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
            aggregate += in[yy * width + xx] * filter[nky * filterDim + nkx];
          }
        }
      }
    }
    aggregate *= filterFactor;
    if (aggregate > 0) {
      out[y * width + x] = (aggregate > 255) ? 255 : aggregate;
    } else {
      out[y * width + x] = 0;
    }
  }
}
/********** SUBTASK 1e END **********/

/*
  This is another implementation using only shared memory in the computation.
  The previous solution accesses the global memory for halo data, which should
  be avoided. Inspired by this idea, I made another implementation where threads
  read halo data into the shared memory as well. This comes at the expensive of
  divergence because of extensive conditional branching.
  This solution should only work for filters with 3 dimensions, because it only
  adds a single halo border. It could be extended for 5 dimensions, but since
  the filter is stated as optional I choose not to implement it.
*/
//__global__ void deviceSharedMemoryApplyFilter(unsigned char *out,
//                                              unsigned char *in,
//                                              unsigned int width,
//                                              unsigned int height, int
//                                              *filter, unsigned int filterDim,
//                                              float filterFactor) {
//  unsigned int const filterCenter = (filterDim / 2);
//
//  int threadRow = threadIdx.y;
//  int threadCol = threadIdx.x;
//
//  int blockRow = blockIdx.y;
//  int blockCol = blockIdx.x;
//
//  int globalRow = blockRow * BLOCKY + threadRow;
//  int globalCol = blockCol * BLOCKX + threadCol;
//
//  if (globalCol < width && globalRow < height) {
//
//    // Shared memory is per block, so must be BLOCKY x BLOCKX size plus halo
//    // passing
//    __shared__ unsigned char workingData[BLOCKY + 2][BLOCKX + 2];
//
//    // Pad indices to fit expanded working data with halo support
//    int row = threadRow + 1;
//    int col = threadCol + 1;
//
//    // Check if index is on block border, but not global border
//    bool isBottomRow = threadRow == 0 && blockRow > 0;
//    bool isTopRow = threadRow == (BLOCKY - 1) && globalRow + 1 < height;
//    bool isLeftColumn = threadCol == 0 && blockCol > 0;
//    bool isRightColumn = threadCol == (BLOCKX - 1) && globalCol + 1 < width;
//
//    /*
//      Each thread reads a pixel from global memory into shared memory. If the
//      thread is located on the border of the block, then it will read
//      additional value(s) into shared memory. Three extra values if it's in a
//      corner and one if it's just on the border, but not in a corner. This
//      creates a single halo border around the block, which will work for all
//      filters of 3 dimensions.
//    */
//    // Bottom row
//    if (isBottomRow) {
//      workingData[row - 1][col] = in[(globalRow - 1) * width + globalCol];
//      // Bottom left
//      if (isLeftColumn) {
//        workingData[row - 1][col - 1] =
//            in[(globalRow - 1) * width + (globalCol - 1)];
//      }
//      // Bottom right
//      else if (isRightColumn) {
//        workingData[row - 1][col + 1] =
//            in[(globalRow - 1) * width + (globalCol + 1)];
//      }
//
//    }
//    // Top row
//    else if (isTopRow) {
//      workingData[row + 1][col] = in[(globalRow + 1) * width + globalCol];
//      // Top left
//      if (isLeftColumn) {
//        workingData[row + 1][col - 1] =
//            in[(globalRow + 1) * width + (globalCol - 1)];
//      }
//      // Top right
//      else if (isRightColumn) {
//        workingData[row + 1][col + 1] =
//            in[(globalRow + 1) * width + (globalCol + 1)];
//      }
//    }
//
//    // Left column
//    if (isLeftColumn) {
//      workingData[row][col - 1] = in[globalRow * width + (globalCol - 1)];
//    }
//    // Right column
//    else if (isRightColumn) {
//      workingData[row][col + 1] = in[globalRow * width + (globalCol + 1)];
//    }
//
//    // Must include offset for padded workingData
//    workingData[row][col] = in[globalRow * width + globalCol];
//
//    // Important: Synchronize all threads before starting the computation
//    __syncthreads();
//
//    int aggregate = 0;
//    for (unsigned int ky = 0; ky < filterDim; ky++) {
//      int nky = filterDim - 1 - ky;
//      for (unsigned int kx = 0; kx < filterDim; kx++) {
//        int nkx = filterDim - 1 - kx;
//
//        int yy = row + (ky - filterCenter);
//        int xx = col + (kx - filterCenter);
//
//        int realyy = blockRow * BLOCKY + yy;
//        int realxx = blockCol * BLOCKX + xx;
//
//        /*
//          Aggregate if the pixel is on the block border, but not on the grid
//          border
//        */
//        if ((xx > 0 || xx == 0 && blockCol > 0) && xx < (int)BLOCKX + 2 &&
//            (yy > 0 || yy == 0 && blockRow > 0) && yy < (int)BLOCKY + 2 &&
//            realyy <= height && realxx <= width) {
//          aggregate += workingData[yy][xx] * filter[nky * filterDim + nkx];
//        }
//      }
//    }
//    aggregate *= filterFactor;
//    if (aggregate > 0) {
//      out[globalRow * width + globalCol] = (aggregate > 255) ? 255 :
//      aggregate;
//    } else {
//      out[globalRow * width + globalCol] = 0;
//    }
//  }
//}

void help(char const *exec, char const opt, char const *optarg) {
  FILE *out = stdout;
  if (opt != 0) {
    out = stderr;
    if (optarg) {
      fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
    } else {
      fprintf(out, "Invalid parameter - %c\n", opt);
    }
  }
  fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
  fprintf(out, "\n");
  fprintf(out, "Options:\n");
  fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

  fprintf(out, "\n");
  fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

int main(int argc, char **argv) {
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  int ret = 0;

  static struct option const long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}};

  static char const *short_options = "hi:";
  {
    char *endptr;
    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options,
                            &option_index)) != -1) {
      switch (c) {
      case 'h':
        help(argv[0], 0, NULL);
        return 0;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          return ERROR_EXIT;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind + 1)) {
    help(argv[0], ' ', "Not enough arugments");
    return ERROR_EXIT;
  }
  input = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(input, argv[optind], strlen(argv[optind]));
  optind++;

  output = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(output, argv[optind], strlen(argv[optind]));
  optind++;

  /*
    End of Parameter parsing!
   */

  /*
    Create the BMP image and load it from disk.
   */
  bmpImage *image = newBmpImage(0, 0);
  if (image == NULL) {
    fprintf(stderr, "Could not allocate new image!\n");
  }

  if (loadBmpImage(image, input) != 0) {
    fprintf(stderr, "Could not load bmp image '%s'!\n", input);
    freeBmpImage(image);
    return ERROR_EXIT;
  }

  // Create a single color channel image. It is easier to work just with one
  // color
  bmpImageChannel *imageChannel =
      newBmpImageChannel(image->width, image->height);
  if (imageChannel == NULL) {
    fprintf(stderr, "Could not allocate new image channel!\n");
    freeBmpImage(image);
    return ERROR_EXIT;
  }

  // Extract from the loaded image an average over all colors - nothing else
  // than a black and white representation extractImageChannel and
  // mapImageChannel need the images to be in the exact same dimensions! Other
  // prepared extraction functions are extractRed, extractGreen, extractBlue
  if (extractImageChannel(imageChannel, image, extractAverage) != 0) {
    fprintf(stderr, "Could not extract image channel!\n");
    freeBmpImage(image);
    freeBmpImageChannel(imageChannel);
    return ERROR_EXIT;
  }

  // Here we do the actual computation!
  // imageChannel->data is a 2-dimensional array of unsigned char which is
  // accessed row first ([y][x])
  bmpImageChannel *processImageChannel =
      newBmpImageChannel(imageChannel->width, imageChannel->height);

  /********** SUBTASK 1a: Allocate memory **********/
  /*
    "You must allocate enough space for both the ﬁnal image and the
    temporary working image on the GPU, and copy the data from the CPU."
  */
  unsigned char *finalData;
  unsigned char *workingData;
  size_t size =
      imageChannel->width * imageChannel->height * sizeof(unsigned char);

  int *deviceFilter;
  int *hostFilter = (int *)laplacian1Filter;
  size_t filterSize = laplacian1FilterDim * laplacian1FilterDim * sizeof(int);

  cudaErrorCheck(cudaMalloc((void **)&deviceFilter, filterSize));
  cudaErrorCheck(
      cudaMemcpy(deviceFilter, hostFilter, filterSize, cudaMemcpyHostToDevice));

  cudaErrorCheck(cudaMalloc((void **)&workingData, size));
  cudaErrorCheck(cudaMalloc((void **)&finalData, size));
  cudaErrorCheck(cudaMemcpy(finalData, imageChannel->rawdata, size,
                            cudaMemcpyHostToDevice));
  /********** SUBTASK 1a END **********/

  for (unsigned int i = 0; i < iterations; i++) {

    /********** SUBTASK 1c: Launch your kernel **********/
    dim3 blockDim(BLOCKX, BLOCKY);
    dim3 gridDim(imageChannel->width / BLOCKX, imageChannel->height / BLOCKY);

    // Shared memory version
    deviceSharedMemoryApplyFilter<<<gridDim, blockDim>>>(
        workingData, finalData, imageChannel->width, imageChannel->height,
        deviceFilter, 3, laplacian1FilterFactor);

    // Swap the data pointers
    unsigned char *tmp = workingData;
    workingData = finalData;
    finalData = tmp;
  }
  /********** SUBTASK 1c: END **********/

  // Check for errors in kernel launch
  cudaErrorCheck(cudaGetLastError());

  // Copy the results back from the GPU and check for errors
  cudaErrorCheck(cudaMemcpy(imageChannel->rawdata, finalData, size,
                            cudaMemcpyDeviceToHost));

  freeBmpImageChannel(processImageChannel);

  // Free the allocated memory on device
  cudaErrorCheck(cudaFree(deviceFilter));
  cudaErrorCheck(cudaFree(finalData));
  cudaErrorCheck(cudaFree(workingData));

  // Map our single color image back to a normal BMP image with 3 color
  // channels mapEqual puts the color value on all three channels the same way
  // other mapping functions are mapRed, mapGreen, mapBlue
  if (mapImageChannel(image, imageChannel, mapEqual) != 0) {
    fprintf(stderr, "Could not map image channel!\n");
    freeBmpImage(image);
    freeBmpImageChannel(imageChannel);
    return ERROR_EXIT;
  }
  freeBmpImageChannel(imageChannel);

  // Write the image back to disk
  if (saveBmpImage(image, output) != 0) {
    fprintf(stderr, "Could not save output to '%s'!\n", output);
    freeBmpImage(image);
    return ERROR_EXIT;
  };

  ret = 0;
  if (input)
    free(input);
  if (output)
    free(output);
  return ret;
};
