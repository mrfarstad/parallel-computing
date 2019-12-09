#include "libs/bitmap.h"
#include <getopt.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5
// If you apply another kernel, remember not only to exchange
// the kernel but also the kernelFactor and the correct dimension.

int const sobelYKernel[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
float const sobelYKernelFactor = (float)1.0;

int const sobelXKernel[] = {-1, -0, -1, -2, 0, -2, -1, 0, -1, 0};
float const sobelXKernelFactor = (float)1.0;

int const laplacian1Kernel[] = {-1, -4, -1, -4, 20, -4, -1, -4, -1};

float const laplacian1KernelFactor = (float)1.0;

int const laplacian2Kernel[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
float const laplacian2KernelFactor = (float)1.0;

int const laplacian3Kernel[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
float const laplacian3KernelFactor = (float)1.0;

// Bonus Kernel:

int const gaussianKernel[] = {1,  4, 6, 4,  1,  4,  16, 24, 16, 4, 6, 24, 36,
                              24, 6, 4, 16, 24, 16, 4,  1,  4,  6, 4, 1};

float const gaussianKernelFactor = (float)1.0 / 256.0;

// Helper function to swap bmpImageChannel pointers

void swapImageChannel(bmpImageChannel **one, bmpImageChannel **two) {
  bmpImageChannel *helper = *two;
  *two = *one;
  *one = helper;
}

// Apply convolutional kernel on image data
void applyKernel(unsigned char *out, unsigned char *in, unsigned int width,
                 unsigned int height, int *kernel, unsigned int kernelDim,
                 float kernelFactor) {
  unsigned int const kernelCenter = (kernelDim / 2);
  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      int aggregate = 0;
      for (unsigned int ky = 0; ky < kernelDim; ky++) {
        int nky = kernelDim - 1 - ky;
        for (unsigned int kx = 0; kx < kernelDim; kx++) {
          int nkx = kernelDim - 1 - kx;

          int yy = y + (ky - kernelCenter);
          int xx = x + (kx - kernelCenter);
          if (xx >= 0 && xx < (int)width && yy >= 0 && yy < (int)height)
            aggregate += in[yy * width + xx] * kernel[nky * kernelDim + nkx];
        }
      }
      aggregate *= kernelFactor;
      if (aggregate > 0) {
        out[y * width + x] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[y * width + x] = 0;
      }
    }
  }
}

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

  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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
        goto graceful_exit;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind + 1)) {
    help(argv[0], ' ', "Not enough arugments");
    goto error_exit;
  }
  input = calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(input, argv[optind], strlen(argv[optind]));
  optind++;

  output = calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(output, argv[optind], strlen(argv[optind]));
  optind++;

  /*
    End of Parameter parsing!
   */

  /*
    Create the BMP image and load it from disk.
   */
  bmpImage *image = newBmpImage(0, 0);
  bmpImageChannel *imageChannel = NULL;

  if (image == NULL) {
    fprintf(stderr, "Could not allocate new image!\n");
  }

  if (world_rank == 0) {
    if (loadBmpImage(image, input) != 0) {
      fprintf(stderr, "Could not load bmp image '%s'!\n", input);
      freeBmpImage(image);
      goto error_exit;
    }
  }

  MPI_Bcast(&image->width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&image->height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Create a single color channel image. It is easier to work just with one
  // color
  imageChannel = newBmpImageChannel(image->width, image->height);
  if (imageChannel == NULL) {
    fprintf(stderr, "Could not allocate new image channel!\n");
    freeBmpImage(image);
    goto error_exit;
  }

  if (world_rank == 0) {
    // Extract from the loaded image an average over all colors - nothing else
    // than a black and white representation extractImageChannel and
    // mapImageChannel need the images to be in the exact same dimensions! Other
    // prepared extraction functions are extractRed, extractGreen, extractBlue
    if (extractImageChannel(imageChannel, image, extractAverage) != 0) {
      fprintf(stderr, "Could not extract image channel!\n");
      freeBmpImage(image);
      freeBmpImageChannel(imageChannel);
      goto error_exit;
    }
  }

  int *send_counts = malloc(world_size * sizeof(int));
  int *send_displs = malloc(world_size * sizeof(int));

  size_t displ = 0;
  for (int i = 0; i < world_size; i++) {
    size_t size = image->width * (image->height / world_size);
    // Handle pictures with odd amount of rows
    if (i < image->height % world_size) {
      size += image->width;
    }
    send_counts[i] = size;
    send_displs[i] = displ; // i * size; // displ;
    displ += size;
  }

  size_t local_height = send_counts[world_rank] / image->width;

  // Here we do the actual computation!
  // imageChannel->data is a 2-dimensional array of unsigned char which is
  // accessed row first ([y][x])
  //
  bmpImageChannel *processImageChannel =
      newBmpImageChannel(image->width, local_height + 2);

  // Original image -> imageChannel
  // Total working image -> processImageChannel
  // Process working image -> localBuffer
  //
  //
  bmpImageChannel *localImageChannel =
      newBmpImageChannel(image->width, local_height + 2);

  MPI_Scatterv(imageChannel->rawdata, send_counts, send_displs,
               MPI_UNSIGNED_CHAR, &localImageChannel->rawdata[image->width],
               send_counts[world_rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  for (unsigned int i = 0; i < iterations; i++) {

    if (world_rank != world_size - 1) {
      MPI_Send(&localImageChannel->rawdata[image->width * local_height],
               image->width, MPI_UNSIGNED_CHAR, world_rank + 1, 0,
               MPI_COMM_WORLD);
      MPI_Recv(&localImageChannel->rawdata[image->width * (local_height + 1)],
               image->width, MPI_UNSIGNED_CHAR, world_rank + 1, 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (world_rank != 0) {
      MPI_Recv(&localImageChannel->rawdata[0], image->width, MPI_UNSIGNED_CHAR,
               world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&localImageChannel->rawdata[image->width], image->width,
               MPI_UNSIGNED_CHAR, world_rank - 1, 1, MPI_COMM_WORLD);
    }

    applyKernel(
        processImageChannel->rawdata, localImageChannel->rawdata, image->width,
        local_height + 2, (int *)laplacian1Kernel, 3, laplacian1KernelFactor
        //               (int *)laplacian2Kernel, 3, laplacian2KernelFactor
        //               (int *)laplacian3Kernel, 3, laplacian3KernelFactor
        //               (int *)gaussianKernel, 5, gaussianKernelFactor
    );
    swapImageChannel(&processImageChannel, &localImageChannel);
  }
  freeBmpImageChannel(processImageChannel);

  MPI_Gatherv(&localImageChannel->rawdata[image->width],
              send_counts[world_rank], MPI_UNSIGNED_CHAR, imageChannel->rawdata,
              send_counts, send_displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  // Map our single color image back to a normal BMP image with 3 color channels
  // mapEqual puts the color value on all three channels the same way
  // other mapping functions are mapRed, mapGreen, mapBlue
  if (world_rank == 0) {
    // printf("hello\n");
    if (mapImageChannel(image, imageChannel, mapEqual) != 0) {
      fprintf(stderr, "Could not map image channel!\n");
      freeBmpImage(image);
      freeBmpImageChannel(imageChannel);
      goto error_exit;
    }

    // Write the image back to disk
    if (saveBmpImage(image, output) != 0) {
      fprintf(stderr, "Could not save output to '%s'!\n", output);
      freeBmpImage(image);
      goto error_exit;
    };
  }
  freeBmpImageChannel(imageChannel);
  freeBmpImageChannel(localImageChannel);

graceful_exit:
  ret = 0;
error_exit:
  if (input)
    free(input);
  if (output)
    free(output);

  free(send_counts);
  free(send_displs);

  MPI_Finalize();
  return ret;
};
