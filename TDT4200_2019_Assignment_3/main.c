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
void applyKernel(unsigned char **out, unsigned char **in, unsigned int width,
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
            aggregate += in[yy][xx] * kernel[nky * kernelDim + nkx];
        }
      }
      aggregate *= kernelFactor;
      if (aggregate > 0) {
        out[y][x] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[y][x] = 0;
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
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  int ret = 0;

  bmpImage *image = NULL;
  bmpImage *subImage = NULL;
  int image_width;
  int image_height;
  // pixel **data = NULL;

  int rows_per_process;
  int rows_extra;
  int sendcount;

  if (world_rank == 0) {

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
    image = newBmpImage(0, 0);
    if (image == NULL) {
      fprintf(stderr, "Could not allocate new image!\n");
    }

    if (loadBmpImage(image, input) != 0) {
      fprintf(stderr, "Could not load bmp image '%s'!\n", input);
      freeBmpImage(image);
      goto error_exit;
    }

    rows_per_process = image->height / world_size;
    rows_extra = image->height % world_size;
    image_width = image->width;
    image_height = image->height;
    sendcount = rows_per_process * image->width * sizeof(pixel);

    // data = image->data;
  }

  int blocklengths[1] = {3};
  MPI_Aint displacements[1] = {0};
  MPI_Datatype types[1] = {MPI_UNSIGNED_CHAR};
  MPI_Datatype mpi_pixel;
  MPI_Datatype mpi_pixel_vector;

  MPI_Type_create_struct(1, blocklengths, displacements, types, &mpi_pixel);
  MPI_Type_commit(&mpi_pixel);

  MPI_Aint extent;
  MPI_Type_extent(mpi_pixel, &extent);

  MPI_Type_vector(image_width, 1, extent, mpi_pixel, &mpi_pixel_vector);
  MPI_Type_commit(&mpi_pixel_vector);

  MPI_Bcast(&rows_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows_extra, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sendcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&image_width, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int *sendcounts = NULL;
  int *displs = NULL;

  bmpImageChannel *imageChannel = NULL;

  printf("sendcount: %d\n", sendcount);
  printf("rows_per_process: %d\n", rows_per_process);
  printf("rows_extra: %d\n", rows_extra);
  printf("image_width: %d\n", image_width);
  printf("image_height: %d\n", image_height);

  /*
    image->data list of rows
    image->data[i] list of pixels

    We want to scatter these rows among the processes
    Base solution: convert into 1-dimensional array and convert back
    Advanced solution: create vector types and distribute them
  */

  pixel *tmp = malloc(image_height * image_width * sizeof(pixel));
  if (world_rank == 0) {
    for (int i = 0; i < image_height; i++) {
      for (int j = 0; j < image_width; j += 1) {
        tmp[i * image_width + j] = image->data[i][j];
      }
    }
  }

  pixel *subTmp = malloc(rows_per_process * image_width * sizeof(pixel));

  MPI_Scatter(tmp, rows_per_process * image_width, mpi_pixel, subTmp,
              rows_per_process * image_width, mpi_pixel, 0, MPI_COMM_WORLD);

  pixel **newData = malloc(rows_per_process * sizeof(pixel *));
  for (int i = 0; i < rows_per_process; i++) {
    newData[i] = malloc(image_width * sizeof(pixel));
  }

  for (int i = 0; i < rows_per_process; i++) {
    for (int j = 0; j < image_width; j++) {
      newData[i][j] = subTmp[i * image_width + j];
    }
  }

  bmpImage newImage = {image_width, rows_per_process, newData};

  // Create a single color channel image. It is easier to work just with one
  // color
  imageChannel = newBmpImageChannel(image_width, rows_per_process);
  if (imageChannel == NULL) {
    fprintf(stderr, "Could not allocate new image channel!\n");
    freeBmpImage(image);
    goto error_exit;
  }

  // Extract from the loaded image an average over all colors - nothing else
  // than
  // a black and white representation
  // extractImageChannel and mapImageChannel need the images to be in the exact
  // same dimensions!
  // Other prepared extraction functions are extractRed, extractGreen,
  // extractBlue
  if (extractImageChannel(imageChannel, &newImage, extractAverage) != 0) {
    fprintf(stderr, "Could not extract image channel!\n");
    freeBmpImage(image);
    freeBmpImageChannel(imageChannel);
    goto error_exit;
  }

  // Here we do the actual computation!
  //   imageChannel->data is a 2-dimensional array of unsigned char which is
  //    accessed row first ([y][x])
  bmpImageChannel *processImageChannel =
      newBmpImageChannel(imageChannel->width, imageChannel->height);
  for (unsigned int i = 0; i < iterations; i++) {
    applyKernel(processImageChannel->data, imageChannel->data,
                imageChannel->width, imageChannel->height,
                (int *)laplacian1Kernel, 3, laplacian1KernelFactor
                //               (int *)laplacian2Kernel, 3,
                // laplacian2KernelFactor
                //               (int *)laplacian3Kernel, 3,
                // laplacian3KernelFactor
                //               (int *)gaussianKernel, 5,
                // gaussianKernelFactor
    );
    swapImageChannel(&processImageChannel, &imageChannel);
  }
  freeBmpImageChannel(processImageChannel);

  // Map our single color image back to a normal BMP image with 3 color channels
  // mapEqual puts the color value on all three channels the same way
  // other mapping functions are mapRed, mapGreen, mapBlue
  if (mapImageChannel(&newImage, imageChannel, mapEqual) != 0) {
    fprintf(stderr, "Could not map image channel!\n");
    freeBmpImage(&newImage);
    freeBmpImageChannel(imageChannel);
    goto error_exit;
  }
  freeBmpImageChannel(imageChannel);

  // MPI_Scatter(tmp, rows_per_process * image_width, mpi_pixel, subTmp,
  //             rows_per_process * image_width, mpi_pixel, 0, MPI_COMM_WORLD);

  pixel *res_tmp = malloc(rows_per_process * image_width * sizeof(pixel));
  for (int i = 0; i < rows_per_process; i++) {
    for (int j = 0; j < image_width; j++) {
      res_tmp[i * image_width + j] = (&newImage)->data[i][j];
    }
  }

  MPI_Gather(res_tmp, rows_per_process * image_width, mpi_pixel, tmp,
             rows_per_process * image_width, mpi_pixel, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    newData = malloc(image_height * sizeof(pixel *));
    for (int i = 0; i < image_height; i++) {
      newData[i] = malloc(image_width * sizeof(pixel));
    }

    for (int i = 0; i < image_height; i++) {
      for (int j = 0; j < image_width; j++) {
        newData[i][j] = tmp[i * image_width + j];
      }
    }

    bmpImage resultImage = {image_width, image_height, newData};

    // Write the image back to disk
    if (saveBmpImage(&resultImage, output) != 0) {
      fprintf(stderr, "Could not save output to '%s'!\n", output);
      freeBmpImage(image);
      goto error_exit;
    };
  }

graceful_exit:
  ret = 0;
error_exit:
  if (input)
    free(input);
  if (output)
    free(output);

  MPI_Type_free(&mpi_pixel);
  MPI_Type_free(&mpi_pixel_vector);

  // Finalize the MPI environment.
  MPI_Finalize();

  return ret;
};
