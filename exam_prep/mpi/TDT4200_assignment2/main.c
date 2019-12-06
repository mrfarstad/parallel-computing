#include "bitmap.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define XSIZE 2560 // Size of before image
#define YSIZE 2048
#define IMAGE_SIZE XSIZE *YSIZE * 3
#define XSIZE_NEW XSIZE * 2
#define YSIZE_NEW YSIZE * 2
#define IMAGE_SIZE_NEW XSIZE_NEW *YSIZE_NEW * 3

void changeColor(uchar *image, int work_size) {
  for (unsigned int i = 0; i < work_size; i += 3) {
    uchar tmp = image[i];
    image[i] = image[i + 2];
    image[i + 2] = tmp;
  }
}

void changeSize(uchar *image, uchar *new_image, int world_size, int work_size) {
  for (size_t i = 0; i < work_size; i++) {
    for (size_t j = 0; j < XSIZE * 3; j += 3) {
      size_t index = i * XSIZE * 3 + j;
      size_t new_index = 2 * (i * XSIZE_NEW * 3 + j);
      for (size_t c = 0; c < 3; c++) {
        new_image[new_index + c] = image[index + c];
        new_image[new_index + c + 3] = image[index + c];
        new_image[new_index + XSIZE_NEW * 3 + c] = image[index + c];
        new_image[new_index + XSIZE_NEW * 3 + c + 3] = image[index + c];
      }
    }
  }
}

int main() {

  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  uchar *image;
  uchar *new_image;

  int *sendcounts = malloc(world_size * sizeof(int));
  int *recvcounts = malloc(world_size * sizeof(int));
  int *send_displs = malloc(world_size * sizeof(int));
  int *recv_displs = malloc(world_size * sizeof(int));

  if (world_rank == 0) {
    image = malloc(IMAGE_SIZE);
    new_image = malloc(IMAGE_SIZE * 4);
    readbmp("before.bmp", image);
  }

  for (int i = 0; i < world_size; i++) {
    int size = IMAGE_SIZE / world_size;
    int size_new = IMAGE_SIZE_NEW / world_size;
    if (i < YSIZE % world_size) {
      size += XSIZE * 3;
      size_new += XSIZE_NEW * 3;
    }

    sendcounts[i] = size;
    recvcounts[i] = size_new;
    send_displs[i] = i * size;
    recv_displs[i] = i * size_new;
  }

  uchar *tmp_image = malloc(sendcounts[world_rank]);
  uchar *tmp_image_new = malloc(recvcounts[world_rank]);

  MPI_Scatterv(image, sendcounts, send_displs, MPI_UNSIGNED_CHAR, tmp_image,
               sendcounts[world_rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  changeColor(tmp_image, sendcounts[world_rank]);

  changeSize(tmp_image, tmp_image_new, world_size,
             sendcounts[world_rank] / (3 * XSIZE));

  MPI_Gatherv(tmp_image_new, recvcounts[world_rank], MPI_UNSIGNED_CHAR,
              new_image, recvcounts, recv_displs, MPI_UNSIGNED_CHAR, 0,
              MPI_COMM_WORLD);

  // Alter the image here
  if (world_rank == 0) {
    savebmp("after.bmp", new_image, XSIZE_NEW, YSIZE_NEW);
    free(image);
    free(new_image);
  }

  free(tmp_image);

  MPI_Finalize();

  return 0;
}
