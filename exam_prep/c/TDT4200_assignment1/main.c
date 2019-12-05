#include "bitmap.h"
#include <stdio.h>
#include <stdlib.h>

#define XSIZE 2560 // Size of before image
#define YSIZE 2048
#define IMAGE_SIZE XSIZE *YSIZE * 3
#define XSIZE_NEW XSIZE * 2
#define YSIZE_NEW YSIZE * 2

void changeColor(uchar *image) {
  for (unsigned int i = 0; i < IMAGE_SIZE; i += 3) {
    uchar tmp = image[i];
    image[i] = image[i + 2];
    image[i + 2] = tmp;
  }
}

void changeSize(uchar *image, uchar *new_image) {
  // Loop over original image and double resolution in each direction
  for (size_t i = 0; i < YSIZE; i++) {
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
  uchar *image = malloc(IMAGE_SIZE); // Three uchars per pixel (RGB)
  uchar *new_image = malloc(IMAGE_SIZE * 4);

  readbmp("before.bmp", image);

  // Alter the image here
  changeColor(image);
  changeSize(image, new_image);

  savebmp("after.bmp", new_image, XSIZE_NEW, YSIZE_NEW);
  free(image);
  free(new_image);
  return 0;
}
