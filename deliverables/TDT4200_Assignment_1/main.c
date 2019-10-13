#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

#define XSIZE_NEW XSIZE * 2
#define YSIZE_NEW YSIZE * 2

#define IMAGE_SIZE XSIZE * 3 * YSIZE

void handleImage(uchar *image, uchar *newImage)
{
  for (int i = 0; i < YSIZE; i++)
  {
    for (int j = 0; j < 3 * XSIZE; j += 3)
    {
      int index = 3 * i * XSIZE + j;

      // Multiply with 2 because we are copying pixels gives offset for new
      /**/
      int newImageIndex = 2 * (3 * i * XSIZE_NEW + j);
      int newImageNextLineIndex = newImageIndex + 3 * XSIZE_NEW;

      // Alter the colors of the image
      uchar tmp = image[index];
      image[index] = image[index + 2];
      image[index + 2] = tmp;

      for (int c = 0; c < 3; c++)
      {
        newImage[newImageIndex + c] = image[index + c];
        newImage[newImageIndex + 3 + c] = image[index + c];
        newImage[newImageNextLineIndex + c] = image[index + c];
        newImage[newImageNextLineIndex + 3 + c] = image[index + c];
      }
    }
  }
}

int main()
{
  uchar *image = calloc(IMAGE_SIZE, 1); // Three uchars per pixel (RGB)
  readbmp("before.bmp", image);

  uchar *newImage = calloc(4 * IMAGE_SIZE, 1); // Three uchars per pixel (RGB)

  handleImage(image, newImage);

  savebmp("after.bmp", newImage, XSIZE * 2, YSIZE * 2);

  free(image);
  free(newImage);
  return 0;
}
