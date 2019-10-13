#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

#define XSIZE_NEW XSIZE * 2 // Size of new image
#define YSIZE_NEW YSIZE * 2

int main()
{
  uchar *image = calloc(XSIZE * YSIZE * 3, 1);            // Three uchars per pixel (RGB)
  uchar *newImage = calloc(XSIZE_NEW * YSIZE_NEW * 3, 1); // Three uchars per pixel (RGB)
  readbmp("before.bmp", image);

  for (int i = 0; i < YSIZE; i++)
  {
    for (int j = 0; j < XSIZE * 3; j += 3)
    {
      int index = i * XSIZE * 3 + j;
      int newIndex = 2 * (i * XSIZE_NEW * 3 + j);

      int tmp = image[index];
      image[index] = image[index + 2];
      image[index + 2] = tmp;

      for (int color = 0; color < 3; color++)
      {
        // Same pixel
        newImage[newIndex + color] =
            // Next pixel
            newImage[newIndex + 3 + color] =
                // Next line pixel
            newImage[newIndex + XSIZE_NEW * 3 + color] =
                // Next line next pixel
            newImage[newIndex + XSIZE_NEW * 3 + 3 + color] =
                // Original pixel
            image[index + color];
      }
    }
  }

  savebmp("after.bmp", newImage, XSIZE_NEW, YSIZE_NEW);
  free(image);
  return 0;
}
