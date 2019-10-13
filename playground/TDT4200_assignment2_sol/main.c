#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

#define XSIZE_NEW XSIZE * 2
#define YSIZE_NEW YSIZE * 2

// Scale the image to twice the size in each direction, becoming four times as large
void handleImage(uchar *image, uchar *newImage, int world_rank, int world_size)
{
	for (int i = 0; i < YSIZE / world_size; i++)
	{
		for (int j = 0; j < 3 * XSIZE; j += 3)
		{
			int index = 3 * i * XSIZE + j;

			// Multiply with 2 because copying pixels needs offset in new array
			int newImageIndex = 2 * (3 * i * XSIZE_NEW + j);
			// Index for the pixels being copied
			int newImageNextLineIndex = newImageIndex + 3 * XSIZE_NEW;

			// Alter the colors of the image
			uchar tmp = image[index];
			image[index] = image[index + 2];
			image[index + 2] = tmp;

			for (int c = 0; c < 3; c++)
			{
				// Pixel at same position as before
				newImage[newImageIndex + c] = image[index + c];
				// Pixel at same row to the right
				newImage[newImageIndex + 3 + c] = image[index + c];
				// Pixel at next row
				newImage[newImageNextLineIndex + c] = image[index + c];
				// Pixel at next row to the right
				newImage[newImageNextLineIndex + 3 + c] = image[index + c];
			}
		}
	}
}

int main()
{
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Amount of elements per row times amount of rows per process
	int num_elements_per_proc = 3 * XSIZE * YSIZE / world_size;

	// Allocate space and read image file
	uchar *image = NULL;
	if (world_rank == 0)
	{
		image = calloc(3 * XSIZE * YSIZE, sizeof(uchar)); // Three uchars per pixel (RGB)
		readbmp("before.bmp", image);
	}

	// For each process, create a buffer that will hold a subset of the entire array
	uchar *sub_image = (uchar *)malloc(num_elements_per_proc);

	// Scatter the rows from the image from the root process to all processes in the MPI world
	MPI_Scatter(image, num_elements_per_proc, MPI_UNSIGNED_CHAR, sub_image,
				num_elements_per_proc, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	// Allocate space for the result image with new dimensions
	uchar *newImage = NULL;
	if (world_rank == 0)
	{
		newImage = calloc(3 * XSIZE_NEW * YSIZE_NEW, sizeof(uchar)); // Three uchars per pixel (RGB)
	}

	// Allocate space for each sub result
	uchar *sub_image_result = (uchar *)malloc(sizeof(uchar) * 3 * XSIZE_NEW * YSIZE_NEW / world_size);

	// Handle each sub array
	handleImage(sub_image, sub_image_result, world_rank, world_size);

	// Combine all sub arrays to the result image array
	MPI_Gather(
		sub_image_result,
		sizeof(uchar) * 3 * XSIZE_NEW * YSIZE_NEW / world_size,
		MPI_UNSIGNED_CHAR,
		newImage,
		sizeof(uchar) * 3 * XSIZE_NEW * YSIZE_NEW / world_size,
		MPI_UNSIGNED_CHAR,
		0,
		MPI_COMM_WORLD);

	// Save the combined result image
	if (world_rank == 0)
	{
		savebmp("after.bmp", newImage, XSIZE_NEW, YSIZE_NEW);
	}

	// Finalize the MPI environment.
	MPI_Finalize();
	// Free allocated space to prevent memory leaks
	free(image);
	free(newImage);
	free(sub_image);
	free(sub_image_result);

	return 0;
}
