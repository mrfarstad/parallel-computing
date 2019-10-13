#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"
#include <mpi.h>

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

#define XSIZE_NEW XSIZE * 2 // Size of new image
#define YSIZE_NEW YSIZE * 2

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

	uchar *image = NULL;	// Three uchars per pixel (RGB)
	uchar *newImage = NULL; // Three uchars per pixel (RGB)
	if (world_rank == 0)
	{
		image = calloc(XSIZE * YSIZE * 3, 1);
		newImage = calloc(XSIZE_NEW * YSIZE_NEW * 3, 1);
		readbmp("before.bmp", image);
	}

	// Number of elements BEFORE scaling image
	int elementsPerProc = XSIZE * YSIZE * 3 / world_size;
	// Number of elements AFTER scaling image
	int resultElementsPerProc = XSIZE_NEW * YSIZE_NEW * 3 / world_size;

	uchar *subImageColor = calloc(elementsPerProc, 1);

	// Specifies number of elements for each prccess
	int *counts = calloc(world_size, sizeof(int));
	// Specifies data offset for each process
	int *offsets = calloc(world_size, sizeof(int));

	for (int i = 0; i < world_size; i++)
	{
		// Even processes get to colors the image
		if (i % 2 == 0)
		{
			counts[i] = elementsPerProc;
		}
		else
		{
			counts[i] = 0;
		}
		offsets[i] = i * elementsPerProc;
	}

	// Use Scatterv and Gatherv to to not color image uniformly
	MPI_Scatterv(
		image,
		counts,
		offsets,
		MPI_UNSIGNED_CHAR,
		subImageColor,
		counts[world_rank],
		MPI_UNSIGNED_CHAR,
		0,
		MPI_COMM_WORLD);

	for (int i = 0; i < counts[world_rank]; i += 3)
	{
		int tmp = subImageColor[i];
		subImageColor[i] = subImageColor[i + 2];
		subImageColor[i + 2] = tmp;
	}

	MPI_Gatherv(
		subImageColor,
		counts[world_rank],
		MPI_UNSIGNED_CHAR,
		image,
		counts,
		offsets,
		MPI_UNSIGNED_CHAR,
		0,
		MPI_COMM_WORLD);

	uchar *subImage = calloc(elementsPerProc, 1);
	uchar *resultSubImage = calloc(resultElementsPerProc, 1);

	//Scatter the image to all processes so they receive sub images
	MPI_Scatter(
		image,
		elementsPerProc,
		MPI_UNSIGNED_CHAR,
		subImage,
		elementsPerProc,
		MPI_UNSIGNED_CHAR,
		0,
		MPI_COMM_WORLD);

	for (int i = 0; i < YSIZE / world_size; i++)
	{
		for (int j = 0; j < XSIZE * 3; j += 3)
		{
			int index = i * XSIZE * 3 + j;
			int newIndex = 2 * (i * XSIZE_NEW * 3 + j);

			for (int color = 0; color < 3; color++)
			{
				// Same pixel
				resultSubImage[newIndex + color] =
					// Next pixel
					resultSubImage[newIndex + 3 + color] =
						// Next line pixel
					resultSubImage[newIndex + XSIZE_NEW * 3 + color] =
						// Next line next pixel
					resultSubImage[newIndex + XSIZE_NEW * 3 + 3 + color] =
						// Original pixel
					subImage[index + color];
			}
		}
	}

	MPI_Gather(
		resultSubImage,
		resultElementsPerProc,
		MPI_UNSIGNED_CHAR,
		newImage,
		resultElementsPerProc,
		MPI_UNSIGNED_CHAR,
		0,
		MPI_COMM_WORLD);

	if (world_rank == 0)
	{
		savebmp("after.bmp", newImage, XSIZE_NEW, YSIZE_NEW);
		free(image);
		free(newImage);
	}

	free(subImage);
	free(resultSubImage);
	// Finalize the MPI environment.
	MPI_Finalize();
	return 0;
}
