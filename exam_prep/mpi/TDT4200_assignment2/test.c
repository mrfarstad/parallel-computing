#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

  MPI_Init(NULL, NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int send_int = 1337;
  int recv_int;

  MPI_Request request;

  int right_rank = (world_rank + 1) % world_size;
  int left_rank = world_rank - 1;
  if (left_rank < 0) {
    left_rank = world_size - 1;
  }

   printf("Process %d send: %d\n", world_rank, send_int);
  MPI_Isend(&send_int, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, &request);

  //
  // MPI_Wait(&request, MPI_STATUS_IGNORE);
  
  MPI_Recv(&recv_int, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
   printf("Process %d recv: %d\n", world_rank, recv_int);

  // MPI_Irecv(&recv_int, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, &request);

  // printf("Process %d send: %d\n", world_rank, send_int);
  // MPI_Ssend(&send_int, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);

  // MPI_Wait(&request, MPI_STATUS_IGNORE);
  // printf("Process %d recv: %d\n", world_rank, recv_int);

  MPI_Finalize();

  return 0;
}
