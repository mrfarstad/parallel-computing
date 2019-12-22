#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

  int N = 10000000;
  long long sum = 0;

  int *a = malloc(N * sizeof(int));
  int threads;
  // int threads = strtol(argv[1], NULL, 10);

#pragma omp parallel for default(none) shared(N, a)
  for (int i = 0; i < N; i++) {
    a[i] = i;
  }

#pragma omp parallel for default(none) shared(N, a, threads) reduction(+ : sum)
  for (int i = 0; i < N; i++) {
    threads = omp_get_num_threads();
    sum += a[i];
  }

  printf("Number of threads: %d\n", threads);
  printf("Sum: %llu\n", sum);
}
