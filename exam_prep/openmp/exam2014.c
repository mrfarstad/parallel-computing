#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {

  int N = 10;

  int *a = calloc(N, sizeof(int));

  for (int i = 0; i < N; i++) {
    a[i] = i;
  }

  int max = a[0];

  omp_lock_t lock;
  omp_init_lock(&lock);

#pragma omp parallel for
  for (int i = 1; i < N; i++) {
    // #pragma omp critical
    omp_set_lock(&lock);
    if (a[i] > max) {
      max = a[i];
    }
    omp_unset_lock(&lock);
  }

  omp_destroy_lock(&lock);

  printf("Max: %d\n", max);

  return 0;
}
