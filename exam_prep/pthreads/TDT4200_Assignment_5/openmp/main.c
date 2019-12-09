#include <cblas.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Threshold for testing validity of matrix matrix multiplication
#define ERROR_THRESHOLD 0.0001

// For measuring wall time using omp_get_wtime()
static double start;
static double end;

// Serial version. Do not change this!
void serial_mxm(const double *A, const double *B, double *C, int m, int n,
                int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      C[i * n + j] = 0;
      for (int l = 0; l < k; l++) {
        C[i * n + j] += A[i * k + l] * B[l * n + j];
      }
    }
  }
}

void omp_mxm(double *A, double *B, double *C, int m, int n, int k) {
  int i = 0;
  int j = 0;
  int l = 0;
#pragma omp parallel for collapse(2) default(none) private(i, j, l)            \
    shared(A, B, C, m, n, k)
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      C[i * n + j] = 0;
      for (l = 0; l < k; l++) {
        C[i * n + j] += A[i * k + l] * B[l * n + j];
      }
    }
  }
}

void blas_mxm(double *A, double *B, double *C, int m, int n, int k) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n,
              0, C, n);
}

int main(int argc, char **argv) {
  if (argc <= optind) {
    printf("Please provide version:\n");
    printf("\ts(serial),\n");
    printf("\to(penmp) or\n");
    printf("\tb(las)\n");
    return 0;
  }
  char input = argv[optind][0];
  optind++;
  // Simple assumptions that any additional arguments means we want to test the
  // results
  bool test = !(argc <= optind);

  int m = 2000;
  int n = 1000;
  int k = 200;

  double *A = (double *)malloc(m * k * sizeof(double));
  double *B = (double *)malloc(k * n * sizeof(double));
  double *C = (double *)malloc(m * n * sizeof(double));

  // Intializing matrix data
  for (int i = 0; i < (m * k); i++) {
    A[i] = (double)(i + 1);
  }

  for (int i = 0; i < (k * n); i++) {
    B[i] = (double)(-i - 1);
  }

  for (int i = 0; i < (m * n); i++) {
    C[i] = 0.0;
  }
  start = omp_get_wtime();
  switch (input) {
  case 's':
    serial_mxm(A, B, C, m, n, k);
    break;
  case 'o':
    omp_mxm(A, B, C, m, n, k);
    break;
  case 'b':
    blas_mxm(A, B, C, m, n, k);
    break;
  default:
    printf("Please provide version:\n");
    printf("\ts(serial),\n");
    printf("\to(penmp) or\n");
    printf("\tb(las)\n");
    return 0;
  }
  end = omp_get_wtime();

  printf("\nTop left of A:\n");
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%8.2f\t", A[i * k + j]);
    }
    printf("\n");
  }

  printf("\nTop left of B:\n");
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%8.2f\t", B[i * n + j]);
    }
    printf("\n");
  }

  printf("\nTop left of C:\n");
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%8.2f\t", C[i * n + j]);
    }
    printf("\n");
  }

  if (test) {
    double *C2 = (double *)malloc(m * n * sizeof(double));
    serial_mxm(A, B, C2, m, n, k);
    bool correct = true;
    for (int i = 0; i < (m * n); i++) {
      if (fabs(C[i] - C2[i]) > ERROR_THRESHOLD) {
        correct = false;
      }
    }
    if (correct) {
      printf("\nMatrix multiplication succeeded\n");
    } else {
      printf("\nMatrix multiplication failed\n");
      printf("Top left of correct C:\n");
      for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
          printf("%8.2f\t", C2[i * n + j]);
        }
        printf("\n");
      }
    }
  }

  printf("\nVersion: %c, time: %.4f\n", input, end - start);

  return 0;
}
