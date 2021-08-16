#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define DIM 3

int main(int argc, char **argv) {
	int i, j, k, n, c;
	double dmin, dx, start, finish, total = 0.0;
	double *x, *mean, *sum;
	int *cluster, *count, color;
	int flips;

	MPI_Init(& argc, &argv);
    
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	if(scanf("%d", &k) != 1) {
		printf("Erro: não foi possível ler o número de centróides.");
		exit(1);
	}
	if(scanf("%d", &n) != 1) {
		printf("Erro: não foi possível ler o número de pontos.");
		exit(1);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	finish = MPI_Wtime();
	total += finish - start;

	x = (double *)malloc(sizeof(double)*DIM*n);
	mean = (double *)malloc(sizeof(double)*DIM*k);
	sum= (double *)malloc(sizeof(double)*DIM*k);
	cluster = (int *)malloc(sizeof(int)*n);
	count = (int *)malloc(sizeof(int)*k);

	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	for (i = 0; i < k; i++)
		if(scanf("%lf %lf %lf", mean+i*DIM, mean+i*DIM+1, mean+i*DIM+2) != 3) {
			printf("Erro: não foi possível ler a coordenada do %dº centróide.", i + 1);
			exit(1);
		}

	for (i = 0; i < n; i++)
		if(scanf("%lf %lf %lf", x+i*DIM, x+i*DIM+1, x+i*DIM+2) != 3) {
			printf("Erro: não foi possível ler a coordenada do %dº ponto.", i + 1);
			exit(1);
		}
	MPI_Barrier(MPI_COMM_WORLD);
	finish = MPI_Wtime();
	total += finish - start;

	flips = n;
	for (i = 0; i < n; i++)
		cluster[i] = 0;
	
	while (flips>0) {
		flips = 0;

		for (j = 0; j < k; j++) {
			count[j] = 0;
			for (i = 0; i < DIM; i++)
				sum[j*DIM+i] = 0.0;
		}

		for (i = 0; i < n; i++) {
			dmin = -1; color = cluster[i];
			for (c = 0; c < k; c++) {
				dx = 0.0;
				for (j = 0; j < DIM; j++)
					dx +=  (x[i*DIM+j] - mean[c*DIM+j])*(x[i*DIM+j] - mean[c*DIM+j]);
				if (dx < dmin || dmin == -1) {
					color = c;
					dmin = dx;
				}
			}
			if (cluster[i] != color) {
				flips++;
				cluster[i] = color;
	      	}
		}

	    for (i = 0; i < n; i++) {
			count[cluster[i]]++;
			for (j = 0; j < DIM; j++)
				sum[cluster[i]*DIM+j] += x[i*DIM+j];
		}

		MPI_Barrier(MPI_COMM_WORLD);
		start = MPI_Wtime();
		for (i = 0; i < k; i++) {
			for (j = 0; j < DIM; j++) {
				mean[i*DIM+j] = sum[i*DIM+j]/count[i];
  			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		finish = MPI_Wtime();
		total += finish - start;
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	for (i = 0; i < k; i++) {
		for (j = 0; j < DIM; j++)
			printf("%5.2f ", mean[i*DIM+j]);
		printf("\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	finish = MPI_Wtime();
	total += finish - start;
    printf("%f\n", total);
	MPI_Finalize();
	return(0);
}
