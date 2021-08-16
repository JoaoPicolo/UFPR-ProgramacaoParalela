#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define DIM 3
#define ROOT 0

int main(int argc, char **argv) {
	int i, j, k, c, flips, total_n, n, last_n;
	int p_rank, n_procs;
	double dmin, dx;
	double *ori_x, *x, *mean, *ori_sum, *sum;
	int *cluster, *ori_count, *count, *sendcounts, *displs, color;

	MPI_Init(& argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

	sendcounts = (int *)malloc(sizeof(int)*n_procs);
	displs = (int *)malloc(sizeof(int)*n_procs);

	if(p_rank == ROOT) {
		if(scanf("%d", &k) != 1) {
			printf("Erro: não foi possível ler o número de centróides.\n");
			exit(1);
		}
		if(scanf("%d", &total_n) != 1) {
			printf("Erro: não foi possível ler o número de pontos.\n");
			exit(1);
		}

		n = ceil((double)total_n / n_procs);
		last_n = total_n - ((n_procs - 1) * n);
		
		int offset = 0;
		for(i = 0; i < n_procs - 1; i++) {
			sendcounts[i] = n*DIM;
			displs[i] = offset;
			offset += n*DIM;
		}
		sendcounts[n_procs-1] = last_n*DIM;
		displs[n_procs-1] = offset;
	}

	MPI_Bcast(&k, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(sendcounts, n_procs, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(displs, n_procs, MPI_INT, ROOT, MPI_COMM_WORLD);
	n = sendcounts[p_rank]/DIM;
	
	if(p_rank == ROOT) {
		ori_x = (double *)malloc(sizeof(double)*DIM*total_n);
		ori_sum = (double *)malloc(sizeof(double)*DIM*k);
		ori_count = (int *)malloc(sizeof(int)*k);
	}
	else {
		ori_x = (double *)malloc(0);
		ori_sum = (double *)malloc(0);
		ori_count = (int *)malloc(0);
	}

	x = (double *)malloc(sizeof(double)*DIM*n);
	mean = (double *)malloc(sizeof(double)*DIM*k);
	sum = (double *)malloc(sizeof(double)*DIM*k);
	cluster = (int *)malloc(sizeof(int)*n);
	count = (int *)malloc(sizeof(int)*k);

	if(p_rank == ROOT) {
		for (i = 0; i < k; i++)
			if(scanf("%lf %lf %lf", mean+i*DIM, mean+i*DIM+1, mean+i*DIM+2) != 3) {
				printf("Erro: não foi possível ler a coordenada do %dº centróide.\n", i + 1);
				exit(1);
			}

		for (i = 0; i < total_n; i++)
			if(scanf("%lf %lf %lf", ori_x+i*DIM, ori_x+i*DIM+1, ori_x+i*DIM+2) != 3) {
				printf("Erro: não foi possível ler a coordenada do %dº ponto.\n", i + 1);
				exit(1);
			}
	}

	MPI_Scatterv(ori_x, sendcounts, displs, MPI_DOUBLE, x, n*DIM, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(mean, k*DIM, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	for (i = 0; i < n; i++)
		cluster[i] = 0;

	flips = n;
	while (flips > 0) {
		flips = 0;

		if(p_rank == ROOT) {
			for (j = 0; j < k; j++) {
				count[j] = 0;
				ori_count[j] = 0;
				for (i = 0; i < DIM; i++)
					sum[j*DIM+i] = 0.0;
					ori_sum[j*DIM+i] = 0.0;
			}
		}
		else {
			for (j = 0; j < k; j++) {
				count[j] = 0;
				for (i = 0; i < DIM; i++)
					sum[j*DIM+i] = 0.0;
			}
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

		MPI_Reduce(count, ori_count, k, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
		MPI_Reduce(sum, ori_sum, k*DIM, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
		MPI_Allreduce(&flips, &flips, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		if(p_rank == ROOT) {
			for (i = 0; i < k; i++) {
				for (j = 0; j < DIM; j++) {
					mean[i*DIM+j] = ori_sum[i*DIM+j]/ori_count[i];
				}
			}
		}

		MPI_Bcast(mean, k*DIM, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	}
	
	if(p_rank == ROOT) {
		for (i = 0; i < k; i++) {
			for (j = 0; j < DIM; j++)
				printf("%5.2f ", mean[i*DIM+j]);
			printf("\n");
		}
	}

	MPI_Finalize();
	return(0);
}