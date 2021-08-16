#include <stdio.h>
#include <stdlib.h>

#define DIM 3
#define NUM_THREADS 4

int main(void) {
	int i, j, k, n, c;
	double dmin, dx;
	double *x, *mean, *sum;
	int *cluster, *count, color;
	int flips;

	if(scanf("%d", &k) != 1) {
		printf("Erro: não foi possível ler o número de centróides.");
		exit(1);
	}
	if(scanf("%d", &n) != 1) {
		printf("Erro: não foi possível ler o número de pontos.");
		exit(1);
	}
	x = (double *)malloc(sizeof(double)*DIM*n);
	mean = (double *)malloc(sizeof(double)*DIM*k);
	sum = (double *)malloc(sizeof(double)*DIM*k);
	cluster = (int *)malloc(sizeof(int)*n);
	count = (int *)malloc(sizeof(int)*k);

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

	#pragma omp parallel for proc_bind(master) num_threads(NUM_THREADS) private(i) shared(n)
	for (i = 0; i < n; i++)
		cluster[i] = 0;

	flips = n;
	while (flips > 0) {
		flips = 0;

        #pragma omp parallel num_threads(NUM_THREADS) private(j,i,dmin,color,c,dx) shared(flips,k,n)
        {
            #pragma omp for schedule(guided)
            for (j = 0; j < k; j++) {
                count[j] = 0;
            }

            #pragma omp for schedule(guided)
            for (j = 0; j < k; j++) {
                for (i = 0; i < DIM; i++)
                    sum[j*DIM+i] = 0.0;
            }

            #pragma omp for schedule(guided) reduction(+:flips)
            for (i = 0; i < n; i++) {
			    dmin = -1; color = cluster[i];

                for (c = 0; c < k; c++) {
                    dx = 0.0;
                    
                    for (j = 0; j < DIM; j++)
                        dx += (x[i*DIM+j] - mean[c*DIM+j])*(x[i*DIM+j] - mean[c*DIM+j]);
                    
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
        }

		#pragma parallel for proc_bind(spread) reduction(+:sum[:k*DIM]) num_threads(NUM_THREADS)
		for (i = 0; i < n; i++) {
            count[cluster[i]]++;
			for (j = 0; j < DIM; j++)
				sum[cluster[i]*DIM+j] += x[i*DIM+j];
		}
		
		#pragma parallel for proc_bind(spread) simd num_threads(NUM_THREADS)
		for (i = 0; i < k; i++)
			for (j = 0; j < DIM; j++)
				mean[i*DIM+j] = sum[i*DIM+j]/count[i];
	}

	for (i = 0; i < k; i++) {
		for (j = 0; j < DIM; j++)
			printf("%5.2f ", mean[i*DIM+j]);
		printf("\n");
	}
	
	#ifdef DEBUG
	for (i = 0; i < n; i++) {
		for (j = 0; j < DIM; j++)
			printf("%5.2f ", x[i*DIM+j]);
		printf("%d\n", cluster[i]);
	}
	#endif
	return(0);
}
