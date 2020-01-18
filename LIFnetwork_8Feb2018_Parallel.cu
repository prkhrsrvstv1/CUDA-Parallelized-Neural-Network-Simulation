#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 20
#define MXCOL 10000
#define NL_min 0
#define NL_max 0
#define NL_step 1
#define Ng_max 1

typedef struct {
  double vth;
} shared_mem;

typedef struct {
  int iL, nL_break, ig, ic;
} simulation_params;

typedef struct {

} simulation_result;

int synaptic_weighs_connected_network(double w[][N], int nL);

__global__ void simulate() {
  
}

int main() {
  
  return 0;
}