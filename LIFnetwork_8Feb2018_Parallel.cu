#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 2
#define MAXCOL 10000
#define Nic 1
#define NL_min 0
#define NL_max 0
#define NL_step 1
#define Ng_max 1
#define N_THREADS_PER_BLOCK 1
#define N_BLOCKS 1

typedef struct {
  int All_sync_count1[NL_max-NL_min+1][Ng_max];
  int All_sync_count2[NL_max-NL_min+1];
  double dt, epsilon, vth, vreset, a, b, tol;
  long Nstep;
} global_mem;

typedef struct {
  int iL, ig, ic;
} simulation_params;

typedef struct {
  int ic, iL, nL_break, ig;
  double v_init[N];
  double tspike[N][MAXCOL];
} simulation_result;

/* Generate a adjacency matrix for a coonnected graph with nL edges missing */
__device__ int synaptic_weights_connected_network(double w[][N], int nL) {

  int i,j,k,kk,neuron1,neuron2;
  double w_flag[N][N];
  int syn_to_remove, tot_syn_removed ;
  int connected_nodes[N] ;
  int current_ptr, endptr, parent_node;
  int flag_connected = 0 ;
  int flag_already_connected;

  // GENERATE AN ALL-TO-ALL NETWORK ************************************************************************
  for(i = 0; i < N; i++) {
    for(j = 0; j < N; j++) {
      if(j != i){
        w[i][j] = 1;
      }
      else if(j == i){
        w[i][j] =0;
      } 
    }
  }

  // REMOVE SYNAPSES FROM ABOVE ALL-TO-ALL NETWORK *********************************************************

  syn_to_remove = nL;
  tot_syn_removed = 0;

  // Initialize array w_flag
  for(k = 0; k < N; k++) {
    for(kk = 0; kk < N; kk++) {
      w_flag[k][kk] = 0; // w_flag[k][kk] is changed to value 1, if the synapse between k --> kk is removed
    }
  }

  // Generate a new network by removing synapses randomly
  while(tot_syn_removed < syn_to_remove) {
    int neuron1 = rand() % N;
    int neuron2 = rand() % N;
    if(neuron1 != neuron2) {
      if(w_flag[neuron1][neuron2] == 0) { // synapse between these two neurons has not been changed.
        w_flag[neuron1][neuron2] = 1;
        w_flag[neuron2][neuron1] = 1;
        w[neuron1][neuron2] = 0;
        w[neuron2][neuron1] = w[neuron1][neuron2];
        tot_syn_removed++;
      }
    }
  }


  // Is the network generated above connected ? /////////////


  //w[0][0] = 0; w[0][1] = 1; w[0][2] = 1; w[0][3] = 0; w[0][4] = 1; w[0][5] = 0;

  //w[1][0] = w[0][1]; w[1][1] = 0 ; w[1][2] = 1 ; w[1][3] = 0; w[1][4] = 0; w[1][5] = 1;

  //w[2][0] = w[0][2]; w[2][1] = w[1][2] ; w[2][2] = 0 ; w[2][3] = 0; w[2][4] = 1; w[2][5] = 0;

  //w[3][0] = w[0][3]; w[3][1] = w[1][3] ; w[3][2] = w[2][3] ; w[3][3] = 0; w[3][4] = 0; w[3][5] = 0;

  //w[4][0] = w[0][4]; w[4][1] = w[1][4] ; w[4][2] = w[2][4] ; w[4][3] = w[3][4]; w[4][4] = 0; w[4][5] = 1;

  //w[5][0] = w[0][5]; w[5][1] = w[1][5] ; w[5][2] = w[2][5] ; w[5][3] = w[3][5]; w[5][4] = w[4][5]; w[5][5] = 0;

  //w[0][0] = 0 ; w[0][1] = 0; w[0][2] = 1; w[0][3]=0;
  //w[1][0] = w[0][1] ; w[1][1] = 0;  w[1][2] = 1; w[1][3] =0;
  //w[2][0]=w[0][2] ; w[2][1]=w[1][2]; w[2][2] =0; w[2][3] = 1;
  //w[3][0] = w[0][3] ; w[3][1] = w[1][3] ; w[3][2] = w[2][3] ; w[3][3]=0 ;

  for(k = 0; k < N; k++) {
    for(kk = 0; kk < N; kk++) {
      w_flag[k][kk] = 0; // w_flag[k][kk] is changed to value 1, if the synapse between k --> kk is removed
    }
  }

  connected_nodes[0] = 0;
  for(i=1;i<N;i++) {
    connected_nodes[i] = -1;
  }
  current_ptr = 0;
  endptr = 0 ;  // points towards the last non-zero element in the connected_nodes array

  while(current_ptr <= endptr) {

    for(i = 0; i < N; i++) {
      parent_node = connected_nodes[current_ptr] ;

      flag_already_connected = 0 ;

      for(j = 0; j <= endptr; j++) {
        if(connected_nodes[j] == i) {
          flag_already_connected = 1;
        }
      }

      if(w[parent_node][i] == 1) {
        if(w_flag[parent_node][i] == 0) {
          if(flag_already_connected ==0) {
            endptr ++ ;
            connected_nodes[endptr] = i ; // stores node numbers connected to parent_node

            w_flag[parent_node][i] = 1 ;
            w_flag[i][parent_node] = w_flag[parent_node][i] ; //links already visited

            //printf("i= %d \t endptr= %d \t current_ptr= %d \t connected_nodes[endptr] = %d \n",i, endptr,current_ptr,connected_nodes[endptr]);
          }
        }
      }

      if (i == N-1) {
        current_ptr++ ;
      }	
    }
  }

  if(endptr == N-1) {
    flag_connected = 1 ;
  }

  return flag_connected;
}

/* Create weight matrices in GPU memory */
__global__ void store_weights(double w[(NL_max - NL_min) * Ng_max / NL_step][N][N]) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int nL_break = NL_min + threadId * NL_step;
  int flag_connected;
  for(int i = 0; i < Ng_max; ++i) {
    flag_connected = 0;
    do {
      flag_connected = synaptic_weights_connected_network(w[threadId * Ng_max + i], nL_break);
    } while(flag_connected == 0);
  }
}

/* Run a simulation on a single thread */
__global__ void simulate(simulation_params *params, simulation_result *results, global_mem *g_mem, double w[(NL_max - NL_min) * Ng_max / NL_step][N][N]) {
  // "threadId" is used as an index into the arrays "params" and "results".
  // Everything that was being written to a file is now returned in a struct.
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  
  results[threadId].ic = params[threadId].ic;
  results[threadId].iL = params[threadId].iL;
  results[threadId].nL_break = params[threadId].nL_break;
  results[threadId].ig = params[threadId].ig;
  
  int i, k kk, t_old, t_new, InSync_neurons;
  int spike_count[N], spike[N], push_up_flag[N];
  double f0, f1, f2, f3, f4, tspike_diff1, tspike_diff2;
  double v_old[N], v_new[N], push_up_amnt[N];
  double v_initnew[20]= {0.00778832, 0.355919, 0.426307, 0.183062,
                         0.272762, 0.532633, 0.339171, 0.242097,
                         0.523038, 0.638838, 0.632368, 0.778564,
                         0.110892, 0.347691, 0.696286, 0.791943,
                         0.5257, 0.127494, 0.716965, 0.151006};

  // Generate initial state
  for(kk = 0; kk < N; kk++) {
    /* Change rand() to cuRAND:: */
    results[threadId].v_init[kk] = rand() % (g_mem->vth*1000);
    results[threadId].v_init[kk] = results[threadId].v_init[kk] / 1000;
    v_old[kk] = results[threadId].v_init[kk];
  }
  
  for(kk = 0; kk < N; kk++){		
    results[threadId].v_init[kk] = v_initnew[kk];
    v_old[kk] = results[threadId].v_init[kk];
  }

  // initialize arrays
  for(k=0; k < N; k++) {
    spike_count[k] = 0; //keeps a count of the number spikes in neuron k so far
  }

  for(k = 0; k < N; k++){
    for(i = 0; i < MAXCOL; i++){
      results[threadId].tspike[k][i] = 0; // counts the spike time of "i_th" spike of neuron number "k"
    }
  }

  // Time loop begins
  t_old = 0;
  for(i = 1; i < g_mem->Nstep; i++) { 	

    t_new = i*(g_mem->dt);

    // Identify (1) the neurons that spiked in previous time step, (2) time of all the spikes of each neuron
    // (3) total number of spikes in each neuron so far
    for(kk = 0; kk < N; kk++) {
      if(v_old[kk] >= g_mem->vth) {
        spike[kk] = 1; // if neuron spiked
        spike_count[kk]++ ;
        results[threadId].tspike[kk][spike_count[kk]] = t_old;
      }
      else {	
        spike[kk] = 0; // if neuron did not spike
      }
    }

    // Find voltage push-up amount for each neuron (if atleast one neuron other than itself spiked)
    for(kk = 0; kk < N; kk++) {
      push_up_amnt[kk] = 0; // initialize these arrays at every time step
      push_up_flag[kk] = 0;
    }
    for(kk = 0; kk < N; kk++) {
      for(k = 0; k < N; k++) {	
        if(k != kk && spike[kk] != 1 && spike[k]==1) {
          push_up_amnt[kk] = push_up_amnt[kk] +
                             (g_mem->epsilon) * w[threadId % Nic][kk][k] * spike[k];
          push_up_flag[kk] = 1;
        }
      }
    }

    // Finally update voltages of each neuron - using Euler method if no neuron fired & by pushing up the
    // voltage value by push_up_amnt if some neurons fired.
    for(kk = 0; kk < N; kk++) {
      if(v_old[kk] < g_mem->vth) { 
        if(push_up_flag[kk] == 1) {

          v_new[kk] = v_old[kk] + push_up_amnt[kk];
          
          if(v_new[kk] >= g_mem->vth) {
            v_new[kk] = g_mem->vreset;
            spike_count[kk]++;
            results[threadId].tspike[kk][spike_count[kk]] = t_old;
          }

        }
        else if(push_up_flag[kk] == 0) {
          f0 = g_mem->a - g_mem->b * v_old[kk];
          f1 = g_mem->a - g_mem->b * (v_old[kk] + f0 * 0.5 * g_mem->dt);
          f2 = g_mem->a - g_mem->b * (v_old[kk] + f1 * 0.5 * g_mem->dt);
          f3 = g_mem->a - g_mem->b * (v_old[kk] + f2 * g_mem->dt);
          v_new[kk] = v_old[kk] + g_mem->dt * (f0 + 2 * f1 + 2 * f2 + f3) / 6;
        }
      }
      else if (v_old[kk] >= g_mem->vth) {
        v_new[kk] = g_mem->vreset;
      }
    }
    
    // swap v_old & v_new for next time iteration
    for(kk = 0; kk < N; kk++) {
      v_old[kk] = v_new[kk];
    }

    // Advance time
    t_old = t_new;

  } // Time loop ends

  // Count number of iL-networks where all neurons fire in sync

  InSync_neurons = 1;
  for(kk = 1; kk < N; kk++) {
    // TOASK: What are these "10" and "11"?
    tspike_diff1 = fabs(results[threadId].tspike[0][spike_count[0] - 11] -
                        results[threadId].tspike[kk][spike_count[kk] - 11]);
    tspike_diff2 = fabs(results[threadId].tspike[0][spike_count[0] - 10] -
                        results[threadId].tspike[kk][spike_count[kk] - 10]);
    if(tspike_diff1 < g_mem->tol && tspike_diff2 < g_mem->tol) {
      InSync_neurons++; // count number of neurons firing in sync for the chosen initial condition
    }
  }
  if(InSync_neurons == N) {
    //g_mem->All_sync_count1[params[threadId].iL][params[threadId].ig]++; // count number of ic's that yield All-sync for iL-iG network.
    g_mem->All_sync_count2[params[threadId].iL]++;
    //printf("Number of instances of full sync = %d \n",All_sync_count2[iL]);
    //fprintf(all_sync,"Number of instances of full sync = %d \n",All_sync_count2[0]);
  }

  // TOASK: What is happening here?
  // Write spike time on file
  for(kk=0;kk<N;kk++) {
    tmp1 = 10000*results[threadId].tspike[kk][spike_count[kk]-7];
    tmp2 = 10000*results[threadId].tspike[kk][spike_count[kk]-8];
    tmp3 = 10000*results[threadId].tspike[kk][spike_count[kk]-9];
    tmp4 = 10000*results[threadId].tspike[kk][spike_count[kk]-10];
    tmp5 = 10000*results[threadId].tspike[kk][spike_count[kk]-11];
    tmp6 = 10000*results[threadId].tspike[kk][spike_count[kk]-12];
    tmp7 = 10000*results[threadId].tspike[kk][spike_count[kk]-13];
  //fprintf(spike_time,"%d \t %lu \t %lu \t %lu \t %lu \t %lu \t \%d \n",kk,tmp1,tmp2,tmp3,tmp4,tmp5,flag_unconnctd_graph);
                      //fprintf(spike_time,"%d \t %lu \t %lu \t %lu \t %lu \t %lu \t %lu \t %lu \n",kk,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7);
  }
}


int main() {
  int num_simulations = (NL_max - NL_min) / NL_step * Ng_max * Nic;
  int tmax = 20;

  // Initialize the weight matrices in the GPU memory
  void *d_w;
  cudaMalloc(&d_w, (NL_max - NL_min) * Ng_max / NL_step * N * N * sizeof(double));
  store_weights<<<1, (NL_max - NL_min) / NL_step>>>(d_w);

  // Initialize the global GPU memory
  global_mem g_mem;
  global_mem *d_g_mem;
  cudaMalloc(&d_g_mem, sizeof(global_mem));
  g_mem.dt = 0.0002;
  g_mem.epsilon = 0.01;
  g_mem.vth = 0.8;
  g_mem.vreset = 0;
  g_mem.a = 1;
  g_mem.b = 1;
  g_mem.tol = 0.0001;
  g_mem.Nstep = tmax / g_mem.dt;
  for(int i = 0; i < NL_max - NL_min + 1; ++i) {
    for(int j = 0; j < Ng_max; ++j) {
      g_mem.All_sync_count1[i][j] = 0;
    }
    g_mem.All_sync_count2[i] = 0;
  }
  cudaMemcpy(d_g_mem, &g_mem, sizeof(g_mem), cudaMemcpyHostToDevice);

  // Initialize the parameters
  simulation_params params[num_simulations];
  simulation_params *d_params;
  cudaMalloc(&d_params, sizeof(params));
  for(int iL = 0; iL <= (NL_max - NL_min) / NL_step; ++iL) {
    for(int ig = 0; ig < Ng_max; ++ig) {
      for(int ic = 0; ic < Nic; ++ic) {
        params[iL][ig][ic].iL = iL;
        params[iL][ig][ic].ig = ig;
        params[iL][ig][ic].ic = ic;
      }
    }
  }
  cudaMemcpy(d_params, params, sizeof(params), cudaMemcpyHostToDevice);

  // Perform checks (debug)
  check_weights<<<1, 1>>>(d_params);
  check_g_mem<<<1, 1>>>(d_g_mem);
  check_params<<<1, 1>>>(d_params);

  // Allocate memory for storing results
  simulation_result results[num_simulations];
  simulation_result *d_results;
  cudaMalloc(&d_results, sizeof(results));
  // Start all simulations simultaneously
  simulate<<<1, num_simulations>>>(d_params, d_results, d_g_mem, d_w);

  return 0;
}

/* Check the weights on GPU memory */
__global__ void check_weights(double w[(NL_max - NL_min) * Ng_max / NL_step][N][N]) {
  int n = (NL_max - NL_min) * Ng_max / NL_step;
  for(int i = 0; i < n; ++i) {
    printf("\nnL = %d\tng = %d\n\t", i / Ng_max, i % Ng_max);
    for(int j = 0; j < N; ++j) {
      for(int k = 0; k < N; ++k) {
        printf("%.0lf ");
      }
      printf("\n\t");
    }
  }
}

/* Check the global data on GPU memory */
__global__ void check_g_mem(global_mem *g_mem) {
  printf("dt = %lf\n", dt);
  printf("epsilon = %lf\n", epsilon);
  printf("vth = %lf\n", vth);
  printf("vreset = %lf\n", vreset);
  printf("a = %lf\n", a);
  printf("b = %lf\n", b);
  printf("tol = %lf\n", tol);
  printf("\n");
}

/* Check the simulation parameters on GPU memory */
__global__ void check_params(simulation_params *params) {
  int n = (NL_max - NL_min) * Ng_max * Nic / NL_step;
  for(int i = 0; i < n; ++i) {
    printf("iL = %d\t", params[i].iL);
    printf("ig = %d\t", params[i].ig);
    printf("ic = %d\n", params[i].ic);
  }
  printf("\n");
}