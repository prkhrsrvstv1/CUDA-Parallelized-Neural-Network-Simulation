#include <time.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

#define N 100
#define MAXCOL 100
#define Nic 5
#define NL_min 0
#define NL_max 70
#define NL_step 1
#define Ng_max 10
#define N_THREADS_PER_BLOCK 35
#define N_BLOCKS 100
#define FILENAME "results.txt"


typedef struct {
  int All_sync_count1[NL_max-NL_min][Ng_max];
  int All_sync_count2[NL_max-NL_min];
} global_mem;

typedef struct {
  unsigned short ic, iL, nL_break, ig;
  unsigned short spike_count[N];
  double v_init[N];
  double tspike[N * MAXCOL];
} simulation_result;

/************************ DEBUG FUNCTIONS ************************/
/* Check the weights on GPU memory *
__global__ void check_weights(double w[(NL_max - NL_min) * Ng_max / NL_step][N][N]) {
  printf("hi_check_weights\n");
  int n = (NL_max - NL_min) / NL_step * Ng_max;

  for(int i = 0; i < n; ++i) {
    printf("\nnL = %d\tng = %d\n\t", NL_min + NL_step * i / Ng_max, i % Ng_max);
    for(int j = 0; j < N; ++j) {
      for(int k = 0; k < N; ++k) {
        printf("%.2lf ", w[i][j][k]);
      }
      printf("\n\t");
    }
  }
  printf("\n");
}

/* Check the global data on GPU memory *
__global__ void check_g_mem(global_mem *g_mem) {
  double sum1 = 0.0, sum2 = 0.0;
  for(int i = 0; i < NL_max - NL_min; ++i) {
    for(int j = 0; j < Ng_max; ++j) {
      sum1 += g_mem->All_sync_count1[i][j];
    }
    sum2 += g_mem->All_sync_count2[i];
  }
  printf("sum1 = %f\nsum2 = %f\n", sum1, sum2);
}
/*******************************************************************/


/* Generate a adjacency matrix for a coonnected graph with nL edges missing */
__device__ unsigned short synaptic_weights_connected_network(double w[][N], unsigned short nL, curandState *rand_state) {

  unsigned short i,j,k,kk,neuron1,neuron2;
  double w_flag[N][N];
  unsigned short syn_to_remove, tot_syn_removed;
  short connected_nodes[N];
  unsigned short current_ptr, endptr, parent_node;
  unsigned short flag_connected = 0;
  unsigned short flag_already_connected;

  // GENERATE AN ALL-TO-ALL NETWORK ************************************************************************
  for(i = 0; i < N; i++) {
    for(j = 0; j < N; j++) {
      if(j != i){
        w[i][j] = 1;
      }
      else if(j == i){
        w[i][j] = 0;
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
    neuron1 = curand(rand_state) % N;
    neuron2 = curand(rand_state) % N;
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

  //w[1][0] = w[0][1]; w[1][1] = 0; w[1][2] = 1; w[1][3] = 0; w[1][4] = 0; w[1][5] = 1;

  //w[2][0] = w[0][2]; w[2][1] = w[1][2]; w[2][2] = 0; w[2][3] = 0; w[2][4] = 1; w[2][5] = 0;

  //w[3][0] = w[0][3]; w[3][1] = w[1][3]; w[3][2] = w[2][3]; w[3][3] = 0; w[3][4] = 0; w[3][5] = 0;

  //w[4][0] = w[0][4]; w[4][1] = w[1][4]; w[4][2] = w[2][4]; w[4][3] = w[3][4]; w[4][4] = 0; w[4][5] = 1;

  //w[5][0] = w[0][5]; w[5][1] = w[1][5]; w[5][2] = w[2][5]; w[5][3] = w[3][5]; w[5][4] = w[4][5]; w[5][5] = 0;

  //w[0][0] = 0; w[0][1] = 0; w[0][2] = 1; w[0][3]=0;
  //w[1][0] = w[0][1]; w[1][1] = 0;  w[1][2] = 1; w[1][3] =0;
  //w[2][0]=w[0][2]; w[2][1]=w[1][2]; w[2][2] =0; w[2][3] = 1;
  //w[3][0] = w[0][3]; w[3][1] = w[1][3]; w[3][2] = w[2][3]; w[3][3]=0;

  // for(k = 0; k < N; k++) {
  //   for(kk = 0; kk < N; kk++) {
  //     w_flag[k][kk] = 0; // w_flag[k][kk] is changed to value 1, if the synapse between k --> kk is removed
  //   }
  // }

  connected_nodes[0] = 0;
  for(i=1;i<N;i++) {
    connected_nodes[i] = -1;
  }
  current_ptr = 0;
  endptr = 0;  // points towards the last non-zero element in the connected_nodes array

  while(current_ptr <= endptr) {

    for(i = 0; i < N; i++) {
      parent_node = connected_nodes[current_ptr];

      flag_already_connected = 0;

      for(j = 0; j <= endptr; j++) {
        if(connected_nodes[j] == i) {
          flag_already_connected = 1;
        }
      }

      if(w[parent_node][i] == 1) {
        if(w_flag[parent_node][i] == 0) {
          if(flag_already_connected ==0) {
            endptr ++;
            connected_nodes[endptr] = i; // stores node numbers connected to parent_node

            w_flag[parent_node][i] = 1;
            w_flag[i][parent_node] = w_flag[parent_node][i]; //links already visited

            //printf("i= %d \t endptr= %d \t current_ptr= %d \t connected_nodes[endptr] = %d \n",i, endptr,current_ptr,connected_nodes[endptr]);
          }
        }
      }

      if (i == N-1) {
        current_ptr++;
      }
    }
  }

  if(endptr == N-1) {
    flag_connected = 1;
  }

  return flag_connected;
}

/* Create weight matrices in GPU memory */
__global__ void store_weights(double w[(NL_max - NL_min) / NL_step * Ng_max][N][N]) {
  unsigned short threadId = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned short nL_break = NL_min + threadId * NL_step;
  unsigned short flag_connected;
  curandState rand_state;
  curand_init(1234, threadId, 0, &rand_state);
  for(unsigned short i = 0; i < Ng_max; ++i) {
    flag_connected = 0;
    do {
      flag_connected = synaptic_weights_connected_network(w[threadId * Ng_max + i], nL_break, &rand_state);
    } while(flag_connected == 0);
  }
}

/* Run a simulation on a single thread */
__global__ void simulate(simulation_result *results, global_mem *g_mem, double w[(NL_max - NL_min) / NL_step * Ng_max][N][N]) {
  // "threadId" is used as an index into the array "results".
  // Everything that was being written to a file is now returned in a struct.
  unsigned short threadId = blockIdx.x * blockDim.x + threadIdx.x;
  // Initialize and seed the random number generator
  curandState rand_state;
  curand_init(threadId, clock(), clock(), &rand_state);

  double tmax = 20;
  double dt = 0.0002;
  double epsilon = 0.01;
  double vth = 0.8;
  double vreset = 0;
  double a = 1;
  double b = 1;
  double tol = 0.0001;
  int Nstep = tmax / dt;


  unsigned short ic = threadId % Nic;
  unsigned short ig = (threadId / Nic) % Ng_max;
  unsigned short iL = threadId / Ng_max / Nic;
  unsigned short nL_break = NL_min + iL * NL_step;
  results[threadId].ic = ic;
  results[threadId].ig = ig;
  results[threadId].iL = iL;
  results[threadId].nL_break = nL_break;

  int i;
  unsigned short k, kk, InSync_neurons;
  unsigned short spike[N], push_up_flag[N];
  double f0, f1, f2, f3, tspike_diff1, tspike_diff2, t_old, t_new;
  double v_old[N], v_new[N], push_up_amnt[N];
  // double v_initnew[100]= {0.1545367758770665, 0.814818668976894, 0.15320113199547414, 0.8353524225981629, 0.08115890455440067, 0.6914756325608367, 0.4130575136157111, 0.5278299763853765, 0.2812216969669379, 0.8062893532936973, 0.9026514070819015, 0.6496189902535245, 0.6286630367202969, 0.6171265038631547, 0.472005565894945, 0.43981531433376, 0.8449193307307433, 0.3499655732796455, 0.6064637293486522, 0.1567131568957726, 0.6917890946540877, 0.19314656121526463, 0.9715334462829239, 0.42821872654614646, 0.5153519308836192, 0.8849979650599988, 0.6757089505722944, 0.31767924448674467, 0.2910320632769062, 0.32862537004994197, 0.45168148961810184, 0.01955708613009799, 0.5696484846788225, 0.450835587565686, 0.026054486371280938, 0.35039306479694443, 0.4040846812243857, 0.27342993028260487, 0.5638358124122043, 0.9484997135038367, 0.4077636621202826, 0.8220935863179847, 0.7196517781502417, 0.5968801478996293, 0.17909455403785213, 0.9071518551971325, 0.49350749777889813, 0.8002803025938409, 0.3071891631672753, 0.5367924012551228, 0.8628384065372916, 0.9147597382639411, 0.5859467778984498, 0.506728558827792, 0.5444346202867876, 0.7105452431393048, 0.8833280213387779, 0.7101823916271959, 0.21378218672881877, 0.2647380984685085, 0.8051689609566608, 0.636661266440235, 0.1284215317086359, 0.8991055384060852, 0.9185260634481671, 0.7505310205211034, 0.5449904790914537, 0.8418539582522988, 0.8227024116656272, 0.8206769102729885, 0.5615504438601934, 0.9070762107580452, 0.37619234543451996, 0.23085180280640882, 0.6623891864245589, 0.9806074893915904, 0.8067560379883594, 0.9895526050531294, 0.5548342062752014, 0.818488769718889, 0.48622692029833214, 0.6501553126075313, 0.3176597622855678, 0.9742850850234102, 0.6065112069910525, 0.37288262643468995, 0.074431646812396, 0.194162041772725, 0.021779459371789267, 0.2856071586947684, 0.5653325199766001, 0.10132723526598542, 0.7041397023518559, 0.6412510211401311, 0.061293406975714726, 0.2728425423344597, 0.6529094748027036, 0.6152282218769618, 0.2633952283711999, 0.44178953896737416};

  // Generate initial state
  for(kk = 0; kk < N; kk++) {
    results[threadId].v_init[kk] = curand_uniform_double(&rand_state) * (vth);
    v_old[kk] = results[threadId].v_init[kk];
  }
  // for(kk = 0; kk < N; kk++) {
  //   results[threadId].v_init[kk] = v_initnew[kk];
  //   v_old[kk] = results[threadId].v_init[kk];
  // }

  // initialize arrays
  memset(results[threadId].spike_count, 0, N * sizeof(unsigned short));
  memset(results[threadId].tspike, 0, N * MAXCOL * sizeof(double));

  // Time loop begins
  t_old = 0;
  for(i = 1; i < Nstep; i++) {
    t_new = i*dt;

    // Identify (1) the neurons that spiked in previous time step, (2) time of all the spikes of each neuron
    // (3) total number of spikes in each neuron so far
    for(kk = 0; kk < N; kk++) {
	  push_up_amnt[kk] = 0; // initialize these arrays at every time step
      push_up_flag[kk] = 0;
      if(v_old[kk] >= vth) {
        spike[kk] = 1; // if neuron spiked
        results[threadId].spike_count[kk]++;
        results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]] = t_old;
        // printf("%f\n", results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]]);
      }
      else {
        spike[kk] = 0; // if neuron did not spike
      }
    }

    // Find voltage push-up amount for each neuron (if atleast one neuron other than itself spiked)
    // for(kk = 0; kk < N; kk++) {
    //   push_up_amnt[kk] = 0; // initialize these arrays at every time step
    //   push_up_flag[kk] = 0;
    // }
    for(kk = 0; kk < N; kk++) {
      for(k = 0; k < N; k++) {
        if(k != kk && spike[kk] != 1 && spike[k]==1) {
          push_up_amnt[kk] = push_up_amnt[kk] +
                             (epsilon) * w[threadId % Nic][kk][k] * spike[k];
          push_up_flag[kk] = 1;
        }
      }
            if(v_old[kk] < vth) {
        if(push_up_flag[kk] == 1) {

          v_new[kk] = v_old[kk] + push_up_amnt[kk];

          if(v_new[kk] >= vth) {
            v_new[kk] = vreset;
            results[threadId].spike_count[kk]++;
            results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]] = t_old;
            // printf("%f\n", results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]]);
          }

        }
        else if(push_up_flag[kk] == 0) {
          f0 = a - b * v_old[kk];
          f1 = a - b * (v_old[kk] + f0 * 0.5 * dt);
          f2 = a - b * (v_old[kk] + f1 * 0.5 * dt);
          f3 = a - b * (v_old[kk] + f2 * dt);
          v_new[kk] = v_old[kk] + dt * (f0 + 2 * f1 + 2 * f2 + f3) / 6;
        }
      }
      else if (v_old[kk] >= vth) {
        v_new[kk] = vreset;
      }
      // swap v_old & v_new for next time iteration
      v_old[kk] = v_new[kk];
    }

    // Finally update voltages of each neuron - using Euler method if no neuron fired & by pushing up the
    // voltage value by push_up_amnt if some neurons fired.
    // for(kk = 0; kk < N; kk++) {
    //   if(v_old[kk] < vth) {
    //     if(push_up_flag[kk] == 1) {

    //       v_new[kk] = v_old[kk] + push_up_amnt[kk];

    //       if(v_new[kk] >= vth) {
    //         v_new[kk] = vreset;
    //         results[threadId].spike_count[kk]++;
    //         results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]] = t_old;
    //         // printf("%f\n", results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]]);
    //       }

    //     }
    //     else if(push_up_flag[kk] == 0) {
    //       f0 = a - b * v_old[kk];
    //       f1 = a - b * (v_old[kk] + f0 * 0.5 * dt);
    //       f2 = a - b * (v_old[kk] + f1 * 0.5 * dt);
    //       f3 = a - b * (v_old[kk] + f2 * dt);
    //       v_new[kk] = v_old[kk] + dt * (f0 + 2 * f1 + 2 * f2 + f3) / 6;
    //     }
    //   }
    //   else if (v_old[kk] >= vth) {
    //     v_new[kk] = vreset;
    //   }
    //   // swap v_old & v_new for next time iteration
    //   v_old[kk] = v_new[kk];
    // }


    // Advance time
    t_old = t_new;

  } // Time loop ends

  // Count number of iL-networks where all neurons fire in sync
  InSync_neurons = 1;
  for(kk = 1; kk < N; kk++) {
    // TOASK: What are these "10" and "11"?
    tspike_diff1 = fabs(results[threadId].tspike[0 * MAXCOL + results[threadId].spike_count[0] - 11] -
                        results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk] - 11]);
    tspike_diff2 = fabs(results[threadId].tspike[0 * MAXCOL + results[threadId].spike_count[0] - 10] -
                        results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk] - 10]);
    if(tspike_diff1 < tol && tspike_diff2 < tol) {
      InSync_neurons++; // count number of neurons firing in sync for the chosen initial condition
    }
  }
  if(InSync_neurons == N) {
    //g_mem->All_sync_count1[iL][ig]++; // count number of ic's that yield All-sync for iL-iG network.
    g_mem->All_sync_count2[iL]++;
    //printf("Number of instances of full sync = %d \n",All_sync_count2[iL]);
    //fprintf(all_sync,"Number of instances of full sync = %d \n",All_sync_count2[0]);
  }

  // TOASK: What is happening here?
  // Write spike time on file
  /*for(kk=0;kk<N;kk++) {
    tmp1 = 10000*results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]-7];
    tmp2 = 10000*results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]-8];
    tmp3 = 10000*results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]-9];
    tmp4 = 10000*results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]-10];
    tmp5 = 10000*results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]-11];
    tmp6 = 10000*results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]-12];
    tmp7 = 10000*results[threadId].tspike[kk * MAXCOL + results[threadId].spike_count[kk]-13];
  //fprintf(spike_time,"%d \t %lu \t %lu \t %lu \t %lu \t %lu \t \%d \n",kk,tmp1,tmp2,tmp3,tmp4,tmp5,flag_unconnctd_graph);
                      //fprintf(spike_time,"%d \t %lu \t %lu \t %lu \t %lu \t %lu \t %lu \t %lu \n",kk,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7);
  }*/
  printf("Thread #%d finished\n", threadId);
}


int main() {

  unsigned short num_simulations = (NL_max - NL_min) / NL_step * Ng_max * Nic;

  printf("Running %d simulations with N = %d, NL_max = %d, Ng_max = %d, Nic = %d\n\n", num_simulations, N, NL_max, Ng_max, Nic);

  // Initialize the weight matrices in the GPU memory
  void *d_w;
  cudaMalloc(&d_w, (NL_max - NL_min) * Ng_max / NL_step * N * N * sizeof(double));

  store_weights<<<1, (NL_max - NL_min) / NL_step>>>((double (*)[N][N])d_w);
  
  // Initialize the global GPU memory
  global_mem g_mem;
  global_mem *d_g_mem;
  cudaMalloc(&d_g_mem, sizeof(global_mem));
  for(unsigned short i = 0; i < NL_max - NL_min; ++i) {
    for(unsigned short j = 0; j < Ng_max; ++j) {
      g_mem.All_sync_count1[i][j] = 0;
    }
    g_mem.All_sync_count2[i] = 0;
  }
  cudaMemcpy(d_g_mem, &g_mem, sizeof(g_mem), cudaMemcpyHostToDevice);
  
  // Allocate memory for storing results
  simulation_result *results = (simulation_result *) malloc(sizeof(simulation_result) * num_simulations);
  simulation_result *d_results;
  cudaMalloc(&d_results, sizeof(simulation_result) * num_simulations);
  
  // Start all simulations simultaneously
  simulate<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_results, d_g_mem, (double (*)[N][N])d_w);
  
  // Retrieve the results back from GPU
  cudaMemcpy(results, d_results, sizeof(simulation_result) * num_simulations, cudaMemcpyDeviceToHost);
  cudaMemcpy(&g_mem, d_g_mem, sizeof(g_mem), cudaMemcpyDeviceToHost);

  // Open a file to store the results
  FILE *file = fopen(FILENAME, "w");

  // Write the results to file
  for(int i = 0; i < num_simulations; ++i) {
    unsigned short ic = i % Nic;
    unsigned short ig = (i / Nic) % Ng_max;
    unsigned short iL = i / Ng_max / Nic;
    unsigned short nL_break = NL_min + iL * NL_step;
    fprintf(file, "\n------------------------------------------------------------------\n");
    // Simulation parameters
    fprintf(file, "\n\n%d. nL_break = %d\tig = %d\tic = %d :\n\n\t", i+1, nL_break, ig, ic);
    // TODO: Weight matrix
    
    // Initial voltages
    fprintf(file, "Initial voltages:\n\t");
    for(unsigned short j = 0; j < N; ++j) {
      fprintf(file, "%f ", results[i].v_init[j]);
    }
    // All_sync_count2
    fprintf(file, "\n\n\tAll_sync_count2[%d]: %d\n\n\t", iL, g_mem.All_sync_count2[iL]);
    // Spike times
    fprintf(file, "Spike times:\n\t");
    for(unsigned short j = 0; j < N; ++j) {
      for(unsigned short k = 1; k <= results[i].spike_count[j]; ++k) {
        fprintf(file, "%f ", results[i].tspike[j * MAXCOL + k]);
      }
      fprintf(file, "\n\t");
    }
  }

  // Clean-up
  fclose(file);
  free(results);
  cudaFree(d_w);
  cudaFree(d_g_mem);
  cudaFree(d_results);
  return 0;
}
