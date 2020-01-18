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
#define N_THREADS_PER_BLOCK 1024

typedef struct {
  double dt, epsilon, vth, vreset, a, b;
  long Nstep;
} global_mem;

typedef struct {
  int iL, nL_break, ig, ic;
  double w[N][N];
} simulation_params;

typedef struct {
  int ic, iL, nL_break, ig;
  double v_init[N];
  double tspike[N][MAXCOL];
} simulation_result;

int synaptic_weighs_connected_network(double w[][N], int nL);

__global__ void simulate(simulation_params *params, simulation_result *results, global_mem *g_mem) {
  // "threadId" is used as an index into the arrays "params" and "results".
  // Everything that was being written to a file is now returned in a struct.
  int blockId = blockIdx.z * gridDim.x * gridDim.y + 
                blockIdx.y * gridDim.x + 
                blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;
  
  results[threadId].ic = params[threadId].ic;
  results[threadId].iL = params[threadId].iL;
  results[threadId].nL_break = params[threadId].nL_break;
  results[threadId].ig = params[threadId].ig;
  
  int i, k kk, t_old, t_new;
  int spike_count[N], spike[N], push_up_flag[N];
  double f0, f1, f2, f3, f4;
  double v_old[N], v_new[N], push_up_amnt[N];
  double v_initnew[20]= {0.00778832, 0.355919, 0.426307, 0.183062,
                         0.272762, 0.532633, 0.339171, 0.242097,
                         0.523038, 0.638838, 0.632368, 0.778564,
                         0.110892, 0.347691, 0.696286, 0.791943,
                         0.5257, 0.127494, 0.716965, 0.151006};

  // Generate initial state
  for(kk = 0; kk < N; kk++) {
    results[threadId].v_init[kk] = rand() % (g_mem->vth*1000);
    results[threadId].v_init[kk] = results[threadId].v_init[kk] / 100000;
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
                             (g_mem->epsilon) * params.w[kk][k] * spike[k];
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
  for(kk=1;kk<N;kk++) {
    tspike_diff1 = fabs(tspike[0][spike_count[0]-11]-tspike[kk][spike_count[kk]-11]);
    tspike_diff2 = fabs(tspike[0][spike_count[0]-10]-tspike[kk][spike_count[kk]-10]);
    if(tspike_diff1 < tol && tspike_diff2 < tol) {
      InSync_neurons++; // count number of neurons firing in sync for the chosen initial condition
      //printf("%d \n",InSync_neurons);
    }
  }
  if(InSync_neurons == N) {
    //All_sync_count1[iL][ig]++; // count number of ic's that yield All-sync for iL-iG network.
    All_sync_count2[iL]++;
    //printf("Number of instances of full sync = %d \n",All_sync_count2[iL]);
    //fprintf(all_sync,"Number of instances of full sync = %d \n",All_sync_count2[0]);
  }


  // Write spike time on file

  for(kk=0;kk<N;kk++) {
    tmp1 = 10000*tspike[kk][spike_count[kk]-7];
    tmp2 = 10000*tspike[kk][spike_count[kk]-8];
    tmp3 = 10000*tspike[kk][spike_count[kk]-9];
    tmp4 = 10000*tspike[kk][spike_count[kk]-10];
    tmp5 = 10000*tspike[kk][spike_count[kk]-11];
    tmp6 = 10000*tspike[kk][spike_count[kk]-12];
    tmp7 = 10000*tspike[kk][spike_count[kk]-13];
  //fprintf(spike_time,"%d \t %lu \t %lu \t %lu \t %lu \t %lu \t \%d \n",kk,tmp1,tmp2,tmp3,tmp4,tmp5,flag_unconnctd_graph);
                      //fprintf(spike_time,"%d \t %lu \t %lu \t %lu \t %lu \t %lu \t %lu \t %lu \n",kk,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7);
  }
  for(kk=0;kk<N;kk++) {
    for(i=2;i<=spike_count[kk];i++)	{	
      fprintf(spike_time,"%f  ",tspike[kk][i-1]);
    }
    fprintf(spike_time,"\n");
  }
}

int main() {
  /* TODO: Write driver code including the iL, ig and ic loops */
  return 0;
}