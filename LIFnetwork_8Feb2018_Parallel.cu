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
  // Generate initial state
  for(kk=0;kk<N;kk++) {
    tempp = vth*1000;
    v_init[kk] = rand()%tempp; 
    v_init[kk] = v_init[kk]/1000 ;
    v_old[kk] = v_init[kk];
    fprintf(init_condn,"%d \t %f \n",kk, v_init[kk]);
  }	

  double v_initnew[20]= {0.00778832,0.355919,0.426307, 0.183062,0.272762,0.532633,0.339171,0.242097,0.523038,0.638838,0.632368,0.778564,0.110892,0.347691,0.696286,0.791943,0.5257,0.127494,0.716965,0.151006};

  for(kk=0;kk<N;kk++){		
    v_init[kk] = v_initnew[kk];
    v_old[kk] = v_init[kk];
  }

  // initialize arrays
  for(k=0;k<N;k++) {
    spike_count[k] = 0; //keeps a count of the number spikes in neuron k so far
  }

  for(k=0;k<N;k++){
    for(i=0;i<MAXCOL;i++){
			tspike[k][i] = 0; // counts the spike time of "i_th" spike of neuron number "k"
    }
  }

  t_old = 0;
  for(i=1;i<Nstep;i++) { 	

    t_new = i*dt;

    // Identify (1) the neurons that spiked in previous time step, (2) time of all the spikes of each neuron
    // (3) total number of spikes in each neuron so far
    for(kk=0;kk<N;kk++) {
      if(v_old[kk]>=vth) {
        spike[kk]=1; // if neuron spiked
        spike_count[kk]++ ;
        tspike[kk][spike_count[kk]] = t_old;
        //fprintf(spike_time,"%d \t %f \n",kk,t_old);
      }
      else {	
        spike[kk]=0; // if neuron did not spike
      }
    }

    // Find voltage push-up amount for each neuron (if atleast one neuron other than itself spiked)
    for(kk=0;kk<N;kk++) {
      push_up_amnt[kk]=0; // initialize these arrays at every time step
      push_up_flag[kk] =0;
    }
    for(kk=0;kk<N;kk++) {
      for(k=0;k<N;k++) {	
        if(k != kk) {
          if(spike[kk] != 1 && spike[k]==1)
          {
            push_up_amnt[kk] = push_up_amnt[kk] + epsilon*w[kk][k]*spike[k];
            push_up_flag[kk] = 1;
          }
        }
      }
    }

    // Finally update voltages of each neuron - using Euler method if no neuron fired & by pushing up the
    // voltage value by push_up_amnt if some neurons fired.
    for(kk=0;kk<N;kk++) {
      if(v_old[kk] < vth) { 
        if(push_up_flag[kk] ==1) {
    
          v_new[kk] = v_old[kk] + push_up_amnt[kk];
          
          if(v_new[kk] >= vth) {
            v_new[kk] = vreset;
            spike_count[kk]++ ;
            tspike[kk][spike_count[kk]] = t_old;
            //fprintf(spike_time,"%d \t %f \n",kk,t_old);
          }

        }
        else if(push_up_flag[kk] == 0) {
          f0 = a-b*v_old[kk];
          f1 = a-b*(v_old[kk]+f0*0.5*dt);
          f2 = a-b*(v_old[kk]+f1*0.5*dt);
          f3 = a-b*(v_old[kk]+f2*dt);
          v_new[kk] = v_old[kk]+dt*(f0+2*f1+2*f2+f3)/6;
        }
      }
      else if (v_old[kk] >= vth) {
        v_new[kk] = vreset;
      }
    }
    
    // swap v_old & v_new for next time iteration
    for(kk=0;kk<N;kk++) {
      v_old[kk] = v_new[kk];
    }

    t_old = t_new;

  }
}

int main() {
  
  return 0;
}