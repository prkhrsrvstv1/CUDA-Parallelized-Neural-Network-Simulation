#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// ************************************************************************************************************************
// Time advance LIF neurons on the network. The network is constructed here.
//Construct a Raster plot of firing of all neurons.
//*************************************************************************************************************************

//int tot_syn(int n);

#define N 20
#define MAXCOL 10000
#define NL_min 0
#define NL_max 0
#define NL_step 1
#define Ng_max 1

// void synaptic_weights(double w[][N],int );
//void synaptic_weights_connected_network(double w[][N],int nL, int flag_connected);
int synaptic_weights_connected_network(double w[][N],int nL);


int main(){

	double dt = 0.0002; 
	double tmax = 20;
	//double epsilon = 0.06/N;
	double epsilon = 0.01 ;
	double vth = 0.8 ;
	double vreset = 0 ;
	double a = 1;
	double b = 1 ;
	//double tol = 0.0001 ;
	double tol = 0.0001 ;
	
	double f0,f1,f2,f3,f4;
	
	long Nstep = tmax/dt;

	int ic;
	int Nic = 1; 

	double t_old, t_new;

	double w[N][N], v_old[N], v_new[N], v_init[N], w_row[N];

	int spike[N], push_up_flag[N];
	double push_up_amnt[N];
	double tspike[N][MAXCOL], ISI[N];
	int spike_count[N];
	int syn_remove_flag;

	int i, j, k, kk ;

	int iL, ig, iL_max ;

	int InSync_neurons;
	double tspike_diff1, tspike_diff2;
	int All_sync_count1[NL_max-NL_min+1][Ng_max], All_sync_count2[NL_max-NL_min+1];
	int nL_break;
	long tmp1, tmp2, tmp3, tmp4 , tmp5 , tmp6, tmp7;
	int flag_unconnctd_graph, N_connctd_graph[NL_max-NL_min+1] ;
	int flag_connected, nL ;
	int tempp;

	// open files
	FILE *Vvst;
	FILE *raster;
	FILE *end_spike_time;
	FILE *init_condn;
	FILE *debug;
	FILE *ISIfile;
	FILE *spike_time;
	FILE *all_sync ;
	FILE *weights;
	FILE *connected_graphID;

	weights= fopen("fweights.txt", "w");
	Vvst= fopen("V_vs_t.txt", "w");
	raster= fopen("fraster.txt", "w");
	end_spike_time = fopen("fend_spike_time.txt","w");
	init_condn = fopen("finit_condn.txt","w");
	debug = fopen("fdebug.txt","w");
	ISIfile = fopen("fISI.txt","w");
	spike_time = fopen("fspike_time.txt","w");
	all_sync = fopen("fall_sync.txt","w");
	connected_graphID = fopen("fconnected_graphID","w");
	
	printf("Nstep = %lu\n",Nstep);
	
	srand (time(NULL));
	//srand(0.9);

	// initialize array
	iL_max = (NL_max - NL_min)/NL_step ;
	for(iL = 0;iL<=iL_max;iL++)
	{
		nL_break = NL_min + iL*NL_step;

		All_sync_count2[iL] = 0; // will count the number of iL-networks where all neurons fire in sync
		N_connctd_graph[iL] = 0;
	}

	for(ig = 0;ig<Ng_max;ig++)
	{
		for(iL = 0;iL<=NL_max-NL_min;iL++)
		{
		All_sync_count1[iL][ig] = 0;
		}
	}


	// *********************************************************************************************************************************
	// Start off now. First knock off "iL" links randomly "ig" times. Each resulting network is simulated for "ic" diffnt initial condns
	// *********************************************************************************************************************************

	for(iL = 0;iL<=iL_max;iL++) //NL_max is the maximum number of links that we will break.
	{   
	
	nL_break = NL_min + iL*NL_step;


	for(ig = 0;ig<Ng_max;ig++)
	{
	

	printf("iL = %d \t nl_break= %d \t ig = %d \t Ng_max= %d \n",iL, nL_break,ig,Ng_max);

	// Generate the network by randomly removing iL synapses.

	//synaptic_weights(w,nL_break) ; // Remove iL links from the N-graph randomly to get a ig-graph

	flag_connected = 0;
        do
	{
		//synaptic_weights_connected_network(w,nL_break,flag_connected);
		flag_connected = synaptic_weights_connected_network(w,nL_break);
		// fprintf(connected_graphID,"nL_break=%d \t ig = %d \t flag_connected = %d \n",nL_break,ig,flag_connected);
		//printf("nL_break=%d \t ig = %d \t flag_connected = %d \n",nL_break,ig,flag_connected);
	}while(flag_connected == 0);

	//fprintf(connected_graphID,"Generated a connected graph with nL_break = %d \t ig = %d \n",nL_break,ig);

	
	// write the weight matrix on a file
	fprintf(weights,"\n \n");
	fprintf(weights,"iL = %d \t nl_break= %d \t ig = %d \n",iL, nL_break,ig);

	for(kk=0;kk<N;kk++){
		for(k=0;k<N;k++){
		fprintf(weights,"%f\t",w[kk][k]);
		}
		fprintf(weights,"\n");
	} 

	
	for(ic=0;ic<Nic;ic++) // Run simulation for Nic number of initial conditions.
	{

		// some print statements for every new initial state
		fprintf(raster,"\n \n");
		fprintf(raster,"#Initial condition number=%d",ic); // raster plot
		fprintf(init_condn,"\n \n");
		fprintf(init_condn,"iL = %d \t nL_break = %d \t ig = %d \t ic=%d \n",iL,nL_break,ig,ic);
		fprintf(ISIfile,"\n \n");
		fprintf(ISIfile,"#Initial condition number=%d\n",ic);
		fprintf(spike_time,"\n \n");
		fprintf(spike_time,"#iL=%d \t nL_break=%d \t ig=%d \t ic=%d \n",iL,nL_break,ig,ic);
		fprintf(end_spike_time,"\n \n");
		fprintf(end_spike_time,"#Initial condition number=%d\n",ic);

	//	printf("00 \n");

		// Generate initial state
		for(kk=0;kk<N;kk++)
		{
		  //v_init[kk] = rand()%1000; 
		 // v_init[kk] = v_init[kk]/1000 ;
		 // v_old[kk] = v_init[kk];
		 // fprintf(init_condn,"%d \t %f \n",kk, v_init[kk]);

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
		for(k=0;k<N;k++)
		{
		spike_count[k] = 0; // keeps a count of the number spikes in neuron k so far.
		}

		for(k=0;k<N;k++){
			for(i=0;i<MAXCOL;i++){
			tspike[k][i] = 0; // counts the spike time of "i_th" spike of neuron number "k" 
			}
		}


		// Time Loop Starts
		t_old = 0;
		for(i=1;i<Nstep;i++) // loop over time
		{ 	

			t_new = i*dt;

			//Identify (1) the neurons that spiked in previous time step, (2) time of all the spikes of each neuron
			// (3) total number of spikes in each neuron so far
			for(kk=0;kk<N;kk++)
			{
				if(v_old[kk]>=vth)
				{
					spike[kk]=1; // if neuron spiked
					spike_count[kk]++ ;
					tspike[kk][spike_count[kk]] = t_old;
					//fprintf(spike_time,"%d \t %f \n",kk,t_old);
				}
				else
				{	
					spike[kk]=0; // if neuron did not spike
				}
			}

			// Find voltage push-up amount for each neuron (if atleast one neuron other than itself spiked)
			for(kk=0;kk<N;kk++){
				push_up_amnt[kk]=0; // initialize these arrays at every time step
				push_up_flag[kk] =0;
			}
			for(kk=0;kk<N;kk++)
			{
				for(k=0;k<N;k++)
				{	
					if(k != kk)
					{
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
			for(kk=0;kk<N;kk++)
			{
				if(v_old[kk] < vth)
				{ 
					if(push_up_flag[kk] ==1)
					{
			
						v_new[kk] = v_old[kk] + push_up_amnt[kk] ;
						
						if(v_new[kk] >= vth)
						{
							v_new[kk] = vreset;
							spike_count[kk]++ ;
							tspike[kk][spike_count[kk]] = t_old;
							//fprintf(spike_time,"%d \t %f \n",kk,t_old);
						}

					}
					else if(push_up_flag[kk] == 0)
					{
						f0 = a-b*v_old[kk];
						f1 = a-b*(v_old[kk]+f0*0.5*dt);
						f2 = a-b*(v_old[kk]+f1*0.5*dt);
						f3 = a-b*(v_old[kk]+f2*dt);
						v_new[kk] = v_old[kk]+dt*(f0+2*f1+2*f2+f3)/6;
					}
				}
				else if (v_old[kk] >= vth){ 
					       v_new[kk] = vreset ;
				}
				
			}
			
			// swap v_old & v_new for next time iteration
			for(kk=0;kk<N;kk++)
			{
				v_old[kk] = v_new[kk];
			}

			t_old = t_new;


		} //end of time loop & update of each neuron


		// Count number of iL-networks where all neurons fire in sync

		InSync_neurons = 1;
		for(kk=1;kk<N;kk++)
		{
			tspike_diff1 = fabs(tspike[0][spike_count[0]-11]-tspike[kk][spike_count[kk]-11]);
			tspike_diff2 = fabs(tspike[0][spike_count[0]-10]-tspike[kk][spike_count[kk]-10]);
			if(tspike_diff1 < tol && tspike_diff2 < tol)
			{
				InSync_neurons++; // count number of neurons firing in sync for the chosen initial condition
				//printf("%d \n",InSync_neurons);
			}
				
		}
		if(InSync_neurons == N)
		{
			//All_sync_count1[iL][ig]++; // count number of ic's that yield All-sync for iL-iG network.
			All_sync_count2[iL]++;
			//printf("Number of instances of full sync = %d \n",All_sync_count2[iL]);
			//fprintf(all_sync,"Number of instances of full sync = %d \n",All_sync_count2[0]);
		}


		// write spike time on file

		for(kk=0;kk<N;kk++)
		{
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
		for(kk=0;kk<N;kk++)
		{
			for(i=2;i<=spike_count[kk];i++)	
			{	

				fprintf(spike_time,"%f  ",tspike[kk][i-1]);
			}
			fprintf(spike_time,"\n");
		}



	} //end of initial condition loop

	} // end of ig loop


	} //end of iL loop

	// write on file - For each iL network how many times did we find all neurons in sync
	/*for(iL=0;iL<=iL_max;iL++)
	{
		nL_break = NL_min + iL*NL_step;
		//fprintf(all_sync,"%d \t %d \t %d \t  %d \n",iL,nL_break, All_sync_count2[iL],N_connctd_graph[iL]*Nic);
		//printf("%d \t %d \t %d \t  %d \n",iL,nL_break, All_sync_count2[iL],N_connctd_graph[iL]*Nic);
		fprintf(all_sync,"%d \t %d \t %d \t  %d \n",iL,nL_break, All_sync_count2[iL],Ng_max*Nic);
		//printf("%d \t %d \t %d \t  %d \n",iL,nL_break, All_sync_count2[iL],Ng_max*Nic);
		//printf("%d \t %d \t %d \t  %d \n",iL,nL_break, All_sync_count2[iL],Ng_max*Nic);
		//printf("%d \t %d \t %d \t  %d \n",iL,nL_break, All_sync_count2[iL],(4*iL+1)*Nic);
		//fprintf(all_sync,"%d \t %d \t %d \t  %d \n",iL,nL_break, All_sync_count2[iL],(4*iL+1)*Nic);
	}*/


	fclose(Vvst);
	fclose(end_spike_time);
	fclose(raster);
	//fclose(debug);
	fclose(init_condn);
	fclose(ISIfile);
	fclose(all_sync);
	fclose(weights);
	fclose(connected_graphID);

	

} // end of main
	
