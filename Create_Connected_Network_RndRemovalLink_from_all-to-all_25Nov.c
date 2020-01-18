// This code constructs all-to-all network & then randomly removes a fixed number of links randomly

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 20


///////////////////////////////  A NEW FUNCTION STARTS **********************************************
// **************************************************************************************************
//////////////////////SYNAPTIC WEIGHTS ARE FORMED IN THIS FUNCTION **********************************************
//////////////////////////////  The function starts here ********************************************************

int synaptic_weights_connected_network(double w[][N],int nL)

{

	int i,j,k,kk,neuron1,neuron2;
	double w_flag[N][N];
	int syn_to_remove, tot_syn_removed ;


	int connected_nodes[N] ;
	int current_ptr, endptr, parent_node;
	int flag_connected = 0 ;
	int flag_already_connected;

	FILE *debug;

	debug = fopen("fdebug.txt","w");


	// GENERATE AN ALL-TO-ALL NETWORK ************************************************************************
	for(i=0;i<N;i++){
	    for(j=0;j<N;j++) {
		if(j != i){
			w[i][j] = 1;
		}
		else if(j==i){
			w[i][j] =0;
		} 
	    }
	}


	// REMOVE SYNAPSES FROM ABOVE ALL-TO-ALL NETWORK *********************************************************

	syn_to_remove = nL;
	tot_syn_removed = 0;

	// Initialize array w_flag
	for(k=0;k<N;k++)
	{
		for(kk=0;kk<N;kk++)
		{
		w_flag[k][kk] = 0; // w_flag[k][kk] is changed to value 1, if the synapse between k --> kk is removed
		}
	}

	// Generate a new network by removing synapses randomly
	while(tot_syn_removed < syn_to_remove)
	{
		int neuron1 = rand()%N;
		int neuron2 = rand()%N;
		if(neuron1 != neuron2)
		{
			if(w_flag[neuron1][neuron2] == 0) // synapse between these two neurons has not been changed.
			{				
				w_flag[neuron1][neuron2]=1;
				w_flag[neuron2][neuron1]=1;
				w[neuron1][neuron2]=0;
				w[neuron2][neuron1]=w[neuron1][neuron2];
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



	
       	for(k=0;k<N;k++)
	{
		for(kk=0;kk<N;kk++)
		{
		w_flag[k][kk] = 0; // w_flag[k][kk] is changed to value 1, if the synapse between k --> kk is removed
		}
	}

	

	
	connected_nodes[0] = 0;
	for(i=1;i<N;i++)
	{
		connected_nodes[i] = -1;
        }
	current_ptr = 0;
	endptr = 0 ;  // points towards the last non-zero element in the connected_nodes array

	while(current_ptr <= endptr){
      
	for(i = 0; i<N; i++) 
	{    
		
		parent_node = connected_nodes[current_ptr] ;

		flag_already_connected = 0 ;

		for(j=0;j<=endptr;j++)
		{
			if(connected_nodes[j]==i)
			{
			   flag_already_connected = 1;
			}
		}


		if(w[parent_node][i] == 1)
		{
			    
			 if(w_flag[parent_node][i] == 0)
			 {

				if(flag_already_connected ==0)
			
			 	{
			     
		            	 endptr ++ ;
		             	 connected_nodes[endptr] = i ; // stores node numbers connected to parent_node

			    	 w_flag[parent_node][i] = 1 ;
			     	 w_flag[i][parent_node] = w_flag[parent_node][i] ; //links already visited
				 
				 //printf("i= %d \t endptr= %d \t current_ptr= %d \t connected_nodes[endptr] = %d \n",i, endptr,current_ptr,connected_nodes[endptr]);
				 }

			  }

                }
		
	
		if (i == N-1)
		{

			current_ptr++ ;
		}	
			
	}
	}

		
	if(endptr == N-1)
        {
		flag_connected = 1 ;
	}

	return flag_connected;

} // end of function

	
