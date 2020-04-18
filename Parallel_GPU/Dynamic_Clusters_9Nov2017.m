
% This code will find (for successive spikes)
% number of clusters
% size of largest cluster
% ids of neurons in each cluster
clear
tic

%N = 6 ;
%NL_max =10;
%NL_min = 2;
%Ngmax =50;
%Nic =50;

%n_spikes = 5;

N = 7;
NL_max = 12;
NL_min = 1;
Ngmax = [50 50 50 200 200 600 600 900 1300 1000 1000 1000];
Nic =50;

n_spikes = 7;


% open files
fileID = fopen('fspike_time.txt','r');
num_dyn_clstrID = fopen('fnum_dyn_clstr.txt','w');
size_lrgst_dyn_clstrID = fopen('fsize_lrgst_dyn_clstr.txt','w');

% files - put labels on each column
fprintf(num_dyn_clstrID,'# nL \t ng \t nic \t  num_clstr \t num_clstr \n');
fprintf(num_dyn_clstrID,'\n');
fprintf(size_lrgst_dyn_clstrID,'# nL \t ng \t nic \t size_clstr \t size_clstr \n');
fprintf(size_lrgst_dyn_clstrID,'\n');

count_of_graphs_sync_only_state = zeros(1,NL_max-NL_min+1);
count_of_graphs_with_dyn_clstr = zeros(1,NL_max-NL_min+1);
count_of_graphs_multistable = zeros(1,NL_max-NL_min+1);

for iL=1:NL_max-NL_min+1
    iiL=iL
    
sizes_dyn_clstr{iL}=[];
secnd_lrgst_dyn_clstr{iL} = [];
period_lgst_clstr_vs_time{iL} = [];
prb_lrgst_dyn_clstr_in_a_graph{iL} = [];
prb_2ndlrgst_dyn_clstr_in_a_graph{iL} = [];
period{iL}=[];
lrgst_dyn_clstr{iL} = [];
%num_graphs_initcond_that_sync{iL}

for ig = 1:Ngmax(iL)
    
    flag_dyn_clstr_in_graph = 0 ; % if this flag=0, implies dynmc clstr in this graph for some initial condn
    flag_multistable_graph = 0 ;
    flag_sync_in_a_graph = 0;
    
for ic = 1:Nic
    
    nL = NL_min+iL-1;
    
    % Read data from file that will processed in this code
    label1 = textscan(fileID,'%s %s %s %s',1);
    tspk = fscanf(fileID,'%f %f %f \n',[n_spikes+1 N]);
    tspk = tspk';
    newline = fscanf(fileID,'\n ',[1 1]) ;

    sizes_dyn_clstr_in_a_graph = [];
    unq_sizes_dyn_clstr_in_a_graph = [];
    sort_unq_sizes_dyn_clstr_in_a_graph = [];
    
    % DYNAMIC CLUSTERS  
    lrgst_clstr_vs_time = [];
    clear unq_tspk_sort;
    for ispk=2:n_spikes+1
        tspk_sort = sort(tspk(:,ispk)');
        unq_tspk_sort =unique(tspk_sort);
        temp = size(unq_tspk_sort);
        num_clstr = temp(1,2);
        clear temp
           
        % sizes of dynamic clusters in several spike clusters (in single
        % simulation)
        for iclstr=1:num_clstr
             temp = size(find(tspk_sort == unq_tspk_sort(1,iclstr)));
             sizes_dyn_clstr_in_a_graph = [sizes_dyn_clstr_in_a_graph temp(1,2)];
        end
        clear temp    
    end
    
    % sizes of dynamic clusters for different iL's
    sizes_dyn_clstr{iL} = [sizes_dyn_clstr{iL} sizes_dyn_clstr_in_a_graph];
    
    % largest dynamic clusters in each simulation 
    lrgst_dyn_clstr_in_a_graph = max(sizes_dyn_clstr_in_a_graph);
    lrgst_dyn_clstr{iL} = [lrgst_dyn_clstr{iL} lrgst_dyn_clstr_in_a_graph];
    
    % sizes of 2nd largest dynamic cluster in a graph
    unq_sizes_dyn_clstr_in_a_graph = unique(sizes_dyn_clstr_in_a_graph);
    sort_unq_sizes_dyn_clstr_in_a_graph = sort(unq_sizes_dyn_clstr_in_a_graph,'descend');
    temp = size(unq_sizes_dyn_clstr_in_a_graph);
    if(temp(1,2) == 1)
        secnd_lrgst_dyn_clstr{iL} = [secnd_lrgst_dyn_clstr{iL} lrgst_dyn_clstr_in_a_graph];
    else 
        secnd_lrgst_dyn_clstr{iL} = [secnd_lrgst_dyn_clstr{iL} sort_unq_sizes_dyn_clstr_in_a_graph(2)];
    end
    % count number of graphs (for each iL) on which dynamic clusters are
    % formed. 
    if(temp(1,2)>1 && flag_dyn_clstr_in_graph == 0)
        flag_dyn_clstr_in_graph = 1;
        count_of_graphs_with_dyn_clstr(iL) =  count_of_graphs_with_dyn_clstr(iL) + 1;
    end
    % count number of graphs (for each iL) in which sync is the only state (& hence no dynamic clusters are
    % formed).
    if(ig==Ngmax(iL))
        count_of_graphs_sync_only_state(iL) = Ngmax(iL) - count_of_graphs_with_dyn_clstr(iL);
    end
    % count graphs that are multi-stable (& sync being one of the possible
    % states)
    clear temp
    
    if(flag_dyn_clstr_in_graph == 1 && flag_sync_in_a_graph == 0)
        if(ismember(N,unq_sizes_dyn_clstr_in_a_graph) == 1)
             flag_sync_in_a_graph = 1;
             count_of_graphs_multistable(iL) = count_of_graphs_multistable(iL)+1;
        end
    end
           
    clear temp;
    
    
    % Periodicity of the time-sequence of dynamic clusters in a simulaton
    period_in_a_graph = seqperiod(sizes_dyn_clstr_in_a_graph) ;
    period{iL} = [period{iL} period_in_a_graph];
    
    % sizes of dynamic clusters in the periodic sequence
    temp = period_in_a_graph;
    periodic_seq_clstr_in_a_graph = sizes_dyn_clstr_in_a_graph(1:temp);
    clear temp
    
    % probability of occurance of largest dynamic cluster in single simln
    %temp1 = size(find(periodic_seq_clstr_in_a_graph == lrgst_dyn_clstr_in_a_graph));
    %temp2 = size(find(periodic_seq_clstr_in_a_graph == (lrgst_dyn_clstr_in_a_graph-1) ));
    temp1 = size(find(periodic_seq_clstr_in_a_graph == 6));
    temp2 = size(find(periodic_seq_clstr_in_a_graph == 5 ));
    num_lrgst_clstr_in_prd_seq = temp1(1,2);
    num_2ndlrgst_clstr_in_prd_seq = temp2(1,2);
    clear temp1;
    clear temp2;
    
    temp1 = num_lrgst_clstr_in_prd_seq ./ period_in_a_graph;
    temp2 = num_2ndlrgst_clstr_in_prd_seq ./ period_in_a_graph;
    prb_lrgst_dyn_clstr_in_a_graph{iL} = [ prb_lrgst_dyn_clstr_in_a_graph{iL} temp1];
    prb_2ndlrgst_dyn_clstr_in_a_graph{iL} = [ prb_2ndlrgst_dyn_clstr_in_a_graph{iL} temp2];
    
    %if(lrgst_dyn_clstr_in_a_graph == N)
    %    num_graphs_initcond_that_sync{iL} = num_graphs_initcond_that_sync{iL}+1;
    %end
    
end
end
totprob{iL} = prb_lrgst_dyn_clstr_in_a_graph{iL} + prb_2ndlrgst_dyn_clstr_in_a_graph{iL}
end
toc

fclose('all');

