
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

N = 7 ;
NL_max = 12;
NL_min = 1;
Ngmax = [50 50 50 200 200 600 600 900 1300 1000 1000 1000];
Nic =50;

n_spikes = 7;


% open files
fileID = fopen('fspike_time.txt','r');
num_dyn_clstrID = fopen('fnum_dyn_clstr.txt','w');
size_lrgst_dyn_clstrID = fopen('fsize_lrgst_dyn_clstr.txt','w');
nrnID_frzn_clusterID = fopen('fnrnID_frzn_cluster.txt','w');
num_frzn_clstrID = fopen('fnum_frzn_clstr.txt','w');
size_frzn_clstrID = fopen('fsize_frzn_clstr.txt','w');
size_lrgst_frzn_clstrID = fopen('fsize_lrgst_frzn_clstr.txt','w');

% files - put labels on each column
fprintf(num_dyn_clstrID,'# nL \t ng \t nic \t  num_clstr \t num_clstr \n');
fprintf(num_dyn_clstrID,'\n');
fprintf(size_lrgst_dyn_clstrID,'# nL \t ng \t nic \t size_clstr \t size_clstr \n');
fprintf(size_lrgst_dyn_clstrID,'\n');
fprintf(num_frzn_clstrID,'# nL \t ng \t nic \t num frzn clusters \n');
fprintf(num_frzn_clstrID,'\n');
fprintf(size_lrgst_frzn_clstrID,'# nL \t ng \t nic \t \t  size of largst frozen cluster \n');
fprintf(size_lrgst_frzn_clstrID,'\n');

old_index = 1;

for iL=1:NL_max-NL_min+1
    iiL=iL
for ig = 1:Ngmax(iL)
   % ig
for ic = 1:Nic
    nL = NL_min+iL-1;
    % Labels on files
    %fprintf(nrnids_dyn_clstrID,'il=%d , ig=%d , ic=%d \n', il, ig, ic);
    fprintf(nrnID_frzn_clusterID,'# nL=%d , ig=%d , ic=%d \n', nL, ig, ic);
    fprintf(nrnID_frzn_clusterID,'\n');
    fprintf(size_frzn_clstrID,'#nL=%d, ig=%d, ic=%d \n', nL, ig, ic);
    fprintf(size_frzn_clstrID,'\n');
    
    % Read data from file that will processed in this code
    label1 = textscan(fileID,'%s %s %s %s',1);
    tspk = fscanf(fileID,'%f %f %f \n',[n_spikes+1 N]);
    tspk = tspk';
    newline = fscanf(fileID,'\n ',[1 1]) ;

    
    % DYNAMIC CLUSTERS  
    for ispk=2:n_spikes+1
        unique_tspk{ispk} = unique(tspk(:,ispk));  % spike time
        num_clusters{ispk} = size(unique_tspk{ispk}); % number of clusters
        [hist{ispk},ind{ispk}] = histc(tspk(:,ispk),unique_tspk{ispk}); % no. of neurons in each cluster & identify of the neurons
        largest_cluster{ispk} = max(hist{ispk}); % size of largest cluster
        nrn_ids_mode{ispk} = {find(tspk(:,ispk)==mode(tspk(:,ispk)))}; % identity of neurons in the largest cluster

        %find ids of neurons in each cluster 
        for i=1:num_clusters{ispk}
            %nrn_ids_cluster(i,1:hist(i)) = find(ind{ispk}==i);  %identify of neurons in each cluster
            nrnids_cluster{ispk}{i} = {find(ind{ispk}==i)};
        end

       % celldisp(num_clusters);  % dynamic clusters - number of clusters
       % celldisp(largest_cluster);  % dynamic cluster - size of largest cluster
       % celldisp(nrnids_cluster);  % dynamic cluster - neuron id's in each cluster
    end
    
    % FROZEN CLUSTER STATISTICS
     num_frzn_cluster = 0;
     nrnID_frzn_cluster = [];
     size_frzn_cluster = [];
     count  = 0;
     nc2 = num_clusters{2};
     nc3 = num_clusters{3};
     temp = 0;
     for i=1:nc2(1)
         for j = 1:nc3(1)
                  nrnID_frzn_cluster = intersect(nrnids_cluster{2}{i}{1},nrnids_cluster{3}{j}{1}); 
                  nrnID_frzn_cluster = nrnID_frzn_cluster' ;
                  if(isempty(nrnID_frzn_cluster) == 0)
                      num_frzn_cluster = num_frzn_cluster + 1;
                      count = num_frzn_cluster;
                      size_frzn_cluster(count) = numel(nrnID_frzn_cluster);
                      fprintf(size_frzn_clstrID,'%d \n',size_frzn_cluster(count));
                      fprintf(nrnID_frzn_clusterID,'%d \n',nrnID_frzn_cluster(1,1:end));
                      fprintf(nrnID_frzn_clusterID,'\n');
                  end
                 % if(isempty(nrnID_frzn_cluster) == 1)
                   %  fprintf(nrnID_frzn_clusterID,'\n')
                    % fprintf(nrnID_frzn_clusterID,'\n');
                 % end
          end
     end   
     
     if(iL==8 && ic<3)
         index = [old_index:old_index+count-1];
         stem(index,size_frzn_cluster)
         old_index=index+count+10;
     end
     %ind_separator=old_index-1;
     %temp=0;
     %stem(ind_separator,temp)
     hold on;
     
     size_lrgst_frzn_clstr = max(size_frzn_cluster);
     
     % File write - cumulative numbers related to frozen clusters
     fprintf(num_frzn_clstrID,'%d \t %d \t %d \t  %d \n',nL,ig,ic,num_frzn_cluster);
     fprintf(size_frzn_clstrID,'\n \n');
     fprintf(size_lrgst_frzn_clstrID,'%d \t %d \t %d \t \t \t %d \n',nL,ig,ic,size_lrgst_frzn_clstr);
     %fprintf(size_lrgst_frzn_clstrID,'\n \n');
     
     fprintf(num_dyn_clstrID,'%d \t %d \t %d \t \t %d \t \t %d \n', nL,ig,ic,nc2(1),nc3(1));
     %fprintf(num_dyn_clstrID,'\n \n');
     fprintf(size_lrgst_dyn_clstrID,'%d \t %d \t %d \t \t %d \t %d \n', nL,ig,ic,largest_cluster{2},largest_cluster{3});
     %fprintf(size_lrgst_dyn_clstrID,'\n \n');
    % fprintf(frzn_clusterID,'\n \n');
     

        

     % num_clustersID = fopen('/home/gaurav-dar/Documents/MATLAB/fnum_clusters.txt','w');
      %fprintf(num_clustersID,'%d \t %d \t %d \t %d \n ', num_clustersA(2),num_clustersB(2),largest_clusterA,largest_clusterB);
     % fclose(num_clustersID);
      %dlmwrite('myfile.txt',idsA,'delimiter','\t')
      %dlmwrite('myfile.txt','***','-append','delimiter','\t')
      %dlmwrite('myfile.txt',idsB,'-append','delimiter','\t')
      %type('myfile.txt')
  
end
end
end
toc

fclose('all');

