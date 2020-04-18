% create arrays representing number of frozen clusters
% the array contains cluster info for all graphs (ig) & IC's (ic) for a
% specific nL

% create a cell-array to store spectrum of cluster-sizes for each nL
% cell-array is used because number of different cluster-sizes will vary
% with nL.
clear
tic

%%N = 6 ;
%NL_max =10;
%NL_min = 2;
%Ngmax =50;
%Nic =50;

N = 7 ;
NL_max = 12;
NL_min = 1;
Ngmax = [50 50 50 200 200 600 600 900 1300 1000 1000 1000];
Nic =50;

size_frzn_clstrID = fopen('fsize_frzn_clstr.txt','r');

%unq_clstr_sz_arrayID = fopen('/home/gaurav-dar/Documents/MATLAB/funq_clstr_sz_array.txt','w');

% Read data from file that will processed in this code
for iL=1:NL_max-NL_min+1
    clstr_sizes{iL}=0;
end

for iL=1:NL_max-NL_min+1
    iiL=iL
for ig=1:Ngmax(iL)
   % ig
for ic=1:Nic
    label1 = textscan(size_frzn_clstrID,'%s %s %s',1);
    clstr_size_read = fscanf(size_frzn_clstrID,'%d \n',[1 Inf]);
    clstr_sizes{iL} = [clstr_sizes{iL} clstr_size_read 0];
    newline = fscanf(size_frzn_clstrID,'\n ',[1 1]) ;
end
end
end
    

%clstr_sizes{1} =  [0 2 3 3 0 1 4 4 2 0 2 2 1 1 3 5 5 5 0];
%clstr_sizes{2} =  [0 3 4 4 3 0 4 5 5  0 3 3 3 4 2 1 1 3 8 9 0];
 
num_clstr=[];

for iL=1:NL_max-NL_min+1
    clear split_clstr_sizes 
    clear unq_split 
    join_unq_split{iL} = [];
    
    findzeros = find(clstr_sizes{iL} == 0);   
    count = 0;
    start = findzeros(1)+1;
    eend = findzeros(2)-1;
    
    for ig=1:Ngmax(iL)
        for ic=1:Nic
             
             count = count + 1; 
             start = findzeros(count)+1;
             eend = findzeros(count+1)-1;
             
             split_clstr_sizes{count} = clstr_sizes{iL}(start:eend); % for current iL
             unq_split{count} = unique(split_clstr_sizes{count}); % unique cluster sizes for current ig-ic
             join_unq_split{iL} = [join_unq_split{iL} unq_split{count}]; % for current iL
             
        end
    end
end

num_clstr = [];
for iL=1:NL_max-NL_min+1   
    unq_clstr_sz{iL} = unique(join_unq_split{iL}); %unique cluster-sizes over all ig-ic for a specific iL
    unq_clstr_sz{iL} = sort(unq_clstr_sz{iL},'descend');
    num_clstr = [num_clstr numel(unq_clstr_sz{iL})];
end

max_num_clstr = max(num_clstr);     % max no. of cluster-sizes
unq_clstr_sz_array = zeros(NL_max-NL_min+1,max_num_clstr); % create zero matrix

for iL=1:NL_max-NL_min+1
    temp = size(unq_clstr_sz{iL}); % number of unique clusters with nL=1
    unq_clstr_sz_array(iL,1:temp(1,2)) = unq_clstr_sz{iL}; %save the unique clusters for iL in row-iL
end

% Finally we are ready. row-iL contains unique cluster sizes for iL.
clear temp
hist_clstr_sizes = zeros(NL_max-NL_min+1,max_num_clstr);
z = max_num_clstr;

binranges = 1:6
for iL=1:NL_max-NL_min+1
   clear temp
   clear temp1
   temp = histcounts(join_unq_split{iL});
   [bincounts{iL}] = histc(join_unq_split{iL},binranges)
   temp1 = temp(find(temp)) ;
   hist_clstr_sizes(iL,z-num_clstr(iL)+1:z) = temp1;
end

for isize = 1:6
    prob_clstr_size_vs_iL{isize} = [];
    for iL=1:NL_max-NL_min+1
        prob_clstr_size_vs_iL{isize} = [prob_clstr_size_vs_iL{isize} bincounts{iL}(1,isize)]
    end
end


clear temp
clear temp1
% find modes

sz_mode = [];
for iL=1:NL_max-NL_min+1
    size_mode{iL}=mode(join_unq_split{iL});
    sz_mode = [sz_mode size_mode{iL}];
end

frq_sz_mode = [];
for iL=1:NL_max-NL_min+1
    temp = numel(find(join_unq_split{iL} == size_mode{iL}));
    frq_sz_mode = [frq_sz_mode temp];
end

for iL=1:NL_max-NL_min+1
    hist{1} = histcounts(join_unq_split{iL});
    hist{1} = sort(hist{1},'descend');
end


    
%celldisp(clstr_sizes) ;
%celldisp(join_unq_split) ;
unq_clstr_sz_array;
hist_clstr_sizes;
sz_mode;
frq_sz_mode;

dlmwrite('ffrzn_unq_clstr_sz_array.txt',unq_clstr_sz_array,'delimiter','\t');
dlmwrite('ffrzn_hist_clstr_sizes.txt',hist_clstr_sizes,'delimiter','\t')
dlmwrite('ffrzn_sz_mode.txt',sz_mode,'delimiter','\t')
dlmwrite('ffrzn_frq_sz_mode.txt',frq_sz_mode,'delimiter','\t')

toc
