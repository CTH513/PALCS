%%
clear;
clc;
warning off;
addpath(genpath('./'));
addpath(genpath('./tools/'));

dsPath = './Dataset/';
resPath = './res-lmd0/';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};
%%
% load data & make folder
dataName = 'Caltech101-20';
disp(dataName);
load(strcat(dsPath,dataName));
matpath = strcat(resPath,dataName);
txtpath = strcat(resPath,strcat(dataName,'.txt'));
if (~exist(matpath,'file'))
    mkdir(matpath);
    addpath(genpath(matpath));
end
dlmwrite(txtpath, strcat('Dataset:',cellstr(dataName), '  Date:',datestr(now)),'-append','delimiter',' ','newline','pc');

% gt: ground truth label
% X: input data
nView = length(X);
for i =1:nView
    X{i} = X{i}';
end
gt = Y;
K = length(unique(gt)); % number of clusters
%%
beta = [0.001 0.01 0.1 1 10 100,1000,10000];
gamma = [0.001 0.01 0.1 1 10 100,1000,10000];
eta =[0.001 0.01 0.1 1 10 100,1000,10000];
anchor = [K 2*K 3*K];
    %%
    for ichor = 1:length(anchor)
        for i = 1:length(beta)
            for g = 1:length(gamma)
                for e = 1:length(eta)
                    tic;
                    F = PALCS(X, K, anchor(ichor),  beta(i),  gamma(g), eta(e)); 
                    [~,idx]=max(F);
                    res = Clustering8Measure(Y,idx); 
                    timer(ichor,e)  = toc;
                    str = strcat('anchor:',num2str(anchor(ichor)),'       beta:',num2str(beta(i)),'       gamma:',num2str(gamma(g)),'       eta:',num2str(eta(e)),'       res:',num2str(res),'       Time:',num2str(timer(ichor,e)));
                    disp(str);
                    resall{ichor,e} = res;
                    dlmwrite(txtpath, [anchor(ichor) beta(i) gamma(g) eta(e) res timer(ichor,e)],'-append','delimiter','\t','newline','pc');
                end
            end
        end
    end
    
    clear resall objall X Y k