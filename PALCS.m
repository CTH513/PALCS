function  F = PALCS(X, k, m, beta,gamma,eta)

maxIters = 100;

nView = length(X);
nSample = size(X{1},2);
Obj = [];

%Initilize G,F
G = eye(m,k);
F = eye(k,nSample); 

parfor i = 1:nView
   di = size(X{i},1); 
   A{i} = zeros(di,m);
end

C = zeros(m,nSample); % m  * n
XX = [];
parfor p = 1 : nView
    X{p} = mapstd(X{p},0,1);
    XX = [XX;X{p}];
end
[XU,~,~]=svds(XX',m);
rand('twister',12);
[IDX,~] = kmeans(XU,m, 'MaxIter',100,'Replicates',5);

for i = 1:nSample
    C(IDX(i),i) = 1;
end

%% Initialization
E = cell(1,nView);
AA = cell(1,nView);
sumAA = zeros(m,m);

parfor v = 1:nView    
    E{v} = zeros(m,nSample);
    AA{v} = A{v}' * A{v};
    sumAA = sumAA + AA{v};
end

temp_inv = cell(1,nView);
for v = 1:nView
    temp_inv{v} = (AA{v} + beta*eye(m))\eye(m);
end

h = m/k;
I = eye(k); Y= [];
parfor ik = 1:k
    Y = [Y repmat(I(:,ik),1,h)];
end
clear I

for iv = 1:nView
    [Up,~,Sp] = svd(A{iv}*Y', 'econ');
    P{iv} = Up*Sp';
end
clear Up Sp

%% Alternate minizing strategy
for t = 1:maxIters
    %------------- update A -------------  
    parfor iv = 1:nView
        A{iv} = (X{iv}*(C+E{iv})' + eta*P{iv}*Y)/((C+E{iv})*(C+E{iv})' + eta*eye(m));
    end

    %------------- update P -------------
     for iv = 1:nView
        [Up,~,Sp] = svd(A{iv}*Y', 'econ');
        P{iv} = Up*Sp';
     end

    %------------- update C -------------
    A_syl = 2*sumAA + 2*gamma*eye(m) ;
%     B_syl = zeros(nSample);
    C_syl = 2* gamma * G * F;
    for v=1:nView
        C_syl = C_syl + 2*A{v}'*(X{v} - A{v} * E{v});
    end
    C = A_syl \ (C_syl);   

    %------------- update E -------------
    parfor v = 1:nView
        temp2 = (A{v}'* X{v} - AA{v}*C);
        E{v} = temp_inv{v} * temp2;
    end

    %------------- update G -------------
    J = C*F';      
    [Ug,~,Vg] = svd(J,'econ');
    G = Ug*Vg';

    %------------- update F -------------
     F=zeros(k,nSample);
    for i=1:nSample
        Dis=zeros(k,1);
        for j=1:k
            Dis(j)=(norm(C(:,i)-G(:,j)))^2;
        end
        [~,r]=min(Dis);
        F(r(1),i)=1;
    end

    obj1 = 0;
    obj3 = 0;
    obj4 = 0;
    parfor i = 1:nView
        obj1 = obj1 + norm((X{i}-A{i}*(C+E{i})),'fro')^2;
        obj3 = obj3 + norm(E{i},'fro')^2;
        obj4 = obj4 + norm(A{i} - P{i} * Y,'fro')^2;
    end
    Obj(t) = obj1 + beta*obj3 + gamma * norm(C - G * F,'fro')^2+ eta * obj4;
   
    if (t>1) && (abs((Obj(t-1)-Obj(t))/(Obj(t-1)))<1e-3 || t>maxIters || Obj(t) < 1e-10) 
        break;
    end
end


