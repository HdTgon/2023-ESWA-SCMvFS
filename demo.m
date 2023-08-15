clc;clear all;

addpath('util');

a='handwritten';

load(a);

[~,v]=size(fea);
n=size(fea{1},1);
class_num=size(unique(gt),1);
label=gt;
dimension=0;
for num = 1:v
    dimension=dimension+size(fea{num},2);
end

feanum=[10:10:300]; % feature dimension

ACC=[];
NMI=[];
ACCstd=[];
NMIstd=[];
rankmvufs=[];
mvufsobj=[];

alpha=[0.001:0.002:0.009];
alpha=alpha*2;
beta=[1e-2,1e-1,1,1e1,1e2];
gamma=[1e-2,1e-1,1,1e1,1e2];

iter=0;

for i=1:size(alpha,2)
for o=1:size(beta,2)
for l=1:size(gamma,2)

[P,obj]=SCMvFS(fea,alpha(i),beta(o),gamma(l),n,v,class_num);

iter=iter+1

allP=[];
X=[];
for num = 1:v
    allP=[allP;P{num}];
    X=[X,fea{num}];
end

W1 = [];
for m = 1:dimension
    W1 = [W1 norm(allP(m,:),2)];
end
%% test stage
[~,index] = sort(W1,'descend');
rankmvufs(:,iter)=index;
mvufsobj(1:size(obj,2),iter)=obj;

for j = 1:length(feanum)
acc=[];
nmi=[];
for k = 1:20
    new_fea = X(:,index(1:feanum(j)));
    idx = kmeans(new_fea, class_num,'MaxIter',200);
    res = bestMap(label,idx);
    acc111 = length(find(label == res))/length(label); % calculate ACC 
    nmi111 = MutualInfo(label,idx); % calculate NMI
    acc=[acc;acc111];
    nmi=[nmi;nmi111];
end
ACC=[ACC;sum(acc)/20];
ACCstd=[ACCstd;std(acc)];
NMI=[NMI;sum(nmi)/20];
NMIstd=[NMIstd;std(nmi)];
end

end
end
end

number=size(ACC,1);
final=zeros(number,4);
for j=1:number
    final(j,1:4)=[ACC(j,1),ACCstd(j,1),NMI(j,1),NMIstd(j,1)];
end

[newvalue,endindex]=sort(ACC,'descend');

final(endindex(1:10),:);

