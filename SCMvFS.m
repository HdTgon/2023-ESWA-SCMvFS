function [Hv,obj]=SCMvFS(fea,alpha,beta,eta,n,v,c)

% gamma=1e3;
[v1,v2]=size(fea);
Hv=cell(v1,v2);
Sv=cell(v1,v2);
Lv=cell(v1,v2);
Dv=cell(v1,v2);
F=orth(randn(n,c));
d=zeros(v2,1);
% vectorn=ones(n,1);   
MaxIter=30;

for num = 1:v
    fea{num}=fea{num}';
    d(num)=size(fea{num},1);
    Hv{num}=randn(d(num),c);
    Dv{num}=eye(d(num));
end

for iter=1:MaxIter

    %%update Sv
    for num = 1:v
        intermedia=fea{num}'*Hv{num}*Hv{num}'*fea{num};
        Q=calculate(F);
        Sv{num}=(intermedia+eta*eye(n))\(intermedia-alpha*Q/4);
        Lv{num} = diag(sum(Sv{num}))-(Sv{num}+Sv{num}')/2;
    end
     
    %%update Hv
    for num = 1:v
        LG = (eye(n) - Sv{num});
        LG = LG * LG';
        LG = (LG + LG') / 2; %%LLE L
        [Y, ~, ~]=eig1(LG, c, 0);   
       %%solve ||Y-XtH||F+beta||H||21
        Hv{num}=(fea{num}*fea{num}'+beta*Dv{num})\(fea{num}*Y);
        Hi=sqrt(sum(Hv{num}.*Hv{num},2)+eps);
        diagonal=0.5./Hi;
        Dv{num}=diag(diagonal);
    end
    
    %%update F
    Feig = 0;
    for num = 1:v
        Feig = Feig + Lv{num};
    end
    [F, ~, ~]=eig1(Feig, c, 0);   
    
    sumobj=0;
    for num = 1:v
       sumobj=sumobj+norm(Hv{num}'*fea{num}-Hv{num}'*fea{num}*Sv{num},'fro').^2+alpha*trace(F'*Lv{num}*F)+beta*trace(Hv{num}'*Dv{num}*Hv{num})+eta*norm(Sv{num},'fro').^2;
    end
    obj(iter)=sumobj;
     if iter >= 2 && (abs(obj(iter)-obj(iter-1)/obj(iter))<eps)
        break;
    end
    
end

end

