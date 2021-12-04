% Stochastic Gradient Method solver
% Marcel, Mengxue
% OTDM-NN-Nov21

function [wk,niter] = uo_nn_SGM(w,f,g,Xtr,ytr,Xte,yte,sg_seed,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest)

rng(sg_seed);
p = size(Xtr,2);
m = floor(sg_ga*p);
sg_ek = ceil(p/m);
sg_kmax = sg_emax * sg_ek;
e = 0;
s = 0;
L_te_best = +inf;
sg_k = ceil(sg_be*sg_kmax);
sg_al = 0.01*sg_al0;
k=0;

while e < sg_emax && s < sg_ebest
    % random permutations
    P = randperm(p); 
    P_Xtr = Xtr(:,P);
    P_ytr = ytr(:,P);
    for i=0:ceil(p/m-1)
        S_Xtr = P_Xtr(:,i*m+1:min((i+1)*m,p));
        S_ytr = P_ytr(i*m+1:min((i+1)*m,p));
        d = -g(w, S_Xtr,S_ytr);
        if k <= sg_k
            al = (1-k/sg_k)*sg_al0+(k/sg_k)*sg_al;
        else
            al = sg_al;
        end
        k = k+1;
        w = w+al*d;
    end
    e = e+1;
    
    L_te = f(w,Xte,yte);
    if L_te < L_te_best
        L_te_best = L_te;
        wk = w;
        s = 0;
    else
        s = s+1;
    end

end
niter = k;

end