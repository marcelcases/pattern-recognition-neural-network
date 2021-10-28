function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
%
% Solver
%  

% 1. Generate the training dataset (Xtr, ytr) and the test dataset (Xte, yte)
[Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
[Xte,yte] = uo_nn_dataset(te_seed, te_q, num_target, 0);

%uo_nn_Xyplot(Xtr,ytr,[])
%uo_nn_Xyplot(Xte,yte,[])

% 2. To find the value of w* minimizing L with Backtracking Line Search
sig = @(Xds) 1./(1+exp(-Xds));
y = @(Xds,w) sig(w'*sig(Xds));
L = @(w,Xds,yds) (norm(y(Xds,w)-yds)^2)/size(yds,2)+(la*norm(w)^2)/2;   % loss function (=objective function)
gL = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w)))')/size(yds,2)+la*w;

while norm(gL(w,Xtr,ytr)) >= epsG
  d = -gL(w,Xtr,ytr);
  [alphas,iout] = uo_BLSNW32(L(w,Xtr,ytr),gL(w,Xtr,ytr),x0,d,ialmax,c1,c2,kmaxBLS,epsal); % BLS
  x = x + al*d;
  k = k+1;  xk = [xk,x]; alk = [alk,al];
end




% Temporary returns
wo=0;fo=0;tr_acc=0;te_acc=0;niter=0;tex=0;
