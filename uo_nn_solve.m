function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
%
% Solver
%  

tic

% Generate the training dataset (Xtr, ytr) and the test dataset (Xte, yte)
[Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
[Xte,yte] = uo_nn_dataset(te_seed, te_q, num_target, 0);

w=zeros(35,1);

sig = @(X) 1./(1+exp(-X));
y = @(X,w) sig(w'*sig(X));
L = @(w) norm(y(Xtr,w)-ytr)^2 + (la*norm(w)^2)/2;
gL = @(w) 2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))'+la*w;

%{
sig = @(Xds) 1./(1+exp(-Xds));
y = @(Xds,w) sig(w'*sig(Xds));
L = @(w,Xds,yds) (norm(y(Xds,w)-yds)^2)/size(yds,2)+(la*norm(w)^2)/2;   % loss function (=objective function)
gL = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w)))')/size(yds,2)+la*w;
%}

% Find the value of w* minimizing L with Backtracking Line Search
[wk,niter]=uo_nn_GM(w,L,gL,epsG,kmax,epsal,kmaxBLS,ialmax,c1,c2);

kmaxO = size(wk,2);
wo = wk(:,kmaxO);

% Train accuracy
y_calc = y(Xtr, wo);
sum_tr = 0;
for i = 1:tr_p
    sum_tr = sum_tr + (round(y_calc(i)) == ytr(i));
end
tr_acc = double(sum_tr/ tr_p);

% Test accuracy
y_pred = y(Xte, wo);
sum_te = 0;
for i = 1:te_q
    sum_te = sum_te + (round(y_pred(i)) == yte(i));
end
te_acc = double(sum_te/ te_q);
tex=toc;

% Temporary returns
fo=0;

end