% Neural Network solver
% Marcel, Mengxue
% OTDM-NN-Nov21

function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
%
% Solver
%  

tic

% Generate the training dataset (Xtr, ytr) and the test dataset (Xte, yte)
[Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
[Xte,yte] = uo_nn_dataset(te_seed, te_q, num_target, 0);

w=zeros(35,1);

% Functions definition
sig = @(Xds) 1./(1+exp(-Xds)); % activation function
y = @(Xds,w) sig(w'*sig(Xds));
L1 = @(w) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2) + (la*norm(w)^2)/2; % loss (=obj) for GM and QNM
gL1 = @(w) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w; % for GM and QNM
L2 = @(w,Xds,yds) (norm(y(Xds,w)-yds)^2)/size(yds,2) + (la*norm(w)^2)/2; % loss (=obj) for SGM
gL2 = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w)))')/size(yds,2)+la*w; % for SGM


% Find the value of w* minimizing L
if isd == 1 % Gradient Method
    [w,niter] = uo_nn_GM(w,L1,gL1,epsG,kmax,epsal,kmaxBLS,ialmax,c1,c2);

elseif isd == 3 % Quasi-Newton Method
    [w,niter] = uo_nn_QNM(w,L1,gL1,epsG,kmax,epsal,kmaxBLS,ialmax,c1,c2);

elseif isd == 7 % Stochastic Gradient Method
    [w,niter] = uo_nn_SGM(w,L2,gL2,Xtr,ytr,Xte,yte,sg_seed,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest);

end

kmaxO = size(w,2);
wo = w(:,kmaxO); % return w*
fo=L1(wo); % return L*

% Train accuracy
y_fit = y(Xtr, wo);
tr_sum = 0;
for i = 1:tr_p
    tr_sum = tr_sum + (round(y_fit(i)) == ytr(i));
end
tr_acc = 100*double(tr_sum/tr_p);

% Test accuracy
y_pred = y(Xte, wo);
te_sum = 0;
for i = 1:te_q
    te_sum = te_sum + (round(y_pred(i)) == yte(i));
end
te_acc = 100*double(te_sum/te_q);
tex=toc;

%fprintf('w* = %1.1f\n', wo);
fprintf('L* = %6.2d\n', fo);
fprintf('niter = %1.0f\n', niter);
fprintf('Train accuracy = %1.3f\n', tr_acc);
fprintf('Test accuracy = %1.3f\n', te_acc);

end