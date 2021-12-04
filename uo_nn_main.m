% NN main
% Marcel, Mengxue
% OTDM-NN-Nov21

clear;

%
% Parameters for dataset generation
%
num_target = [1:10];  % [1:10] % 10 és 0
tr_freq    = .5;      % .0  
tr_p       = 250;     % 20000   
te_q       = 250;     % tr_p/10  
tr_seed    = 57052680;    
te_seed    = 35520487;    

%
% Parameters for optimization
%
la = .01;                                                      % L2 regularization.
epsG = 10^-6; kmax = 1000;                                    % Stopping criterium.
ils=3; ialmax = 2; kmaxBLS=30; epsal=10^-3;c1=0.01; c2=0.45;  % Linesearch.
isd = 7; icg = 2; irc = 2 ; nu = 1.0;                         % Search direction.
sg_seed = 350415; sg_al0 = 2; sg_be = 0.3; sg_ga = 0.01;      % SGM iteration.
sg_emax = kmax; sg_ebest = floor(0.01*sg_emax);               % SGM stopping condition.

%
% Optimization
%
t1=clock;
[Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
t2=clock;
fprintf('wall time = %6.1d s.\n', etime(t2,t1));

%uo_nn_Xyplot(Xtr,ytr,wo);

%


