rng('default');
clear all;
p=200;
n=5000;
L=50;
th0=normrnd(0,1,[p,1])/4;
th1=normrnd(0,1,[p,1])/4;
th1=-th0;
X=normrnd(0,1,[n,p]);
Y=zeros([n,1]);
Z=1./(1+exp(-(X*th0)));
Y=(unifrnd(0,1,[n,1])<Z);
R=zeros([n,1]);
my_prob=1./(1+exp(-X*th1));
heta_val=my_prob;
tau=0.3;
alpha=0.1;
f_div="H";
[reject_finite,reject_asym,p_val_finite,p_val_asym]=GRASP(X,Y,heta_val,tau,alpha,f_div,L);
fprintf("\n\n\n\n\nRunning distribution_free GRASP with (n, alpha, f_div, tau, L) = (%d, %.2f,%s,%.2f,%d).\n",...
n, alpha, f_div,tau,L);

fprintf("rejection summary:\n")
fprintf("\t rejection_asym= %s \n",string(reject_asym))
fprintf("\t rejection_finite= %s \n",string(reject_finite))
fprintf("p_values:\n")
fprintf("\t p_val_asym= %.4f \n", p_val_asym)
fprintf("\t p_val_finite= %.4f \n", p_val_finite)
