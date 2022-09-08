%%%power of model-X
clear all;
rng('default');
alpha=0.1; %significance level
p=200;   %feature dimension
th0=normrnd(0,1,[p,1])/4; %ground truth logistic model parameter
th1=-th0;  % test estimate logistic model parameter 



%% hypothesis testing
%tau values for hypothesis testing
tau_kl_asym=1.9;
tau_tv_asym=0.67;
tau_hlg_asym=0.76;
tau_kl_finite=1.5;
tau_tv_finite=0.6;
tau_hlg_finite=0.58;


 
n=5000; %validation data size
K=1;    % randomization per label
L=100;  % #labels 
           
M=K*L-1;  % #number of randomizations        
rejected_kl_asym=false;
rejected_tv_asym=false;
rejected_hlg_asym=false;
rejected_kl_finite=false;
rejected_tv_finite=false;
rejected_hlg_finite=false;
my_flag=0;  %check for CVX optimization solve status: 0=ok, 1=CVX fails to solve
%% validation data generation 
X=normrnd(0,1,[n,p]);  %covariates
Y=zeros([n,1]);        %labels
Z=1./(1+exp(-(X*th0)));
Y=(unifrnd(0,1,[n,1])<Z);
%% W auxilary samples
w=zeros([n,1]);      
heta_val=1./(1+exp(-X*th1));  %
tmp=unifrnd(0,1,[n,1]);
w=2* tmp .* Y .* heta_val- Y .* heta_val-tmp .* heta_val - Y.* tmp + tmp + heta_val; 
my_label=zeros(n,1);


%% scoring original samples  
prod_1=X*th1;
heta_o=1./(1+exp(-prod_1)); %original heta values
score_o=(w <= heta_o) .* 0.5 ./ heta_o + (w >= heta_o) .* (0.5) ./ (1-heta_o); %scoring original samples

%% scoring counterfeits
prod_1_c=zeros(n,M); %store <x,\th_1> in each entry for x~N(0,I_p) 
%next step computes it efficiently with tensor product
NN=10; 
tmp_tensor=zeros([M,NN,n/NN]);
for (j=1:(n/NN))
X_c=normrnd(0,1,[M,p,NN]);
tmp_tensor(:,:,j)=squeeze(pagemtimes(X_c,th1));
end
prod_1_c=tmp_tensor(:,:)';
heta_c=1./(1+exp(-prod_1_c)); %heta values of counterfeit covariates x; size is n by M
w_c=unifrnd(0,1,[n,M]);   %uniformly sampled counterfeits w 
scores_c=(w_c<=heta_c) .* 0.5 ./ heta_c + (w_c>heta_c) .* 0.5 ./ (1-heta_c) ; %score of counterfeits
%% Labeling
  tmp=sum((score_o>scores_c),2);
  my_label=floor(tmp/K)+1;
%% V_{n,L} statistics 
V=zeros(L,1);  
for (ell = 1:L)
V(ell)=sum(my_label==ell);
end




%% TV hypothesis testing
%asym
cvx_precision default
cvx_begin quiet
      variable my_p(L);
      minimize( -n+ sum(inv_pos(my_p).*(V.^2)/n) );
      subject to 
      sum(my_p .* ones([L,1]))==1;
      my_p >= 0 ;
      norm(my_p-(1/L)*ones([L,1]),1)<= 2*tau_tv_asym;
cvx_end
if ( ~(cvx_status == "Solved")) my_flag=1; end

T_tv_asym=(1/n) * sum( ((V-n*my_p).^2) ./ my_p);
if(T_tv_asym>chi2inv(1-alpha,L-1))
    rejected_tv_asym=true;
end
p_val_tv_asym=1-chi2cdf(T_tv_asym,L-1);

%finite
cvx_precision default
cvx_begin quiet
      variable my_p(L);
      obj=0;
      for (i=1:L)
         obj=obj+(1/L)*quad_over_lin(V(i)/n-my_p(i), 1/L + my_p(i) );
      end
      minimize( obj );
      subject to 
      sum(my_p .* ones([L,1]))==1;
      my_p >= 0 ;
      norm(my_p-(1/L)*ones([L,1]),1)<=2*tau_tv_finite;
cvx_end
if ( ~(cvx_status == "Solved")) my_flag=1; end

T_tv_finite= (1/n) * sum( (V-n*my_p).^2 ./ (ones([L,1])/L + my_p) );
if(T_tv_finite> L+sqrt(2*L/alpha) )
    rejected_tv_finite=true;
end
if(T_tv_finite>=L) 
          p_val_tv_finite=min(1, 2*L/(T_tv_finite-L)^2);
      else p_val_tv_finite=1; 
      end

%% KL hypothesis testing
% asym
cvx_precision default
cvx_begin quiet
variable my_p(L);
minimize( -n+ sum(inv_pos(my_p).*(V.^2)/n) );
subject to
sum(my_p .* ones([L,1]))==1;
my_p >= 0;
sum(entr(my_p))>=log(L)-tau_kl_asym;
cvx_end
if ( ~(cvx_status == "Solved")) my_flag=1; end


T_kl_asym= (1/n) * sum( ((V-n*my_p).^2) ./ my_p);
if(T_kl_asym>chi2inv(1-alpha,L-1))
    rejected_kl_asym=true;
end
p_val_kl_asym=1-chi2cdf(T_kl_asym,L-1);
% finite 
cvx_precision default
cvx_begin quiet
variable my_p(L);
obj=0;
      for (i=1:L) 
         obj=obj+(1/L)*quad_over_lin(V(i)/n-my_p(i), 1/L + my_p(i) );
      end
      minimize(obj);
subject to
sum(my_p .* ones([L,1]))==1;
my_p >= 0;
sum(entr(my_p))>=log(L)-tau_kl_finite;
cvx_end

if ( ~(cvx_status == "Solved")) my_flag=1; end

T_kl_finite= (1/n) * sum( (V-n*my_p).^2 ./ (ones([L,1])/L + my_p) );
if(T_kl_finite> L+sqrt(2*L/alpha))
    rejected_kl_finite=true;
end

if(T_kl_finite>=L) 
          p_val_kl_finite=min(1, 2*L/(T_kl_finite-L)^2);
      else p_val_kl_finite=1; 
      end


%% HLG hypothesis testing
% asym
cvx_precision default
cvx_begin quiet
variable my_p(L);
minimize( -n+ sum(inv_pos(my_p).*(V.^2)/n) );
subject to
sum(my_p .* ones([L,1]))==1;
my_p >= 0;
sum(sqrt(my_p))>= sqrt(L) * (1-tau_hlg_asym/2);
cvx_end

if ( ~(cvx_status == "Solved")) my_flag=1; end
T_hlg_asym=(1/n) * sum( ((V-n*my_p).^2) ./ my_p);
if(T_hlg_asym>chi2inv(1-alpha,L-1))
    rejected_hlg_asym=true;
end
p_val_hlg_asym=1-chi2cdf(T_hlg_asym,L-1);

%% finite
cvx_precision default
cvx_begin quiet
variable my_p(L);
obj=0;
      for (i=1:L)
         obj=obj+(1/L)*quad_over_lin(V(i)/n-my_p(i), 1/L + my_p(i) ); 
      end
minimize(obj);
subject to
sum(my_p .* ones([L,1]))==1;
my_p >= 0;
sum(sqrt(my_p))>= sqrt(L) * (1-tau_hlg_finite/2);
cvx_end

if ( ~(cvx_status == "Solved")) my_flag=1; end
T_hlg_finite=(1/n) * sum( (V-n*my_p).^2 ./ (ones([L,1])/L + my_p) );

if(T_hlg_finite> L+sqrt(2*L/alpha))
    rejected_hlg_finite=true;
end

if(T_hlg_finite>=L) 
          p_val_hlg_finite=min(1, 2*L/(T_hlg_finite-L)^2);
      else p_val_hlg_finite=1; 
      end
%% printing summary

fprintf("\n\n\n\nModel-X GRASP with n=%d,K=%d,L=%d,alpha=%.2f\n", n, K, L,alpha)
fprintf("tau values:\n")
fprintf("\t tau_kl_asym= %.2f, \t tau_tv_asym= %.2f, \t tau_hlg_asym= %.2f, \n", tau_kl_asym, tau_tv_asym, tau_hlg_asym)
fprintf("\t tau_kl_finite= %.2f \t tau_tv_finite= %.2f \t tau_hlg_finite= %.2f \n", tau_kl_finite, tau_tv_finite, tau_hlg_finite)
fprintf("rejection summary:\n")
fprintf("\t rejection_kl_asym= %s,\t rejection_tv_asym= %s, \t rejection_hlg_asym=%s \n",...
    string(rejected_kl_asym),string(rejected_tv_asym), string(rejected_hlg_asym))
fprintf("\t rejection_kl_finite= %s,\t rejection_tv_finite= %s, \t rejection_hlg_finite=%s \n",...
    string(rejected_kl_finite),string(rejected_tv_finite), string(rejected_hlg_finite))
fprintf("p_values:\n")
fprintf("\t p_val_kl_asym= %.4f, \t p_val_tv_asym= %.4f, \t p_val_hlg_asym= %.4f, \n", p_val_kl_asym,p_val_tv_asym, p_val_hlg_asym)
fprintf("\t p_val_kl_finite= %.4f,\t p_val_tv_finite= %.4f, \t p_val_hlg_finite= %.4f \n", p_val_kl_finite, p_val_tv_finite, p_val_hlg_finite)




















