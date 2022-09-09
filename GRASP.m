function [reject_finite,reject_asym,p_val_finite,p_val_asym]=GRASP(X,Y,heta_val,tau,alpha, f_div,L)

%GRASP return p_values (finite and asymptoticly valid) for the null hypothesis.
%Inputs: 
%X: feature matrix with size n by p (n=#samples and p = #features)
%Y: Binary response values of size n. 
%heta_val= Output of test model heta on features X. 
%tau: the tau value in the hypothesis testing problem
%alpha: predetermined significance level 
%f_div: the f-divergence function. It can be "H" (Hellinger distance), "kl" (kl-divergence, or "tv" (total variation).
%L: number of labels for statistics V_{n,L}


%Outputs:
% p_val_finite: a p-value for the hypothesis testing problem which is valid in finite-sample regime.
%p_val_asym:  a p-value for the hypothesis testing problem which is valid in asymptotic regime.
%reject_finie: rejection status based on p_val_finite atsignificance level alpha.
%reject_asym: rejection status based on p_val_asym at significance level alpha.



% define default values, in case of misspecified values. 
if ~exist('alpha','var'), alpha=0.1; end
if ~exist('L','var'), L=50; end
if ~exist('f_div','var'), f_div="tv"; end

fprintf("Running distribution_free GRASP with (L, alpha, f_div, tau) = (%d, %.2f,%s,%.2f).\n", L, alpha, f_div,tau);
    assert(tau >=0); 

reject_finite=false;
reject_asym=false;

[n,p]=size(X);
tmp=unifrnd(0,1,[n,1]);
W=2* tmp .* Y .* heta_val- Y .* heta_val-tmp .* heta_val - Y.* tmp + tmp + heta_val;    %generating randomizations W     
V=zeros(L,1);  %defining statistics V
for (ell = 1:L)
  V(ell)=sum( W<=(ell/L) & W>((ell-1)/L) );
end

cvx_solve_error_flag=0;


switch f_div
%%total variation distance
case {'tv'}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% asymptotic
    cvx_solver SeDuMi
    cvx_precision default
    cvx_begin 
      variable my_p(L);
    obj=0;
      for (i=1:L)
         obj=obj+(1/L)*quad_over_lin(V(i)/n-my_p(i), my_p(i) ); 
      end
      minimize(obj)
      subject to 
      sum(my_p .* ones([L,1]))==1;
      my_p >= 0 ;
      norm(my_p-(1/L)*ones([L,1]),1)<= 2*tau;
      cvx_end
      if ( ~(cvx_status == "Solved")) cvx_solve_error_flag=1; end
      T_asym=(1/n) * sum( ((V-n*my_p).^2) ./ my_p);
      if(T_asym>chi2inv(1-alpha,L-1))
          reject_asym=true;
      end
    
      p_val_asym=1-chi2cdf(T_asym,L-1);
      
      %% finite 
      cvx_solver SeDuMi
      cvx_precision default
      cvx_begin 
      variable my_p(L);
      obj=0;
      for (i=1:L) 
         obj=obj+(1/L)*quad_over_lin(V(i)/n-my_p(i), 1/L + my_p(i) );
      end
      minimize( obj );
      subject to 
      sum(my_p .* ones([L,1]))==1;
      my_p >= 0 ;
      norm(my_p-(1/L)*ones([L,1]),1)<=2*tau;
      cvx_end
      if ( ~(cvx_status == "Solved")) cvx_solve_error_flag=1;  end
      T_finite= (1/n) * sum( (V-n*my_p).^2 ./ (ones([L,1])/L + my_p) );
      if(T_finite> L+sqrt(2*L/alpha) )
          reject_finite=true;
      end
      if(T_finite>=L) 
          p_val_finite=min(1, 2*L/(T_finite-L)^2);
      else p_val_finite=1; 
      end
%%kl divergence
case {'kl'}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% finite
    cvx_solver SeDuMi
    cvx_precision default
    cvx_begin 
    variable my_p(L);
    obj=0;
      for (i=1:L)
         obj=obj+(1/L)*quad_over_lin(V(i)/n-my_p(i), 1/L + my_p(i) );
      end
      minimize(obj);
      subject to
      sum(my_p .* ones([L,1]))==1;
      my_p >= 0;
      sum(entr(my_p))>=log(L)-tau;
      cvx_end
      if ( ~(cvx_status == "Solved")) cvx_solve_error_flag=1; end
      T_finite= (1/n) * sum( (V-n*my_p).^2 ./ (ones([L,1])/L + my_p) );
      if(T_finite> L+sqrt(2*L/alpha))
          reject_finite=true;
      end
      if(T_finite>=L) 
          p_val_finite=min(1, 2*L/(T_finite-L)^2);
      else p_val_finite=1; 
      end
    
      %% asymptotic
      cvx_solver SeDuMi
      cvx_precision default
      cvx_begin 
      variable my_p(L);
      obj=0;
      for (i=1:L)
         obj=obj+(1/L)*quad_over_lin(V(i)/n-my_p(i), my_p(i) ); 
      end
      minimize(obj)
      subject to
      sum(my_p .* ones([L,1]))==1;
      my_p >= 0;
      sum(entr(my_p))>=log(L)-tau;
      cvx_end
      if ( ~(cvx_status == "Solved")) cvx_solve_error_flag=1; end
      T_asym= (1/n) * sum( ((V-n*my_p).^2) ./ my_p);
      if(T_asym>chi2inv(1-alpha,L-1))
          reject_asym=true;
      end
      
      p_val_asym=1-chi2cdf(T_asym,L-1);
      
%%Hellinger distance    
case{'H'}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% asymptotic
    cvx_solver SeDuMi
    cvx_precision default
    cvx_begin 
    variable my_p(L);
    obj=0;
      for (i=1:L)
         obj=obj+(1/L)*quad_over_lin(V(i)/n-my_p(i), my_p(i) ); 
      end
      minimize(obj);
      subject to
      sum(my_p .* ones([L,1]))==1.00;
      my_p >= 0;
      sum(sqrt(my_p).*ones([L,1]))>= sqrt(L) * (1-tau/2);
      cvx_end
      if ( ~(cvx_status == "Solved")) cvx_solve_error_flag=1; end
      T_asym=(1/n) * sum( ((V-n*my_p).^2) ./ my_p);
      if(T_asym>chi2inv(1-alpha,L-1))
          reject_asym=true;
      end
      p_val_asym=1-chi2cdf(T_asym,L-1);
      
      
     %% finite
      cvx_solver SeDuMi
      cvx_precision default
      cvx_begin 
      variable my_p(L);
      obj=0;
      for (i=1:L)
         obj=obj+(1/L)*quad_over_lin(V(i)/n-my_p(i), 1/L + my_p(i) ); 
      end
      minimize(obj);
      subject to
      sum(my_p .* ones([L,1]))==1;
      my_p >= 0;
      sum(sqrt(my_p))>= sqrt(L) * (1-tau/2);
      cvx_end
      if ( ~(cvx_status == "Solved")) cvx_solve_error_flag=1; end
      T_finite=(1/n) * sum( (V-n*my_p).^2 ./ (ones([L,1])/L + my_p) );
      if(T_finite> L+sqrt(2*L/alpha))
          reject_finite=true;
      end
      if(T_finite>=L) 
          p_val_finite=min(1, 2*L/(T_finite-L)^2);
      else p_val_finite=1; 
      end
      
      
    
      
      
      
      
     otherwise
   fprintf('divergence function not found');
end
if(cvx_solve_error_flag ==1)  fprintf('cvx optimizer failed \n'); end
end



