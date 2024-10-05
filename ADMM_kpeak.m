function [y, x, history] = ADMM_kpeak(A, b, T, lambda, peak_no, NUM_OF_ITER)
% lasso  Solve lasso problem via ADMM
%
% [z, history] = lasso(A, b, lambda, rho, alpha);
%
% Solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
% clear all;
% delete time_ADMM_wrelax.mat;
% delete funconvergence_admm_wrelax.mat;
% delete funconvergence_admm_wrelax_mean.mat;
% delete dfunconvergence_admm_wrelax.mat;
% delete dfunconvergence_admm_wrelax_mean.mat;
% 
% for counter=1:2
MAX_ITER=NUM_OF_ITER;
% Random_Problem_Generation
rho=0.5*peak_no;
alpha=1.5;
t_start = tic;

%Global constants and defaults
QUIET    = 0;
ABSTOL   = 1e-2;
RELTOL   = 1e-4;

%Data preprocessing
[m, n] = size(A);
p=length(T(:,1));
% save a matrix-vector multiply
Atb = A'*b;

%ADMM solver

x = zeros(n,MAX_ITER);
z = zeros(p,1);
u = zeros(p,1);

% cache the factorization
% [L,U,Lt,Ut] = factor(A,T,rho);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER
    % x-update
    q = Atb + rho*T'*(z - u);    % temporary value
    if( m >= n )    % if skinny
%         x(:,k) = U \ (L \ q);
            x(:,k)=inv(A'*A+rho*(T'*T))*q;
        
    else            % if fat
%         x(:,k) = Ut\(Lt\(q/rho)) - ((Ut\(Lt\(A'*(U\(L\(A*(Ut\(Lt\q)))))))))/rho^2;
            x(:,k)=inv(A'*A+rho*(T'*T))*q;
    end
    
    % z-update with relaxation
    zold = z;
    Ax_hat = alpha*T*x(:,k) + (1 - alpha)*zold;
    if(peak_no==1)
        z=prox(T*x(:,k)+u,lambda/rho);
    else
        z = proxkpeak(T*x(:,k)+u,lambda/rho,peak_no);
    end
    % u-Dual Update
    u = u +(Ax_hat - z);
    
%     if k>1
%        if (c(k) < nu*c(k-1))
%                 
%        % k-update
%        K(k+1)=(1+sqrt(1+4*(K(k)^2)))/2;
%                 
%        % z-accelerate
%        ze(:,k+1) = z(:,k)+((K(k)-1)/K(k+1))*(z(:,k)-z(:,k-1));
%                 
%        % y-update
%        ue(:,k+1) = u(:,k)+((K(k)-1)/K(k+1))*(u(:,k)-u(:,k-1));
%                 
%        else
%            K(k+1)=1;
%            ze(:,k+1)=z(:,k-1);
%            ue(:,k+1)=u(:,k-1);
%                 
%            c(k)=(1/nu)*c(k-1);
%        end
%     end
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, x(:,k), T);
    
    history.r_norm(k)  = norm(T*x(:,k) - z);
    history.s_norm(k)  = norm(rho*T'*(z - zold));
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(T*x(:,k)), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*T'*u);
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
    
    %     if (history.r_norm(k) < history.eps_pri(k) && ...
    %             history.s_norm(k) < history.eps_dual(k))
    %         break;
    %     end
%     error_objective(k)=abs(objective(A, b, lambda, x, T)-cvx_optval)/(cvx_optval);
%     error_dualfun(k)=abs(abs(dual_objective(A,b,T,rho*u))-cvx_optval)/cvx_optval;
%     error_x(k)=norm(x-x_opt)/norm(x_opt);
end

if ~QUIET
    toc(t_start);
end

% timer_ADMM_wrelax(counter)=toc(t_start);
% funconvergence_admm_wrelax(counter,:)=error_objective;
% dfunconvergence_admm_wrelax(counter,:)=error_dualfun;
y=x(:,MAX_ITER);
end 
%Primal Function Convergence
% save funconvergence_admm_wrelax;
% funconvergence_admm_wrelax_mean=mean(funconvergence_admm_wrelax);
% save funconvergence_admm_wrelax_mean;
% % semilogy(funconvergence_admm_wrelax_mean,'r');
% 
% %Dual Function Convergence
% save dfunconvergence_admm_wrelax;
% dfunconvergence_admm_wrelax_mean=mean(dfunconvergence_admm_wrelax);
% save dfunconvergence_admm_wrelax_mean;
% semilogy(dfunconvergence_admm_wrelax_mean,'r');
% 
% %Save Timer
% time_ADMM_wrelax=mean(timer_ADMM_wrelax);
% save time_ADMM_wrelax;