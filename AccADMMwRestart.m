% function [x, history] = ADMM_Linf(A, b, T, lambda, rho, alpha)
%%
% Solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda || T*x ||_inf
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
% clear all
% for c=1:1
% Random_Problem_Generation
clear all;
delete time_AccADMM.mat;
delete funconvergence_accadmm.mat;
delete funconvergence_accadmm_mean.mat;
delete dfunconvergence_accadmm.mat;
delete dfunconvergence_accadmm_mean.mat;

for counter=1:10
    lambda=1;
    MAX_ITER=500;
    Random_Problem_Generation
    rho=1;
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
    
    % Step Length Chooser
    e1=eig(A'*A);
    mu=min(e1);%min eigenvalue Hes(f)
    le=max(e1);
    Tmax=norm(T)^2;%max of T
    Tmin=min(eig(T'*T));
    
    %ADMM solver
    
    x = zeros(n,MAX_ITER+1);
    z = zeros(p,MAX_ITER+1);
    u = zeros(p,MAX_ITER+1);
    ue= zeros(p,MAX_ITER+1);
    ze= zeros(p,MAX_ITER+1);
    c=zeros(MAX_ITER+1);
    K=zeros(MAX_ITER+1);
    K(1)=1;
    nu=0.999;
    c(1)=0;
    % cache the factorization
    [L,U,Lt,Ut] = factor(A,T,rho);
    
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
            'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
    end
    
    for k = 1:MAX_ITER
        % x-update
        q = Atb + rho*T'*(ze(:,k) - ue(:,k));    % temporary value
        if( m >= n )    % if skinny
            x = U \ (L \ q);
            %     x=inv(A'*A+rho*(T'*T))*q;
            
        else            % if fat
            x = Ut\(Lt\(q/rho)) - ((Ut\(Lt\(A'*(U\(L\(A*(Ut\(Lt\q)))))))))/rho^2;
            %     x=inv(A'*A+rho*(T'*T))*q;
        end
        
        % z-update with relaxation
        z(:,k) = prox(T*x+ue(:,k),lambda/rho);
        
        % u-update
        u(:,k) = ue(:,k) + (T*x - z(:,k));
                
        if k>1
            if (c(k) < nu*c(k-1))
                
                % k-update
                K(k+1)=(1+sqrt(1+4*(K(k)^2)))/2;
                
                % z-accelerate
                ze(:,k+1) = z(:,k)+((K(k)-1)/K(k+1))*(z(:,k)-z(:,k-1));
                
                % y-update
                ue(:,k+1) = u(:,k)+((K(k)-1)/K(k+1))*(u(:,k)-u(:,k-1));
                
            else
                K(k+1)=1;
                ze(:,k+1)=z(:,k-1);
                ue(:,k+1)=u(:,k-1);
                
                c(k)=(1/nu)*c(k-1);
            end
            
            % diagnostics, reporting, termination checks
            history.objval(k)  = objective(A, b, lambda, x, T);
            
            history.r_norm(k)  = norm(T*x - z(:,k));
            
            history.s_norm(k)  = norm(-rho*T'*((z(:,k) - z(:,k-1))));
            
            history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(T*x), norm(-z(:,k)));
            history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*T'*u(:,k));
            
            if ~QUIET
                fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
                    history.r_norm(k), history.eps_pri(k), ...
                    history.s_norm(k), history.eps_dual(k), history.objval(k));
            end
        end
        %     if (history.r_norm(k) < history.eps_pri(k) && ...
        %             history.s_norm(k) < history.eps_dual(k))
        %         break;
        %     end
        error_objective(k)=abs(objective(A, b, lambda, x, T)-cvx_optval)/(cvx_optval);
        error_dualfun(k)=abs(abs(dual_objective(A,b,T,rho*u(:,k)))-cvx_optval)/cvx_optval;
        error_x(k)=norm(x-x_opt)/norm(x_opt);
    end
    
    if ~QUIET
        toc(t_start);
    end
    
    timer_AccADMM(counter)=toc(t_start);
    funconvergence_accadmm(counter,:)=error_objective;
    dfunconvergence_accadmm(counter,:)=error_dualfun;
end

%Primal Function Convergence
save funconvergence_accadmm;
funconvergence_accadmm_mean=mean(funconvergence_accadmm);
save funconvergence_accadmm_mean;
% semilogy(funconvergence_accadmm_mean);

%Dual Function Convergence
save dfunconvergence_accadmm;
dfunconvergence_accadmm_mean=mean(dfunconvergence_accadmm);
save dfunconvergence_accadmm_mean;
semilogy(dfunconvergence_accadmm_mean);

%Save Timer
time_AccADMM=mean(timer_AccADMM);
save time_AccADMM;