% Accelerated Proximal Gradient Applied to dual problem
function [x,xfull] = Accelerated_Dual_Proximal_Gradient(A,b,T,lambda,t)
% % aka Fast Alternating Minimization Method
% % Solves the following problem via dual proximal gradient:
% %   minimize 1/2*|| Ax - b ||_2^2 + \lambda || T*x ||_inf
% % t is step size of the gradient z+=prox_tg*(z+tAx^)
% % The solution is returned in the vector x.
% %
% % history is a structure that contains the objective value, the primal and
% % dual residual norms, and the tolerances for the primal and dual residual
% % norms at each iteration.
% %
% % x,z alternating directions; u is dual variable.
% % Dual update is the proximal tg applied to gradient step

%     lambda=0.1;
    MAX_ITER=t;
    t_start = tic;
    
    % Step Size Selection
    e1=eig(A'*A);
    mu=min(e1);%min eigenvalue Hes(f)
    le=max(e1);
    Tmax=norm(T)^2;%max of T
    e2=eig(T'*T);
    Tmin=min(e2);
    LGA1=Tmax/mu;
    LGA2=Tmin/le;
    step_size=1/(lambda*LGA1);
    
    % Global constants and defaults
    QUIET    = 0;
    ABSTOL   = 1e-2;
    
    %Data preprocessing
    [m, n] = size(A);
    k=length(T(:,1));
    
    % save a matrix-vector multiply
    Atb = A'*b;
    
    %Alternating Minimization Method
    
    x = zeros(n,1);
    z = zeros(k,1);
    u = zeros(k,1);
    y = zeros(k,1);
    u_1=zeros(k,1);
    K=zeros(MAX_ITER+1);
    K(1)=1;
    % % cache the factorization
%     [L,U] = factor2(A);
    Tinv=pinv(T);
    
    if ~QUIET
        fprintf('%3s\t%6s\t%10s\t%10s\n', 'iter', ...
            'r norm', 's norm', 'objective');
    end
    
    for k = 1:MAX_ITER
        % x-update
        q = (Atb-T'*y) ;    % temporary value
        x = pinv(A'*A)*q;
        % z-update
        zold=z;
        z = prox(lambda*(T*x+y/step_size),(lambda^2)/step_size);
        % u-update
        u_1=u;
        u = y + step_size*(T*x - z/lambda);
        % k-update
        K(k+1)=(1 + sqrt(1 + 4*(K(k)^2)))/2;
        % y-update
        y = u + ((K(k)-1) / K(k+1)) * (u - u_1);
        
        % diagnostics, reporting, termination checks
        history.objval(k)  = objective(A, b, lambda, x, T);
        
        history.r_norm(k)  = norm(T*x - z);
        history.s_norm(k)  = norm(T'*(u - u_1));
        
        if ~QUIET
            fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\n', k, ...
                history.r_norm(k), ...
                history.s_norm(k), history.objval(k));
        end
        %     if k>1 && norm(history.objval(k)-history.objval(k-1))<ABSTOL
        %         break
        %     end
        xfull(k,:)=x;
%         error_objective(k)=abs(objective(A, b, lambda, x, T)-cvx_optval)/(cvx_optval);
%         error_x(k)=norm(x-x_opt)/norm(x_opt);
%         error_dualfun(k)=abs(abs(dual_objective(A,b,T,u))-cvx_optval)/cvx_optval;
    end
    
    if ~QUIET
        toc(t_start);
    end
    
%     timer_fdpgm(counter)=toc(t_start);
%     funconvergence_fdpgm(counter,:)=error_objective;
%     dfunconvergence_fdpgm(counter,:)=error_dualfun;
%     counter
% 
% %Primal Function Convergence
% funconvergence_fdpgm_mean=mean(funconvergence_fdpgm);
% save funconvergence_fdpgm_mean;
% save funconvergence_fdpgm;
% semilogy(funconvergence_fdpgm_mean);
% 
% %Dual Function Convergence
% dfunconvergence_fdpgm_mean=mean(dfunconvergence_fdpgm);
% save dfunconvergence_fdpgm_mean;
% save dfunconvergence_fdpgm;
% semilogy(dfunconvergence_fdpgm_mean);
% 
% %Save Timer
% time_fdpgm=mean(timer_fdpgm);
% save time_fdpgm;