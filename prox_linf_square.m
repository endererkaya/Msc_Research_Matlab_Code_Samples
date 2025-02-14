%Proximal_Inf Operator
function[out]=prox(v,lambda)
[sv,index]=sort(abs(v),'descend');
sv=vertcat(sv,0);
phase=angle(v);
n=length(v);
out=v;
for k=1:n
    xopt = sum(sv(1:k))/(k+2*lambda);
    if sv(k+1) < xopt && sv(k) >= xopt % Clipping operation
        break
    end
end
xopt = sum(sv(1:k))/(k+2*lambda);

    for t=1:k
        out(index(t),1)=xopt*exp(i*phase(index(t)));%r.exp(j*theta)
    end
end

% CVX Solution
% clear h;
% cvx_begin
% variable h(n)
% minimize((lambda*square_pos(norm(h,inf)))+0.5*(square_pos(norm(h-v))))
% cvx_end
% out=h;
%
% cvx_optimal  =lambda*norm(h,inf)^2+0.5*((norm(h-a)^2));
% bizim_optimal=lambda*norm(v,inf)^2+0.5*((norm(v-a)^2));
% error(w)=norm(v-h,1);
