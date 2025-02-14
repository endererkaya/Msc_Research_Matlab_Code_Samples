%Proximal_Inf Operator
function[out]=prox(v,lambda)
[sv,index]=sort(abs(v),'descend');
sv=vertcat(sv,0);
phase=angle(v);
n=length(v);
out=v;
for r=1:n
    index_check=sum(sv(1:r))-sv(r+1)*(r)-lambda;
    if index_check>=0
        break
    end
end

if (r==n) && (sum(sv(1:n))-lambda <= 0)
    out=zeros(r,1);
else
    index_value=(sum(sv(1:r))-lambda)/(r);
    for t=1:r
        out(index(t),1)=index_value*exp(i*phase(index(t)));%r.exp(j*theta)
    end
end

%     A=tril(ones(n,n));
%     A=A+diag(-1*(1:(n-1)),1);
%     c=A*sv-lamb;%index checking vector
%     if c(1)>=0
%         v(index(1))=s(index(1))*(sv(1)-lamb);
%     end
%     if c(1)<0
%         in = find(c > 0, 1, 'first');
%         if in>1
%             x=(sum(sv(1:in))-lamb)/(in);
%             for t=1:in
%                 v(index(t))=s(index(t))*x;
%             end
%         end
%     end
% out=v;

% CVX Solution
% clear h;
% cvx_begin
% variable h(n)
% minimize((lamb*norm(h,inf))+0.5*(square_pos(norm(h-v))))
% cvx_end
% out=h;
%
% cvx_optimal  =lambda*norm(h,inf)+0.5*((norm(h-a)^2));
% bizim_optimal=lambda*norm(v,inf)+0.5*((norm(v-a)^2));
% error(w)=norm(v-h,1);

