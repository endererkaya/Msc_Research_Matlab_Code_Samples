%Proximal K-peak Magnitude Operator
function[out]=proxkpeak(v,lambda,peak_no)
[sv,index]=sort(abs(v),'descend');
sv=vertcat(sv,0);
phase=angle(v);
n=length(v);
check=zeros(peak_no,1);
out=v;
for r=1:peak_no
    check(r)=sv(r)-sv(r+1)-lambda;
end
if isempty(check(check>0))
        for ind=0:peak_no-2
            check2=sv(peak_no-ind-1)-lambda-max(abs(prox(sv(peak_no-ind:n),(ind+1)*lambda)));
            if check2>0
                break;
            end
        end
        if ind==peak_no-2
            if check2>0
            last_index=peak_no-ind;
            temp(1:last_index-1)=sv(1:last_index-1)-lambda*ones(last_index-1,1);
                temp(last_index:n)=prox(sv(last_index:n),(ind+1)*lambda);
            else
                temp(1:n)=prox(sv(1:n),peak_no*lambda);
            end
        else
            last_index=peak_no-ind;
            temp(1:last_index-1)=sv(1:last_index-1)-lambda*ones(last_index-1,1);
            temp(last_index:n)=prox(sv(last_index:n),(ind+1)*lambda);
        end
else
    k=find(check>=0,1,'last');
    if k<peak_no
        K=peak_no-k;
        for ind=0:K-1
            check2=sv(peak_no-ind-1)-lambda-max(abs(prox(sv(peak_no-ind:n),(ind+1)*lambda)));
            if check2>0
                break;
            end
        end
        last_index=peak_no-ind;
        temp(1:last_index-1)=sv(1:last_index-1)-lambda*ones(last_index-1,1);
        temp(last_index:n)=prox(sv(last_index:n),(ind+1)*lambda);
    else
        temp(1:k)=sv(1:k)-lambda*ones(k,1);
        temp(k+1:n)=sv(k+1:n);
    end
end
for t=1:n
    out(index(t))=temp(t)*exp(i*phase(index(t)));
end

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

%     cvx_begin
%     variable h(p)
%     minimize((sum_largest(abs(h),peak_no))+(1/(2*lambda))*sum_square(h-v));
%     cvx_end
%     out2=h;
