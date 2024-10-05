% Optimal Linear Phase Filter Design
clear all
%Constants, Defaults, Initializations
N=20; %Number of filter coef.(positive side)
F=zeros(100,N+1); 
m=zeros(100,1);
w=0.01*pi:0.01*pi:pi;% discrete normalized angular frequency
sup=90; % suppression level (dB)
NUM_OF_ITER=2; %Number of iterations+1 (for ADMM&&SG) 
NUM_OF_SIM=1000; %Number of simulations (for time-mean)

% creating weighting matrix
W=diag(horzcat(ones(1,30),zeros(1,20),ones(1,20),zeros(1,15),ones(1,15)));
for i=1:30
    m(i)=10^(0/20);
end
for i=51:70
    m(i)=10^(-sup/20);
end
for i=86:100
    m(i)=10^(0/20);
end

% creating freq. response matrix
F(:,1)=1;
for i=1:100
    for k=2:N+1
        F(i,k)=2*cos((k-1)*w(i));
    end
end
mdb=20*log10(m);

%Weighted Cost Function
W1=W./diag(m);
W1(isnan(W1))=0;
% W1=eye(100);

A=-W1*F;
b=W1*m;

% calculate with cvx
cvx_begin
variable h2(N+1) complex
minimize( norm((A*h2+b),inf) )
cvx_end
save h2;
primal_opt=h2;
fun_opt=cvx_optval;

% %calculate with Subgradient
% [x_SG, timer_SG] = Subgradient(A, b, NUM_OF_ITER);
% for i=1:length(x_SG(1,:))
%     fun_SG(i)=abs(norm(A*x_SG(:,i)+b,inf)-fun_opt)/abs(fun_opt);
%     primal_SG(i)=norm(abs(x_SG(:,i)-primal_opt)/(primal_opt));
% end
% h_SG=x_SG(:,NUM_OF_ITER); 

% calculate with ADMM
time_ADMM=[];
for j=1:NUM_OF_SIM
    [x_ADMM, timer_ADMM] = ADMM(A, b, NUM_OF_ITER);
    time_ADMM=horzcat(time_ADMM,timer_ADMM);
end
for i=1:length(x_ADMM(1,:))
    fun_ADMM(i)=abs(norm(A*x_ADMM(:,i)+b,inf)-fun_opt)/abs(fun_opt);
    primal_ADMM(i)=norm(abs(x_ADMM(:,i)-primal_opt))/norm(abs(primal_opt));
end
h_ADMM=x_ADMM(:,NUM_OF_ITER);
h_ADMM=real(h_ADMM);

% calculate with Parks-McClellan Algorithm
w_firpm=[0,0.3,0.5,0.7,0.85,1];
m_firpm=[1,1,10^(-sup/20),10^(-sup/20),1,1];
time_firpm=[];
for j=1:NUM_OF_SIM
    t_start=tic;
    h_firpm=fir2(2*N,w_firpm,m_firpm);
    time_firpm=horzcat(time_firpm,toc(t_start));
end

% %Compare Convergences
% semilogy(primal_SG,'b');
% hold on
% semilogy(primal_ADMM,'r');
% hold on
% ylabel('|f(xk)-f(x*)|/|f(x*)|');
% xlabel('Number of iterations');
% legend('SG','ADMM');

%Compare Frequency Responses
% H_SGdb=20*log10(abs(F*h_SG));
H_CVXdb=20*log10(abs(F*primal_opt));
H_ADMMdb=20*log10(abs(F*h_ADMM));
h_ADMM_all=vertcat(flipud(h_ADMM),h_ADMM(2:N+1));
H_ADMM2=freqz(h_ADMM_all,1,2048);

H_FIRPM=freqz(h_firpm,1,2048);
h_firpmpos=(fliplr(h_firpm(1:N+1)))';
H_FIRPM2=20*log10(abs(F*h_firpmpos));

% Plotting & Comparison
plot([0:1/2048:1-1/2048],20*log10(abs(H_FIRPM)),'b');
% plot(w/pi,H_FIRPM2,'b');
hold on
% plot(w/pi,mdb,'--y');
% hold on
plot([0:1/2048:1-1/2048],20*log10(abs(H_ADMM2)),'r');
% plot(w/pi,H_ADMMdb,'r')
hold on
xlabel('${w(\pi)}$','Interpreter','Latex');
ylabel('${H(e^{jw})_{db}}$','Interpreter','Latex');
t_ADMM=mean(time_ADMM)
t_firpm=mean(time_firpm)
legend('FIRPM','ADMM');






%Part A
% ELEC-505 HW Part A
%Infinity Norm Cost Function
% cvx_begin
% variable h1(N+1)
% minimize( norm((W)*(m-F*h1),inf) )
% cvx_end
% 
% H1db=20*log10(abs(F*h1));

% plot(w/pi,mdb,'*r');
% hold on
% plot(w/pi,H1db);
% xlabel('w(\pi)');
% ylabel('H1(e^jw)db');

%Part C
%Added penalty with L1 of h
% lambda=0.7;
%
% cvx_begin
% variable h3(N+1)
% minimize(norm((W1)*(m-F*h3),inf)+lambda*norm(h3,1))
% cvx_end
%
% H3db=20*log10(abs(F*h3));

% plot(w/pi,mdb,'*r');
% hold on
% plot(w/pi,H3db);
% xlabel('w(\pi)');
% ylabel('H3(e^jw)db');

% %Part D
% %To make little coefficients exactly zero
%
% Ind=h3;%Indicator vector
% Ind=abs(Ind)<1e-3;
% I=diag(Ind);
%
% lambda=0.7;
%
% cvx_begin
% variable h4(N+1)
% minimize(norm((W1)*(m-F*h4),inf)+lambda*norm(h4,1))
% subject to
% I*h4==zeros(N+1,1)
% cvx_end
%
% H4db=20*log10(abs(F*h4));
%
% plot(w/pi,mdb,'*r');
% hold on
% plot(w/pi,H4db);
% xlabel('w(\pi)');
% ylabel('H4(e^jw)db');
