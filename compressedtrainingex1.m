%function [SIRc,norm_wco,norm_zc,SINRc,actualerrorc,BRECompared,SIR,norm_w,norm_z,SINR,actualerror,BREProposed,SIR1,norm_w1,...
%                norm_z1,SINR1,actualerror1,BRELeast,SIRmmse,norm_wmmse,norm_zmmse,SINRmmse,actualerrormmse,BREMMSE] = ...
%                breCalculation1(LD,LE,LC,LT,LData,constellation,EqDelay,noiseamp,H,lambd,alf,muCMA,muSDD,ro,L)
alf = 1;
QAM16 = 1;
BPSK  = 2;
QAM4 = 3;
% Channel Length
LC = 15;
% Equalizer Length
LE = 20;
% 
lambd=30;
EqDelay=17;
SNR = 15;
%Training Length
LT=20;
% Data Length
LData=400;
LB=LT+LData;
LD=LB+400;

h1 = (randn(1,LC) + 1i*randn(1,LC))/sqrt(2);
            h2 = (randn(1,LC) + 1i*randn(1,LC))/sqrt(2);
            noiseamp=10^(-SNR/20)*norm([h1 h2])/sqrt(2);
  %Create Channel Matrix
            H1=toeplitz([h1 zeros(1,LE-1)],[h1(1) zeros(1,LE-1)]);
            H2=toeplitz([h2 zeros(1,LE-1)],[h2(1) zeros(1,LE-1)]);
            H=[H1 H2];
            [Hr, ~]=size(H);
            L=zeros(Hr,1);
            L(EqDelay+1)=1;
            
            
QAM16 = 1;
BPSK  = 2;            
SI=round((LData-LT)/2);           
n=LE+LC-1;      
constellation=QAM16;
if constellation == QAM16
    QAM = 16;
    borders = sqrt(QAM) - 1 ;
    imaginaryAxis = (borders:-2:-borders)*1i;
    realAxis = -borders:2:borders;
    [RA,IA] = meshgrid(realAxis,imaginaryAxis);
    constellationPoints = RA + IA;
    lc = length(realAxis);
    numberOfConstellationPoint = QAM;
    cornerIndices = [1 lc (lc-1)*lc+1 lc*lc];
    averagePower = constellationPoints(:)'*constellationPoints(:)/numberOfConstellationPoint;
    constellationPoints = constellationPoints / sqrt(averagePower);
    delta = sum(abs(constellationPoints(:)).^4)/sum(abs(constellationPoints(:)).^2);
    corners = constellationPoints(cornerIndices);
elseif constellation == BPSK
    constellationPoints = [-1 1];
    numberOfConstellationPoint = 2 ;
    averagePower = 1;
    delta = 1;
end

% Center of each subgroup
centers = borders - 1;
imaginaryCenter = (centers:-4:-centers)*1i;
realCenter = -centers:4:centers;
[RC,IC] = meshgrid(realCenter,imaginaryCenter);
constellationCenters = RC + IC;
constellationCenters = constellationCenters / sqrt(averagePower);


cpoints = repmat(constellationPoints(:).', LData ,1);

vecIndices = randi(numberOfConstellationPoint,1,LD);

%x=2*(randn(1,LD)>0)-1;
x=constellationPoints(vecIndices);
if constellation == QAM16
    trainIndices = randi(4,1,LT);
    exactTrainIndices = cornerIndices(trainIndices);
    x(SI+n-1-EqDelay:SI+n-1-EqDelay+LT-1)= corners(trainIndices);
    vecIndices(SI+n-1-EqDelay:SI+n-1-EqDelay+LT-1) = exactTrainIndices;
end
%Data Matrix Generation
vecIndices  = vecIndices(n-EqDelay:n-EqDelay + LData - 1);

S=toeplitz(x,[x(1) zeros(1,(LE+LC-2))]);
S=S(n:n+LData-1,:);


%Convolution
YC=S*H;

%Noise Addition
[m1 n1]=size(YC);

noise_c=noiseamp*(randn(m1,1) + 1i*randn(m1,1))/sqrt(2);
noise_r=noiseamp*(randn(1,LE) + 1i*randn(1,LE))/sqrt(2);
noise_r(1)=noise_c(1);
noise1=toeplitz(noise_c,noise_r);

noise_c=noiseamp*(randn(m1,1) + 1i*randn(m1,1))/sqrt(2);
noise_r=noiseamp*(randn(1,LE) + 1i*randn(1,LE))/sqrt(2);
noise_r(1)=noise_c(1);
noise2=toeplitz(noise_c,noise_r);

noise=[noise1 noise2];



%Input to equalizer
Y=YC+noise;
[r1 c1]=size(Y);
NT=noise(SI:SI+LT-1,:);
ST=S(SI:SI+LT-1,:);
YT=Y(SI:SI+LT-1,:);
st=S(SI-EqDelay:SI-EqDelay+LT-1,1);


% ------------------ ---------------------------
% ---------------------------------------------
% Proposed Algorithm
YD=Y;

Ry=YD'*YD/size(YD,1);
%Ry = Ry + 0.0002*eye(size(Ry));
%cond(Ry)
%pRy=pinv(Ry);
pRy = Ry \ eye(size(Ry));
[U, D]=eig(Ry);
DD=diag(sqrt(1./diag(D)));
Wpre=U*DD*U';
Wpre=eye(size(Wpre));

YD2=YD*Wpre;
YT2=YT*Wpre;

w3=pinv(Wpre)*pinv(YT)*st;
w3=randn(size(w3));
Updatexkm1=w3;
tk=1;
[YTr YTc]=size(YT);
Jmax=0;

YD3=YD*pRy;
w3 = YT2 \ st;
J=[];
for k=1:1200
    zD=YD2*w3;
    zinf=norm(real(zD),'inf');
    zinf2=norm(imag(zD),'inf');
    zinf3=max(zinf,zinf2);
    te=YT2*w3-st;
    ls=norm(te,2);
    J(k)=ls/sqrt(YTr)+lambd*max([zinf zinf2]);
    sz=sign(real(zD)).*(abs(real(zD))>=(alf*zinf3));
    sz2=sign(imag(zD)).*(abs(imag(zD))>=(alf*zinf3));
    
    if (zinf>zinf2)
        Uinf=YD3'*sz/(sum(abs(sz)));
    else
        Uinf=(-j*YD3)'*sz2/(sum(abs(sz2)));
    end
    %Uinf=(YD3'*sz +(-j*YD3)'*sz2)/(sum(abs(sz))+sum(abs(sz2)));
    Uls=YT2'*te/sqrt(YTr);
    U=Uls/norm(Uls)+lambd*Uinf/norm(Uinf);
    
    w3=w3-15/(2*k)*U/norm(U);
    if mod(k,3)==0
    alpha = w3'*YT2'*st/(w3'*(YT2'*YT2)*w3);
    w3= alpha*w3;
    end
    
end

w=Wpre*w3;
g=H*w;

SIR = 10*log10(norm(L-g)^2);
norm_w = norm(w);
norm_z = norm(NT*w);
SINR = 10*log10(max(abs(g))^2/(norm(g)^2-max(abs(g))^2+norm(w)^2*noiseamp^2));
actualerror = 10*log10(norm(L-g)^2+norm(w)^2*noiseamp^2);

sHat = repmat(Y*w/g(EqDelay+1),1,numberOfConstellationPoint);
distances = abs(cpoints - sHat);
[~, y]=min(distances,[],2);
BREProposed = (LData - sum(y == vecIndices')) / LData;

% ---------------------------------------------
% ---------------------------------------------
w1 = pinv(YT) * st;
g1=H*w1;
SIR1 = 10*log10(norm(L-g1)^2);
norm_w1 = norm(w1);
norm_z1 = norm(NT*w1);
SINR1 = 10*log10(max(abs(g1))^2/(norm(g1)^2-max(abs(g1))^2+norm(w1)^2*noiseamp^2));
actualerror1 = 10*log10(norm(L-g1)^2+norm(w1)^2*noiseamp^2);
sHat = repmat(Y*w1/g1(EqDelay+1),1,numberOfConstellationPoint);
distances = abs(cpoints - sHat);
[~, y]=min(distances,[],2);
BRELeast  = (LData - sum(y == vecIndices')) / LData;
% ---------------------------------------------
% ---------------------------------------------
[Hr Hc]=size(H);
weq=(H'*H+noiseamp^2*eye(Hc))^(-1)*H'*L;
geq = H *weq;
SIRmmse = 10*log10(norm(L-geq)^2);
norm_wmmse = norm(weq);
norm_zmmse = norm(NT*weq);
SINRmmse = 10*log10(max(abs(geq))^2/(norm(geq)^2-max(abs(geq))^2+norm(weq)^2*noiseamp^2));
actualerrormmse = 10*log10(norm(L-geq)^2+norm(weq)^2*noiseamp^2);
sHat = repmat(Y*weq/geq(EqDelay+1),1,numberOfConstellationPoint);
distances = abs(cpoints - sHat);
[~, y]=min(distances,[],2);
BREMMSE  = (LData - sum(y == vecIndices')) / LData;
