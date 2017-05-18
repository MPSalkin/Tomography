clear all, close all, clc

% Image
A = phantom(80);
SINO = radon(A)';
% imagesc(SINO)
[Nt, N] = size(SINO);
Nt/N

%% Geometry Initializaiton
M = 3;
beta = pi/M;
a_R = sin(beta/2)/(1 + sin(beta/2));
a_r = (cos(beta/2) - sin(beta/2)) / (1 + sin(beta/2));

%Cartesian Grid %%%%%%%%%%%%%%%%%
j1 = -N/2:N/2-1;
j2 = N/2-1:-1:-N/2;
x = j1/N;
y = j2/N;
[X,Y] = meshgrid(x,y);

%Log Polar Grid %%%%%%%%%%%%%%%%%
dp = -log(1 - 2*a_R/N);
dtl = 2*a_R/N;
Np = ceil(log(a_r)/log(1-2*a_R/N));
% Nt = 3*N/2;
jtl = ceil(-beta/(2*dtl)):floor(beta/(2*dtl))-1;
jp = -Np+1:0;
theta_lp = jtl*dtl';
rho_lp = jp*dp;
% 
% THETA_LP = repmat(theta_lp,[1,length(rho_lp)]);
% RHO_LP = repmat(rho_lp,[length(theta_lp),1]);
[RHO_LP,THETA_LP] = meshgrid(rho_lp, theta_lp);
[Xlp, Ylp] = XcoorLP( THETA_LP, RHO_LP);

%Log Polar Down Sample Grid %%%%%%%%%%%%%%%%%
dtp = 2/N;
jtp = ceil(-Nt/(2*M)):floor(Nt/(2*M))-1;
theta_lpp = [jtp*dtp]';
% 
% THETA_LPP = repmat(theta_lpp,[1,length(rho_lp)]);
% RHO_LPP = repmat(rho_lp,[length(theta_lpp),1]);
[RHO_LPP,THETA_LPP] = meshgrid(rho_lp, theta_lpp);
[Xlpp, Ylpp] = XcoorLP( THETA_LPP, RHO_LPP);

%Polar Grid %%%%%%%%%%%%%%%%%
dt = 2/N;
ds = 1/N;
jt =  0:Nt-1; %-Nt/(2*M):((Nt-1)-Nt/(2*M));%
js = j1; %0:N-1; %
theta = jt*dt';
s = js*ds;

% THETA = repmat(theta,[1,length(s)]);
% S = repmat(s,[length(theta),1]);

[S, THETA] = meshgrid(s, theta);
XP = S.*cos(THETA);
YP = S.*sin(THETA);
[THETA_p2lp, RHO_p2lp] = p2lpT(THETA,S,beta, M, a_R);
% THETA_p2lp = repmat(theta_p2lp,[length(s),1]);

%%%%%%%%%%%%%%%%% Zeta %%%%%%%%%%%%%%%%%
kt = jtp;
kp = 0:Np-1;  %jp;%
theta0=-beta:1/Nt:beta;
zetafun = @(theta,kt,kp) exp(-pi*1i*theta*kt/beta)...
            .*cos(theta).^(-2*pi*1i*kp/(log(a_r)))/...
                (-2*beta*log(a_r));
zeta = zeros(length(kt),length(kp));
for i = 1:length(kt)
    for j = 1:length(kp)
        zeta(i,j) = trapz(theta0,zetafun(theta0,kt(i),kp(j)));
    end
end
% imagesc(real((zeta)))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN LOOP FOR FORWARD RADON   %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kp = jp;
FFTG=0;
FFTg=[];
Rf = [];
Rff = 0;
Rfp = [];
Rffp = 0;
iTheta = [];
iS = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for m = 0:M-1
    
%%%%%%%%%%%%%%%%% Polar Grid %%%%%%%%%%%%%%%%%
% G = griddata(THETA_p2lp, RHO_p2lp, SINO, THETA_LPP, RHO_LPP,'cubic');
[ tXP, tYP ] = CoordT( XP,YP,a_R,m,beta );
[ iXlpp, iYlpp ] = invCoordT( Xlpp, Ylpp, a_R, m, beta );
ss = (sqrt(iXlpp.^2 + iYlpp.^2));
the = atan(iYlpp./iXlpp);
G = interp2(S, THETA, SINO, ss, the,'cubic',0);
% G = griddata(tXP, tYP, SINO, Xlpp, Ylpp,'cubic');
figure, surf(Xlpp,Ylpp,G,'EdgeColor','none');view(2)

% imagesc(G)

%%%%%%%%%%%%%%%%% Log Polar Grid %%%%%%%%%%%%%%%%%
% [ iXlp, iYlp ] = invCoordT( Xlp,Ylp,a_R,m,beta );

% Alp = interp2(X,Y,A,iXlp,iYlp,'cubic',0 );

% figure, surf(Xlp,Ylp,Alp,'EdgeColor','none');view(2)
% figure
% imagesc(Alp')

%%%%%%%%%%%%%%%%% Log Polar Down Sample Grid %%%%%%%%%%%%%%%%%
% Alpp = interp2(RHO_LP,THETA_LP, Alp, RHO_LPP, THETA_LPP, 'cubic',0);
 
% figure, surf(Xlpp,Ylpp,Alpp,'EdgeColor','none');view(2)
% figure
% imagesc(Alp')

%%%%%%%%%%%%%%%%% FFT Image %%%%%%%%%%%%%%%%%
fftG = zeros(size(G));
fftG1 = fft(G,[],2);
for j = 1:length(kp)
    fftG(:,j) = fft(fftG1(:,j),[],1).* exp(1i*pi*kt');
end
fftG = fftG*(M/(Nt*Np));
% figure, imagesc(real(fftshift(fftA)))
% figure, imagesc(real((fftA.*zeta')))

FFTG = FFTG + fftG;
FFTg = [FFTg; fftG];

%%%%%%%%%%%%%%%%% IFFT Image %%%%%%%%%%%%%%%%%
GZ = fftG;%.*zeta;
iFFTGZ1 = zeros(size(GZ));
iFFTG = iFFTGZ1;
for i = 1:length(kt) % step through the row vectors - rho dimensions
    for j = 1:length(jp) % step through the rho frequency dim
        iFFTGZ1(i,j) = sum(GZ(i,:).*exp(2*pi*1i*rho_lp(j).*kp/-log(a_r)));
    end
end
for j = 1:length(kp) % step through the col vectors - theta dimensions
    for i = 1:length(jtp) % step through the theta frequency dim
        iFFTG(i,j) = sum(iFFTGZ1(:,j).*exp(2*pi*1i*theta_lpp(i).*kt'/(2*beta)));
    end
end
R = -2*beta*log(a_r)*iFFTG;

% ss = (sqrt(iXlpp.^2 + iYlpp.^2));
% the = atan(iYlpp./iXlpp);

% [ thetaa, ss ] = invp2lpT( THETA_LPP, RHO_LPP, beta, m, a_R );

Rp = griddata(iXlpp,iYlpp,R,X,Y,'cubic');
% Rp = griddata(ss,thetaa,R',THETA',S','cubic');

% [ itheta, is ] = invp2lpT( THETA_LPP, RHO_LPP, beta, m, a_R );

% Rp = interp2(RHO_LPP, THETA_LPP, R, RHO_p2lp, THETA_p2lp, 'cubic');
% % figure, imagesc(real((Rp)))
% 

Rfp = [Rfp;Rp];
Rffp = Rffp + Rp;

% iTheta = [iTheta;itheta];
% iS = [iS;is];

Rf = [Rf;R];
Rff = Rff + R;
end

% Rp = griddata(iS, iTheta, Rf, S, THETA, 'cubic');


% figure,imagesc(real(fftshift(FFTA)))
figure, imagesc(real((Rp)))
figure, imagesc(real((Rff)))

figure, imagesc(real((Rfp')))
figure, imagesc(real((Rffp')))








