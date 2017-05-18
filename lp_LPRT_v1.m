clear all, close all, clc

% Image
A = phantom(250);
[~, N] = size(A);

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
Nt = 3*N/2;
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
js = j1;
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
kp = 0:Np-1; %jp;  % 
theta0=-beta:1/(8*Nt):beta;
zetafun = @(theta,kt,kp) exp(-pi*1i*theta*kt/beta)...
            .*cos(theta).^(-2*pi*1i*kp/(-log(a_r))-1)/...
                (-2*beta*log(a_r));
zeta = zeros(length(kt),length(kp));
for i = 1:length(kt)
    for j = 1:length(kp)
        zeta(i,j) = trapz(theta0,zetafun(theta0,kt(i),kp(j)));
    end
end
zeta = zeta/Nt;
figure,imagesc(fftshift(fftshift(real((zeta))),2))
% 
% run LPRT_init.m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN LOOP FOR FORWARD RADON   %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kp = jp; 
jp = kp; 

FFTA=0;
FFTa=[];
Rf = [];
Rff = 0;
Rfp = [];
Rffp = 0;
iTheta = [];
iS = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for m = 0:M-1
%     %%%%%%%%%%%%%%%%% Cartesian Grid Transform %%%%%%%%%%%%%%%%%
%     [ Xm, Ym ] = CoordT( x,y, a_R, m, beta );
%     [Xm, Ym] = meshgrid(Xm, Ym );
%     Alp = interp2(Xm,Ym,A,Xlp,Ylp,'cubic',0);

    % Am = interp2(Xo,Yo,A,Xm,Ym,'cubic',0);
    % 
%     surf(Xlp,Ylp,Alp,'EdgeColor','none');view(2)
    % figure
    % imagesc(Am)

    %%%%%%%%%%%%%%%%% Log Polar Grid Transform %%%%%%%%%%%%%%%%%
    [ iXlp, iYlp ] = invCoordT( Xlp,Ylp,a_R,m,beta );

    Alp = interp2(X,Y,A,iXlp,iYlp,'cubic',0 );

    % figure, surf(Xlp,Ylp,Alp,'EdgeColor','none');view(2)
    % figure
    % imagesc(Alp')

    %%%%%%%%%%%%%%%%% Log Polar Down Sample Grid Transform %%%%%%%%%%%%%%%%%
    Alpp = interp2(RHO_LP,THETA_LP, Alp, RHO_LPP, THETA_LPP, 'cubic',0);

    % figure, surf(Xlpp,Ylpp,Alpp,'EdgeColor','none');view(2)
    % figure
    % imagesc(Alp')

    %%%%%%%%%%%%%%%%% FFT Image Built in %%%%%%%%%%%%%%%%%
    Ae = Alpp.*exp(RHO_LPP);

%     fftA = zeros(size(Ae));
%     fftA1 = (fft(Ae,[],2));
%     for j = 1:length(kp)
%         fftA(:,j) = fft(fftA1(:,j),[],1).* exp(1i*pi*kt');
%     end
%     fftA = (fftA/(2/(Nt*Np))); % (fftshift(fft2(Ae))); %
%     figure, imagesc(real(fftshift(fftA)))
%     % figure, imagesc(real((fftA.*zeta)))
% 
%     FFTA = FFTA + fftA;
%     FFTa = [FFTa; fftA];
    %%%%%%%%%%%%%%%%% FFT Image Explicit %%%%%%%%%%%%%%%%%
    fftA = zeros(size(Ae));
    fftA1 = fftA;
    for i = 1:length(kt) % step through the row vectors - rho dimensions
        for j = 1:length(jp) % step through the rho frequency dim
            fftA1(i,j) = sum(Ae(i,:).*exp(-2*pi*1i*jp(j).*kp/Np));
        end
    end
    for j = 1:length(kp) % step through the col vectors - theta dimensions
        for i = 1:length(jtp) % step through the theta frequency dim
            fftA(i,j) = sum(fftA1(:,j).*exp(-2*pi*1i*jtp(i).*kt'/(Nt/M)));
        end
    end
    fftA = fftA*M/(Nt*Np);
%     figure, imagesc(real((fftA)))

    FFTA = FFTA + fftA;
    FFTa = [FFTa; fftA];
    %%%%%%%%%%%%%%%%% IFFT Image %%%%%%%%%%%%%%%%%
    FZ = (fftA).*zeta;
    iFFTAZ1 = zeros(size(FZ));
    iFFTA = iFFTAZ1;
    for i = 1:length(kt) % step through the row vectors - rho dimensions
        for j = 1:length(jp) % step through the rho frequency dim
            iFFTAZ1(i,j) = sum(FZ(i,:).*exp(2*pi*1i*rho_lp(j).*kp/-log(a_r)));
        end
    end
    for j = 1:length(kp) % step through the col vectors - theta dimensions
        for i = 1:length(jtp) % step through the theta frequency dim
            iFFTA(i,j) = sum(iFFTAZ1(:,j).*exp(2*pi*1i*theta_lpp(i).*kt'/(2*beta)));
        end
    end
    R = ((-2*beta*log(a_r)*iFFTA)); %(fftshift(ifft2(FZ),2));%


    [ iXlpp, iYlpp ] = invCoordT( Xlpp,Ylpp,a_R,m,beta );


    %INTERP  
    % ss = (sqrt(iXlpp.^2 + iYlpp.^2));
    % the = atan(iYlpp./iXlpp);
    % [ thetaa, ss ] = invp2lpT( THETA_LPP, RHO_LPP, beta, m, a_R );
    % [ itheta, is ] = invp2lpT( THETA_LPP, RHO_LPP, beta, m, a_R );
    % iTheta = [iTheta;itheta];
    % iS = [iS;is];

    Rp = griddata(iXlpp,iYlpp,R,XP,YP,'cubic');
    % Rp = griddata(ss,thetaa,R,THETA,S,'cubic');


    % Rp = interp2(RHO_LPP, THETA_LPP, R, RHO_p2lp, THETA_p2lp, 'cubic');
    % % figure, imagesc(real((Rp)))
    % 

    Rfp = [Rfp;Rp];
    Rffp = Rffp + Rp;


    Rf = [Rf;R];
    Rff = Rff + R;
end

% % Rp = griddata(iS, iTheta, Rf, S, THETA, 'cubic');
%     Rpp = griddata(X,Y,Rp,XP,YP,'cubic');
%     figure, imagesc(real((Rpp)))


% figure,imagesc(real(fftshift(FFTA)))
figure, imagesc(real((Rp')))
figure, imagesc(real((Rfp')))

figure, imagesc(real((Rf)))
figure, imagesc(real((Rff)))

% figure, imagesc(real((Rfp')))
% figure, imagesc(real((Rffp')))

