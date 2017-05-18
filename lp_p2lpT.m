function [ theta_lp, rho ] = p2lpT( theta, s, beta, M, a_R )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

m = round(mod(theta/beta,M));

theta_lp = theta - m*beta;
rho = log(a_R*s + (1 - a_R)*cos(theta_lp));
% 
% for i = 1:length(theta_lp)
%     
%     rho(i,:) = log(a_R*s + (1 - a_R)*cos(theta_lp(i)));
%     
% end
% rho = rho';
end

