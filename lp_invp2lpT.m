function [ theta, s ] = invp2lpT( theta_lp, rho, beta, m, a_R )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

theta = theta_lp + m*beta;
s = (exp(rho) - (1 - a_R)*cos(theta))/a_R;

% for i = 1:length(theta)
%     s(i,:) = (exp(rho) - (1 - a_R)*cos(theta(i)))/a_R;
% end
% theta = repmat(theta,[length(s),1]);
% theta=theta';
end

