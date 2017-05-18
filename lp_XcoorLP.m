function [ X, Y ] = XcoorLP( THETA, RHO )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
X = exp(RHO).*cos(THETA);
Y = exp(RHO).*sin(THETA);
end

