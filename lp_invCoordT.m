function [ ix, iy ] = invCoordT( x,y,a_R,m,beta )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
ix = ((x - 1 + a_R)*cos(m*beta) - y*sin(m*beta))/a_R;
iy = ((x - 1 + a_R)*sin(m*beta) + y*cos(m*beta))/a_R;
end

