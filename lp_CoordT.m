function [ T1, T2 ] = CoordT( x,y,a_R,m,beta )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
T1 = a_R*(x*cos(m*beta) + y*sin(m*beta)) + (1 - a_R);
T2 = a_R*(-x*sin(m*beta) + y*cos(m*beta));
end
