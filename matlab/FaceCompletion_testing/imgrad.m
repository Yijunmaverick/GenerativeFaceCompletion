% imgrad calculates horizontal and vertical gradients.
%
% [Fh Fv Bh Bv] = imgrad(X)
%
%Output parameters:
% Fh: forward horizontal diference
% Fv: forward vertical diference
% Bh: forward horizontal diference
% Bv: forward vertical diference
%
%
%Input parameters:
% X: input image
%
%
%Example:
% X = imread('img.png');
% [Fh Fv] = imgrad(X);
%
%
%Version: 20120604

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Miscellaneous tools for image processing                 %
%                                                          %
% Copyright (C) 2012 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Fh Fv Bh Bv] = imgrad(X)

Kh = [ 0,-1, 1 ];
Kv = [ 0;-1; 1 ];

Fh = imfilter(X,Kh,'replicate');
Fv = imfilter(X,Kv,'replicate');

if( nargout >= 3 )
 Bh = circshift(Fh,[0,1]);
 Bv = circshift(Fv,[1,0]);
end
