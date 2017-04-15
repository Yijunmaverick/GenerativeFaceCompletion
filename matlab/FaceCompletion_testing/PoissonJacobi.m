% PoissonJacobi reconstructs an image based on the Poisson equation
%
% dst = PoissonJacobi(src, Fh, Fv, msk, itr, th, verbose)
%
%
%Output parameter:
% dst: reconstructed image
%
%
%Input parameters:
% src: initial image
% Fh: forward horizontal difference map
% Fv: forward vertical difference map
% msk: pixels whose value is greater than zero are processed
% itr (optional): maximum iteration (default: 1024)
% th (optional): stopping criteria (default: 1E-3)
% verbose (optional): if true, display at each iteration (default: false)
%
%
%Example:
% X = double(imread('img.jpg'));
% msk = ones(size(img));
% msk(1,:,:) = 0;
% msk(:,1,:) = 0;
% msk(size(img,1),:,:) = 0;
% msk(:,size(img,2),:) = 0;
% [Fh Fv] = imgrad(X);
% Fh = Fh * 2;
% Y = PoissonJacobi(X, Fh, Fv, msk);
%
%
%Version: 20120605
% Poisson Image Reconstruction by Jacobi Algorithm         
                                                          
% Copyright (C) 2012 Masayuki Tanaka. All rights reserved. 
%                    mtanaka@ctrl.titech.ac.jp             
                                                         
function dst = PoissonJacobi(src, Fh, Fv, msk, itr, th, verbose)

if( nargin < 5 )
 itr = 200;
end

if( nargin < 6 )
 th = 1E-3;
end

if( nargin < 7 )
 verbose = false;
end

K=[0,1,0;1,0,1;0,1,0];
p = ( msk > 0 );
lap = grad2lap(Fh,Fv);

df0 = 1E32;
dst = src;
dst0 = dst;
for i = 1:itr
 lpf = imfilter(dst,K,'replicate');
 dst(p) = (lap(p) + lpf(p))/4;
 
 dif = abs(dst-dst0);
 df = max(dif(:));
 
 if( verbose )
  fprintf('%d %g %g\n',i, df, (df0 - df)/df0);
 end
 
 if( (df0 - df)/df0 < th )
  break;
 end
 dst0 = dst;
 df0 = df;
end

function lap = grad2lap(Fh, Fv)
lap = circshift(Fh,[0,1]) + circshift(Fv,[1,0]) - Fh - Fv;
