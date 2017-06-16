function [delta, error] = GFC_Softmax_loss(active, GT, mode)

bz = size(active,4);
cha = size(active,3);
dt_loss = zeros(size(active));
loss = 0;

for i = 1:bz
   output = active(:,:,:,i);
   t=max(output,[],3);
   for s=1:cha
      output(:,:,s) = output(:,:,s)-t;
   end
   
   output = exp(output);
   p=zeros(size(output));
   q=sum(output,3);
   for s=1:cha    
      p(:,:,s) = output(:,:,s)./q;
   end
   
   y=zeros(size(output));
   for j=1:cha     
      gt = GT(:,:,i);
      y(:,:,j) = gt.*(gt==j)./j;
   end
   dt_loss(:,:,:,i) = p - y;
   
   gt = GT(:,:,i);
   for v=1:size(active,1)
       for t=1:size(active,2)
           loss = loss - log(p(v,t,gt(v,t)));
       end
   end
end

error = loss./bz;

if strcmp(mode, 'train')
    delta = single(dt_loss/bz);
else
    delta = 0;
end

end
