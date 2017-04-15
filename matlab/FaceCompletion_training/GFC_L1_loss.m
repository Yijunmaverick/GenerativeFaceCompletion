function [delta, loss] = GFC_L1_loss(active, gt, mode)

eps = 1e-12;
[r,c,cha,bz] = size(active);
if size(gt,1)~= r
    gt = imresize(gt,[r,c]);
end

dt = active - gt;
loss = (sum(sqrt(dt(:).^2+ eps.^2)))/bz;
dt = dt./sqrt(dt.^2 + eps.^2);
if strcmp(mode, 'train')
    delta = single(dt/bz);
else
    delta = 0;
end

end