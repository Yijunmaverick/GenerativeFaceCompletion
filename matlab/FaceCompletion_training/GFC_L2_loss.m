function [delta, loss] = GFC_L2_loss(active, gt, mode)

[r,c,cha,bz] = size(active);
if size(gt,1)~= r
    gt = imresize(gt,[r,c]);
end
dt = active - gt;
loss = 0.5 * sum(dt(:).^2)/bz;
if strcmp(mode, 'train')
    delta = single(dt/bz);
else
    delta = 0;
end
end
