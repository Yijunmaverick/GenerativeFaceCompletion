function [delta, loss] = GFC_BCE_loss2(active, gt, mode)

eps = 1e-12;

bz = size(active,4);
output = active;

% bz = size(active,4);
% output = zeros(bz, 1);
% for i = 1:bz
%     output(i) = active(:,:,:,i);
% end

loss = -log(output + eps) .* gt;

dt_loss = - gt ./ (output + eps);

loss = sum(loss(:)) ./ bz;

% dt_loss = zeros(size(active,1), size(active,2), size(active,3), size(active,4));
% for i = 1:bz
%     dt_loss(:,:,:,i) = dt(i,:,:,:);
% end

if strcmp(mode, 'train')
%     msk1 = hardsample(dt, [1:cha], [r,c,cha,bz], 0.4);
%     msk2 = uni_balance([r,c,cha,bz], 0.1);
%     msk = max(cat(5,msk1,msk2),[],5);
%     delta = single((msk.*dt)/bz);
    delta = single(dt_loss/bz);
else
    delta = 0;
end

end
