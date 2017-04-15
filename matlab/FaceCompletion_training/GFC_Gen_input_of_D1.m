function [batch_global, batch, gt_D] = GFC_Gen_input_of_D1(active_G, batch_D, gt_D, mask, parm)

active_G = cell2mat(active_G);

len = size(batch_D, 4);
batch_global = single(zeros(parm.patchsize, parm.patchsize, size(active_G, 3), len));


batch_global(:,:,:,1:len/2) = batch_D(:,:,:,1:len/2);
batch_global(:,:,:,len/2+1:len) = batch_D(:,:,:,1:len/2);
for i = 1:len/2
    pos = mask(:,:,i);
    batch_global(max(1,pos(:,2)):max(1,pos(:,2))+parm.masksize-1, max(1,pos(:,1)):max(1,pos(:,1))+parm.masksize-1, :, i) = ...
        active_G(max(1,pos(:,2)):max(1,pos(:,2))+parm.masksize-1, max(1,pos(:,1)):max(1,pos(:,1))+parm.masksize-1, :, i);
end

batch_D(:,:,:,len/2+1:len) = batch_D(:,:,:,1:len/2);
batch_D(:,:,:,1:len/2) = active_G(:,:,:,1:len/2);

batch = single(zeros(parm.masksize, parm.masksize, size(active_G, 3), len));

for i = 1:len/2
    pos = mask(:,:,i);
    for j = 1:size(mask, 1)
        batch(:,:,:,i) = batch_D(max(1,pos(j,2)):max(1,pos(j,2))+parm.masksize-1, max(1,pos(j,1)):max(1,pos(j,1))+parm.masksize-1, :, i);
        batch(:,:,:,i+len/2) = batch_D(max(1,pos(j,2)):max(1,pos(j,2))+parm.masksize-1, max(1,pos(j,1)):max(1,pos(j,1))+parm.masksize-1, :, i+len/2);
    end
end

% half real with labe1 1
% half face with label 0
gt_D(1, 1, 1, 1:len/2) = single(0);

batch_global = GFC_BatchProcess(batch_global);
batch = GFC_BatchProcess(batch);

end