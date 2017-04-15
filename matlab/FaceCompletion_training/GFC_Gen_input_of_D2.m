function batch = GFC_Gen_input_of_D2(active_G, mask, parm)

active_G = cell2mat(active_G);

batch = single(zeros(parm.masksize, parm.masksize, size(active_G, 3), size(active_G, 4)));

for i = 1:size(mask,3)
    pos = mask(:,:,i);
    for j = 1:size(mask,1)
        batch(:,:,:,i) = active_G(max(1,pos(j,2)):max(1,pos(j,2))+parm.masksize-1, max(1,pos(j,1)):max(1,pos(j,1))+parm.masksize-1, :, i);
    end
end

batch = GFC_BatchProcess(batch);

end