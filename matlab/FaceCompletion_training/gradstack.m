function delta_G = gradstack(delta_G, f, f3, f_parsing, mask, parm)

lamda_local_D = 300;
lamda_global_D = 300;
lamda_parsing = 0.05;

f = cell2mat(f);
f3 = cell2mat(f3);
f_parsing = cell2mat(f_parsing);
delta_G = cell2mat(delta_G);

for i = 1:size(mask,3)
    pos = mask(:,:,i);
    delta_G(max(1,pos(:,2)):max(1,pos(:,2))+parm.masksize-1, max(1,pos(:,1)):max(1,pos(:,1))+parm.masksize-1, :, i) = lamda_local_D.* f(:,:,:,i) + ...
        delta_G(max(1,pos(:,2)):max(1,pos(:,2))+parm.masksize-1, max(1,pos(:,1)):max(1,pos(:,1))+parm.masksize-1, :, i)+...
     lamda_parsing.*f_parsing(max(1,pos(:,2)):max(1,pos(:,2))+parm.masksize-1, max(1,pos(:,1)):max(1,pos(:,1))+parm.masksize-1, :, i);
end

delta_G = delta_G + lamda_global_D.*f3;

delta_G = {single(delta_G)};
end