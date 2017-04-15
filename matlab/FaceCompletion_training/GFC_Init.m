function parm = GFC_Init(Solver)

parm.patchsize = 128;
parm.batchsize = 2;
parm.masksize = 64;
parm.interval = 1;

% training
parm.train_folder = Solver.folder_img;

tdir = dir(fullfile(parm.train_folder, '*.jpg'));

if isempty(tdir)
tdir = dir(fullfile(parm.train_folder, '*.png'));
end

parm.train_num = length(tdir);
fprintf('training number: %d.\n',parm.train_num);
parm.trainlst = cell(1,parm.train_num);
for m = 1:parm.train_num
   parm.trainlst{1,m}  = tdir(m).name;
end

end
