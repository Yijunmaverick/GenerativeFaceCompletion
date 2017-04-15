function [Solver_, Solver] = GFC_caffeinit(solver_file, caffemodel)
%
if ~exist('solver_file','var')
    error('You need a network prototxt definition');
end
Solver_ = caffe.Solver(solver_file); % c++
Solver = GFC_SolverParser(solver_file); % mat
if exist('caffemodel','var') 
    Solver_.net.copy_from(caffemodel);
end

% Your training data path
Solver.folder_img = '/home/yijun/dcgan-torch-master/data/celebA/train';

% Set caffe mode
if strcmp(Solver.solver_mode,'GPU')
    caffe.set_mode_gpu();
    caffe.set_device(Solver.device_id);
else
    caffe.set_mode_cpu();
end
fprintf('Done with init\n');

% put into train mode
fprintf(sprintf('Solving net: %s\n',solver_file));
fprintf(sprintf('Learning rate: %d\n',Solver.base_lr));
end
