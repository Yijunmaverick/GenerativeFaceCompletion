function Solver = demo_GFC_training()
% version of +caffe
mex_path = '../+caffe';
addpath('..');
model_path = 'model';


% updated prototxt and model file
solver_file_G = fullfile(model_path,sprintf('Model_G_solver.prototxt'));
save_file_G = fullfile(model_path, sprintf('Face128_Model_G_recons.caffemodel'));

solver_file_D = fullfile(model_path,sprintf('Model_localD_solver.prototxt'));
save_file_D = fullfile(model_path, sprintf('Model_localD.caffemodel'));

solver_file_D3 = fullfile(model_path,sprintf('Model_globalD_solver.prototxt'));
save_file_D3 = fullfile(model_path, sprintf('Model_globalD.caffemodel'));


solver_file_parsing = fullfile(model_path,sprintf('Model_parsing_solver.prototxt'));
save_file_parsing = fullfile(model_path, sprintf('Model_parsing.caffemodel'));

[Solver_G_, Solver_G] = GFC_caffeinit(solver_file_G, save_file_G); 


[Solver_D_, Solver_D] = GFC_caffeinit(solver_file_D);  
% Continue the training from a existing D model
%[Solver_D_, Solver_D] = GFC_caffeinit(solver_file_D, solver_file_D );  

[Solver_D3_, Solver_D3] = GFC_caffeinit(solver_file_D3);

% The parsing network is fixed by setting base_lr = 0 in solver
[Solver_parsing_, Solver_parsing] = GFC_caffeinit(solver_file_parsing, save_file_parsing);

begin = Solver_G_.iter()+1;

parm_G = GFC_Init(Solver_G);

for iter = begin: Solver_G.max_iter
    Solver_G.iter = iter;
    Solver_G_.set_iter(double(iter));
   
    %% (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    for iter_D = 1:1
        [batch_G, batch_D, gt_D, mask] = GFC_Gen_training_D(parm_G);

        batch_G = GFC_BatchProcess(batch_G);
        active_G = Solver_G_.net.forward(batch_G);

        [batch_global, batch_D, gt_D] = GFC_Gen_input_of_D1(active_G, batch_D, gt_D, mask, parm_G);

        Update_D(Solver_D_, Solver_D, batch_D, gt_D, iter_D);  % local D
        Update_D(Solver_D3_, Solver_D3, batch_global, gt_D, iter_D); % global D
    end
    
    %% (2) Update G network: maximize log(D(G(z)))
    [batch, gt_G, gt_D, mask] = GFC_Gen_training_G(parm_G);
    batch = GFC_BatchProcess(batch);
    active_G = Solver_G_.net.forward(batch);
    
    % visualize one generated image
    ind = randperm(parm_G.batchsize); ind = ind(1,1);
    sample_GT = gt_G(:,:,:,ind); sample_GT = (sample_GT+1)/2;
    sample_output = active_G; sample_output = sample_output{1}; sample_output = sample_output(:,:,:,ind);
    sample_input = batch{1}; sample_input = sample_input(:,:,1:3,ind);
    sample_output = (sample_output+1)/2;
    sample_input = (sample_input+1)/2;
    subplot(231);imshow(sample_GT);title('GT');
    subplot(232);imshow(sample_input);title('Masked');
    subplot(233);imshow(sample_output);title('Output');
    
    pos = mask(:,:,ind);
    sample_input(max(1,pos(:,2)):max(1,pos(:,2))+parm_G.masksize-1, max(1,pos(:,1)):max(1,pos(:,1))+parm_G.masksize-1, :) = ...
        sample_output(max(1,pos(:,2)):max(1,pos(:,2))+parm_G.masksize-1, max(1,pos(:,1)):max(1,pos(:,1))+parm_G.masksize-1, :);
    subplot(234);imshow(sample_input);title('Replaced');
    
    % for local D
    batch_D = GFC_Gen_input_of_D2(active_G, mask, parm_G);
    delta_D = get_delta_D(Solver_D_, Solver_D, batch_D, gt_D, iter);
    
    % for global D
    G_output = active_G(1,1); 
    G_output = G_output{1};
    batch_ori = gt_G(:,:,:,:);
    for i = 1:size(mask,3)
        pos = mask(:,:,i);
        batch_ori(max(1,pos(:,2)):max(1,pos(:,2))+parm_G.masksize-1, max(1,pos(:,1)):max(1,pos(:,1))+parm_G.masksize-1, :, i) = ...
        G_output(max(1,pos(:,2)):max(1,pos(:,2))+parm_G.masksize-1, max(1,pos(:,1)):max(1,pos(:,1))+parm_G.masksize-1, :, i);
    end
    batch_ori = GFC_BatchProcess(batch_ori);
    delta_D3 = get_delta_D(Solver_D3_, Solver_D3, batch_ori, gt_D, iter);
    
    
    % reconstruction loss
    delta_G = cell(size(active_G));
    for c = 1:length(active_G)
        active_G_ = active_G{c};
        delta_G{c} = zeros(size(active_G{c}));
        [delta_G_, loss_G] = GFC_L1_loss(active_G_, gt_G, 'train');
        Solver_G.loss(iter) = loss_G;
        fprintf('loss_L1 = %i', loss_G); fprintf('  ');
        delta_G{c} = delta_G_;
    end  
    
    
    % Parsing regularization
    active_parsing_gt = Solver_parsing_.net.forward(GFC_BatchProcess(gt_G)); 
    
    active_parsing_gt = active_parsing_gt{1};
    
    parsing_gt = zeros(size(active_parsing_gt));
    for i=1:size(active_parsing_gt,4)
       sample_output = active_parsing_gt(:,:,:,i);
       [~, pos] = max(sample_output,[],3);
       parsing_gt(:,:,i) = pos;
    end
    parsing_gt = GFC_BatchProcess(parsing_gt);
    
    active_parsing = Solver_parsing_.net.forward(batch_ori);
    delta_parsing = cell(size(active_parsing));
    for c = 1:length(active_parsing)
        active_parsing_ = active_parsing{c};
        parsing_gt_ = parsing_gt{c};  
        delta_parsing{c} = zeros(size(active_parsing{c}));
    
        [delta_parsing_, loss_parsing] = GFC_Softmax_loss(active_parsing_, parsing_gt_, 'train');
        Solver_parsing.loss(iter) = loss_parsing;
   
        fprintf('parsing error = %i', Solver_parsing.loss(iter)); fprintf('  ');     
        delta_parsing{c} = delta_parsing_;
    end  
    
    fpts = active_parsing_gt;
    fpts = fpts(:,:,:,ind);
    [~, pos] = max(fpts,[],3);
    subplot(235);imshow(uint8(visualization(pos)));title('GTParsing');
    
    
    fpts = active_parsing{1};
    fpts = fpts(:,:,:,ind);
    [~, pos] = max(fpts,[],3);
    subplot(236);imshow(uint8(visualization(pos)));title('OursParsing');
    
    pause(0.1);
    fprintf('\n');
    
    
    % stack the gradient from difference losses
    if ~isnan(Solver_G.loss(iter))
        f = Solver_D_.net.backward(delta_D);
        f3 = Solver_D3_.net.backward(delta_D3);
        fparsing = Solver_parsing_.net.backward(delta_parsing);
        delta_G = gradstack(delta_G, f, f3, fparsing, mask, parm_G);        
        Solver_G_.net.backward(delta_G);   
        Solver_G_.update();
    else
        error('NAN');
    end
       
    if mod(iter, 100) == 0
        Solver_D_.net.save(save_file_D);
        Solver_D3_.net.save(save_file_D3);
        Solver_G_.net.save(save_file_G);
    end
end

end
