function demo_face128(use_gpu)

% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system

% Add caffe/matlab to you Matlab search PATH to use matcaffe
    if exist('../+caffe', 'dir')
      addpath('..');
    else
      error('Please run this demo from caffe/matlab/demo');
    end

    % Set caffe mode
    if exist('use_gpu', 'var') && use_gpu
      caffe.set_mode_gpu();
      gpu_id = 0;  % we will use the first gpu in this demo
      caffe.set_device(gpu_id);
    else
      caffe.set_mode_cpu();
    end

    model_dir = './model/';
    
    %% Load the model
    net_model = [model_dir 'Model_G.prototxt'];    
    net_weights = [model_dir 'Model_G.caffemodel'];

    phase = 'test'; % run with phase test

    if ~exist(net_weights, 'file')
        error('Please download CaffeNet from Model Zoo before you run this demo');
    end

    net = caffe.Net(net_model, net_weights, phase);

    %% Run
    im = imread('./TestImages/182701.png');
    im = im2single(im);
    
    if rand > 0.5
        im = fliplr(im);
    end

    target = im;

    mx = 28; % the position of the top-left
    my = 25; 

    masksize_x = 56; % the height and width of the mask
    masksize_y = 77; 
    
    mask = zeros(size(im));
    mask(mx:mx+masksize_x-1,my:my+masksize_y-1,:) = 1;

    input_data= prepare_image(im, mx, my, masksize_x, masksize_y);
        
    figure();
    subplot(221);imshow(target);title('Original Face');

    scores = net.forward({input_data});
    scores = scores{1};

    % range goes back to [0,1]
    output_data = (scores + 1)./2;
    input_data = (input_data + 1)./2;  
    subplot(222);imshow(input_data);title('Masked Input');
    
    % put the generated content (mask region) in the original image
    im(mx:mx+masksize_x-1,my:my+masksize_y-1,:) = output_data(mx:mx+masksize_x-1,my:my+masksize_y-1,:);
    subplot(223);imshow(im);title('Our Completion');

    pb = PBlending(im, target, mask);
    subplot(224);imshow(pb);title('PB Refinement');

caffe.reset_all();


function im_input = prepare_image(im, mx, my, masksize_x, masksize_y)

% range goes to [-1, 1], according to the training settings
im_input = -1 + 2 * im;
mask = single(-1+2*rand(masksize_x,masksize_y,3));

im_input(mx:mx+masksize_x-1,my:my+masksize_y-1,:) = mask;



function Y = PBlending(source, target, mask)

for i=1:size(mask,3)
    tmask = mask(:,:,i);
    mask(:,:,i) = bwmorph(tmask,'thin');
end

[Lh, Lv] = imgrad(target);
[Gh, Gv] = imgrad(source);

X = target;

Fh = Lh;
Fv = Lv;

for i=1:size(mask,1)
    for j=1:size(mask,2)
        if(mask(i,j,1)==1)
            X(i,j,:) = source(i,j,:);
            Fh(i,j,:) = Gh(i,j,:);
            Fv(i,j,:) = Gv(i,j,:);
        end
    end
end

Y = PoissonJacobi(X, Fh, Fv, mask);
