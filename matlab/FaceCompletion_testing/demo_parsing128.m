function demo_parsing128(use_gpu)

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
    net_model = [model_dir 'Model_parsing.prototxt'];
    net_weights = [model_dir 'Model_parsing.caffemodel'];

    phase = 'test'; % run with phase test
    if ~exist(net_weights, 'file')
      error('Please download the parsing model before you run this demo');
    end
    
    net = caffe.Net(net_model, net_weights, phase);
    
    %% Run 
    im = imread('./TestImages/182701.png');
    im = im2single(im);
    
    % range goes to [-1, 1], according to the training settings
    im_input = -1 + 2 * im;
    
    scores = net.forward({im_input});
    scores = scores{1};
    [~, pos] = max(scores,[],3);

    figure();
    subplot(121);imshow(im);title('Input Face');
    subplot(122);imshow(uint8(visulization(pos)));title('Parsing Result');

caffe.reset_all();
