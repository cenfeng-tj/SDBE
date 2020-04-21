close all;
clear all;

% Setup MatConvNet.
% Suppose vl_setupnn.m locate in the folder './matconvnet/matlab'.
vl_setup_file = fullfile('matconvnet','matlab','vl_setupnn.m'); 
run(vl_setup_file);

USEGPU = true;   % false: use CPU only 
                 % true: use GPU to accelerate computing
model_type = 'ResNet';
model_file = fullfile('cnn_models','imagenet-resnet-152-dag.mat');
data_folder = fullfile('..','datasets','ilsvrc2012');

lambda = 0.005;

if USEGPU 
    % Reduce the time to initialize GPU
    % Refer to https://www.mathworks.com/matlabcentral/answers/329505-gpudevice-takes-long-time-to-initialize-with-a-pascal-architecture-gpu-card
    % setenv('CUDA_CACHE_MAXSIZE','1073741824') 
    
    % Initialize GPU
    G = gpuDevice();
end

% Get the base-CNN
net = get_resnet(model_file,USEGPU) ;

%================ Prepare dictionary ======================================
% Construct CD
CD_im_folder = fullfile(data_folder,'cd','org');
CD_dfv_folder = fullfile(data_folder,'cd','dfv');
[t1,t2,t3] = mkdir(CD_dfv_folder);
extract_dfv2folder(CD_im_folder, CD_dfv_folder, net, model_type, USEGPU);
A = construct_CD(CD_dfv_folder);

% Construct OED 
OED_org_im_folder = fullfile(data_folder,'oed','org');
OED_org_dfv_folder = fullfile(data_folder,'oed','org_dfv');
[t1,t2,t3] = mkdir(OED_org_dfv_folder);
extract_dfv2folder(OED_org_im_folder, OED_org_dfv_folder, net, model_type, USEGPU);

OED_occ10_im_folder = fullfile(data_folder,'oed','occ','10');
OED_occ10_dfv_folder = fullfile(data_folder,'oed','occ_dfv','10');
[t1,t2,t3] = mkdir(OED_occ10_dfv_folder);
extract_dfv2folder(OED_occ10_im_folder, OED_occ10_dfv_folder, net, model_type, USEGPU);
B10 = construct_OED(OED_occ10_dfv_folder,OED_org_dfv_folder);

OED_occ20_im_folder = fullfile(data_folder,'oed','occ','20');
OED_occ20_dfv_folder = fullfile(data_folder,'oed','occ_dfv','20');
[t1,t2,t3] = mkdir(OED_occ20_dfv_folder);
extract_dfv2folder(OED_occ20_im_folder, OED_occ20_dfv_folder, net, model_type, USEGPU);
B20 = construct_OED(OED_occ20_dfv_folder,OED_org_dfv_folder);

B = cat(2,B10,B20);
clear B10 B20

% Calculate projection matrix
% W = cal_W(A,B,lambda,USEGPU,true); % If the size of your GPU memory is larger than 24 GB, 
                                     % you can replace next line with this line to save computing time.
W = cal_W(A,B,lambda,false,false);  
clear A B

save('W_data.mat', 'W');
% load('W_data.mat');

% Get the classifier
net_softmax = get_resnet_softmax(model_file,false);  % Pre-trained softmax classifier

%================ Evaluate on the test images =============================
folders{1} = fullfile('test','org');       msg{1} = 'occlusion-free result:';
folders{2} = fullfile('test','occ','10');  msg{2} = '10% occlusion result:';
folders{3} = fullfile('test','occ','20');  msg{3} = '20% occlusion result:';

for idx = 1:numel(folders)
    tst_im_folder = fullfile(data_folder,folders{idx});
    
    cls_lst = dir(tst_im_folder); cls_lst = cls_lst(3:end);
    
    N_tst = 0;
    N_SDBE_pos = 0;
    N_OrgNet_pos = 0;
    for ii = 1:numel(cls_lst)
        tst_id = find(strcmp(cls_lst(ii).name,net_softmax.meta.classes.name));
        tst_lst = dir(fullfile(tst_im_folder,cls_lst(ii).name,'*.JPEG'));
        
        SDBE_id = [];
        OrgNet_id = [];
        for jj = 1:numel(tst_lst)
            v = extract_dfv(fullfile(tst_im_folder,cls_lst(ii).name,tst_lst(jj).name),net,model_type,USEGPU);
            
            % Predict with the SDBE_L2 scheme
            v0 = W * v;
            v0 = reshape(v0, [1,1,length(v0),1]);
            net_softmax.eval({'pool5', v0}) ;
            [score,pred] = max(net_softmax.vars(3).value);
            SDBE_id(jj) = pred(1);
            
            % Predict with the original network
            v = reshape(v, [1,1,length(v),1]);
            net_softmax.eval({'pool5', v}) ;
            [score,pred] = max(net_softmax.vars(3).value);
            OrgNet_id(jj) = pred(1);
        end
        N_tst = N_tst + numel(tst_lst);
        N_SDBE_pos = N_SDBE_pos + sum(SDBE_id==tst_id);
        N_OrgNet_pos = N_OrgNet_pos + sum(OrgNet_id==tst_id);
    end
    
    SDBE_rcg_ratio = N_SDBE_pos./N_tst;
    OrgNet_rcg_ratio = N_OrgNet_pos./N_tst;
    
    disp(msg{idx});
    disp('Original network accuracy:');
    disp(OrgNet_rcg_ratio);
    disp('SDBE_L2 scheme accuracy:')
    disp(SDBE_rcg_ratio);
end
