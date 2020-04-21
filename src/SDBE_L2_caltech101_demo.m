close all;
clear all;

% Setup linear SVM
addpath(fullfile('liblinear','matlab'));

% Setup MatConvNet.
% Suppose vl_setupnn.m locates in the folder './matconvnet/matlab'.
vl_setup_file = fullfile('matconvnet','matlab','vl_setupnn.m'); 
run(vl_setup_file);

USEGPU = true; % false: use CPU only 
               % true: use GPU to accelerate computing
model_type = 'ResNet';
model_file = fullfile('cnn_models','imagenet-resnet-152-dag.mat');
data_folder = fullfile('..','datasets','caltech101');

OR = '25';     % 25% occlusion ratio

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
[A,A_lbl] = construct_CD(CD_dfv_folder);
A = normc(double(A));  % normalize the DFVs to unit length
label_set = unique(A_lbl);
for ii = 1:numel(A_lbl)
    A_id(ii) = find(strcmp(A_lbl{ii},label_set));
end

% Construct OED 
OED_org_im_folder = fullfile(data_folder,'oed','org');
OED_org_dfv_folder = fullfile(data_folder,'oed','org_dfv');
[t1,t2,t3] = mkdir(OED_org_dfv_folder);
extract_dfv2folder(OED_org_im_folder, OED_org_dfv_folder, net, model_type, USEGPU);

OED_occ_im_folder = fullfile(data_folder,'oed','occ',OR);
OED_occ_dfv_folder = fullfile(data_folder,'oed','occ_dfv',OR);
[t1,t2,t3] = mkdir(OED_occ_dfv_folder);
extract_dfv2folder(OED_occ_im_folder, OED_occ_dfv_folder, net, model_type, USEGPU);
B = construct_OED(OED_occ_dfv_folder,OED_org_dfv_folder);
B = normc(double(B));  % normalize the OEVs to unit length

%========Example for evaluation on the test images with 25% occlusion======
test_folder = fullfile('test','occ',OR);    msg = sprintf('%s%% occlusion result:',OR);
t = rand()*10;

lambda = 0.1; C = 1; % Hyperparameters of the linear SVM with SDBE_L2 for testing on 25% occlusion.

% Calculate projection matrix
W = cal_W(A,B,lambda,USEGPU,false);

% Train linear SVM
SVMOption = sprintf('-c %f -q',C);
svmModel = train(A_id', sparse(double(A')),SVMOption);

% Test
tst_im_folder = fullfile(data_folder,test_folder);
cls_lst = dir(tst_im_folder); cls_lst = cls_lst(3:end);

N_tst = 0;
N_SDBE_pos = 0;
N_OrgNet_pos = 0;
for ii = 1:numel(cls_lst)
    tst_id = find(strcmp(cls_lst(ii).name,label_set));
    tst_lst = dir(fullfile(tst_im_folder,cls_lst(ii).name,'*.jpg'));
    
    SDBE_id = [];
    OrgNet_id = [];
    for jj = 1:numel(tst_lst)
        v = extract_dfv(fullfile(tst_im_folder,cls_lst(ii).name,tst_lst(jj).name),net,model_type,USEGPU);
        v = normc(double(v));
        
        % Predict with the linear SVM
        [pred, svm_accuracy, svm_prob_estimates] = predict(t, sparse(v(:)'), svmModel,'-q');
        OrgNet_id(jj) = pred;
        
        % Estimate the original DFV with the SDBE_L2
        v0 = W * v;
        v0 = normc(double(v0));
        
        % Predict with the linear SVM
        [pred, svm_accuracy, svm_prob_estimates] = predict(t, sparse(v0(:)'), svmModel,'-q');
        SDBE_id(jj) = pred;
    end
    N_tst = N_tst + numel(tst_lst);
    N_SDBE_pos = N_SDBE_pos + sum(SDBE_id==tst_id);
    N_OrgNet_pos = N_OrgNet_pos + sum(OrgNet_id==tst_id);
end

SDBE_rcg_ratio = N_SDBE_pos./N_tst;
OrgNet_rcg_ratio = N_OrgNet_pos./N_tst;

disp(msg);
disp('Linear SVM without SDBE_L2:');
disp(OrgNet_rcg_ratio);
disp('Linear SVM with SDBE_L2:')
disp(SDBE_rcg_ratio);

