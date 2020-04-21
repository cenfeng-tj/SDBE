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

% Prepare SDBE_L1
A = double(A);
n_A = size(A,2);
D = cat(2,A,double(B)); clear B
if USEGPU
    D = gpuArray(D);
    Dt = gpuArray(D');
    DD = D * Dt;
else
    Dt = D';
    DD = D * Dt;
end

%========Examples for evaluation on the test images =======
test_folders{1} = fullfile('test','org');       msg{1} = 'occlusion-free result:';
test_folders{2} = fullfile('test','occ',OR);    msg{2} = sprintf('%s%% occlusion result:',OR);

lambda_set = [0.005,0.01];   % Hyperparameters of the SDBE_L1
C_set = [2,1];               % Hyperparameters of the linear SVM

t = rand()*10;

for idx = 1:numel(test_folders)
    lambda = lambda_set(idx);
    C = C_set(idx);
    
    % Train linear SVM
    SVMOption = sprintf('-c %f -q',C);
    svmModel = train(A_id', sparse(A'),SVMOption);
    
    % Test
    tst_im_folder = fullfile(data_folder,test_folders{idx});
    cls_lst = dir(tst_im_folder); cls_lst = cls_lst(3:end);
    
    N_tst = 0;
    N_SDBE_pos = 0;
    for ii = 1:numel(cls_lst)
        tst_id = find(strcmp(cls_lst(ii).name,label_set));
        tst_lst = dir(fullfile(tst_im_folder,cls_lst(ii).name,'*.jpg'));
        
        SDBE_id = [];
        for jj = 1:numel(tst_lst)
            v = extract_dfv(fullfile(tst_im_folder,cls_lst(ii).name,tst_lst(jj).name),net,model_type,USEGPU);
            v = normc(double(v));
            
            % Estimate the original DFV with the SDBE_L1             
            [omega, nIter] = L1LS_DALM(DD, D, Dt, v, USEGPU, 'lambda',lambda,'tolerance',1e-3);
            v0= A*omega(1:n_A);
            v0 = normc(double(v0));
            
            % Predict with the linear SVM
            [pred, svm_accuracy, svm_prob_estimates] = predict(t, sparse(v0(:)'), svmModel,'-q');
            SDBE_id(jj) = pred;
            
        end
        N_tst = N_tst + numel(tst_lst);
        N_SDBE_pos = N_SDBE_pos + sum(SDBE_id==tst_id);
    end
    
    SDBE_rcg_ratio = N_SDBE_pos./N_tst;
    
    disp(msg{idx});
    disp('Linear SVM with SDBE_L1 accuracy:')
    disp(SDBE_rcg_ratio);
end
