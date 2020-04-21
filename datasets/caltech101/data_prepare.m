clear all
close all

% Download and untar Caltech 101 dataset
untar('http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz')

caltech101_srcfolder = '101_ObjectCategories'; % The root folder of Caltech
                                               % 101 dataset.
output_folder = '.';   % The root folder where the 'CD', 'OED', and 'test' 
                       % folders locate.  

occluder_file = fullfile('occluders','occluder_10cls_1percls.mat');
patches = load(occluder_file);

src_rtfolder = caltech101_srcfolder;

%=========== Prepare training images ======================================
% Copy training images to 'cd/org' folder
org_im_folder = fullfile(output_folder,'cd','org');
[t1,t2,t3] = mkdir(org_im_folder);
src_lst = load('cd_file_list.mat');
copy_imgs(src_lst,src_rtfolder,org_im_folder);

%=========== Prepare extra images =========================================
% Copy occlusion-free extra images to 'oed/org' folder
org_im_folder = fullfile(output_folder,'oed','org');
[t1,t2,t3] = mkdir(org_im_folder);
src_lst = load('oed_org_file_list.mat');
copy_imgs(src_lst,src_rtfolder,org_im_folder);

% Generate occluded extra images by superimposing occlusion patches
ORs = [5,15,25,35];      % four occlusion ratios: 5%,15%,25%, and 35%    
occ_im_rtfolder = fullfile(output_folder,'oed','occ');
[t1,t2,t3] = mkdir(occ_im_rtfolder);
gen_resize_occ_im(patches,org_im_folder,occ_im_rtfolder,ORs);

%============ Prepare the test images from the validation set =============
% Copy occlusion-free test images to 'test/org' folder
org_im_folder = fullfile(output_folder,'test','org');
[t1,t2,t3] = mkdir(org_im_folder);
src_lst = load('test_org_file_list.mat');
copy_imgs(src_lst,src_rtfolder,org_im_folder);

% Generate occluded test images by superimposing occlusion patches
ORs = [5,15,25,35];      % four occlusion ratios: 5%,15%,25%, and 35%   
occ_im_rtfolder = fullfile(output_folder,'test','occ');
[t1,t2,t3] = mkdir(occ_im_rtfolder);
gen_resize_occ_im(patches,org_im_folder,occ_im_rtfolder,ORs);
