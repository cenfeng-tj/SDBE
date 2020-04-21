clear all
close all

FULLVALSET = false; % true:  the occluded test images are synthesized from
                    %        full validation set. Note that some validation
                    %        images are encoded in CMYK jpeg format that
                    %        are unable to be read by imread function.
                    %        Please convert these images to RGB jpeg format
                    %        before conducting this script.
                    % false: the occluded test images are synthesized from 
                    %        a subset of the validation set.

ILSVRC2012_srcfolder = '.'; % The root folder where the 'train' and 'val' 
                            % folders locate.
output_folder = '.';   % The root folder where the 'CD', 'OED', and 'test' 
                       % folders locate.  

occluder_file = fullfile('occluders','occluder_ilsvrc_12cls_1percls.mat');
patches = load(occluder_file);

%=========== Prepare training images ======================================
% Generate training images by resizing short side to 256
% and cropping at the center to 224x224
src_rtfolder = fullfile(ILSVRC2012_srcfolder,'train');
org_im_folder = fullfile(output_folder,'cd','org');
[t1,t2,t3] = mkdir(org_im_folder);
src_lst = load('cd_file_list.mat');
resize_crop(src_lst,src_rtfolder,org_im_folder,256,224);

%=========== Prepare extra images =========================================
% Generate occlusion-free extra images by resizing short side to 256
% and cropping at the center to 224x224
src_rtfolder = fullfile(ILSVRC2012_srcfolder,'train');
org_im_folder = fullfile(output_folder,'oed','org');
[t1,t2,t3] = mkdir(org_im_folder);
src_lst = load('oed_org_file_list.mat');
resize_crop(src_lst,src_rtfolder,org_im_folder,256,224);

% Generate occluded extra images by superimposing occlusion patches
ORs = [10,20];      % two occlusion ratios: 10% and 20%
occ_im_rtfolder = fullfile(output_folder,'oed','occ');
[t1,t2,t3] = mkdir(occ_im_rtfolder);
gen_occ_im(patches,org_im_folder,occ_im_rtfolder,ORs);

%============ Prepare the test images from the validation set =============
src_rtfolder = fullfile(ILSVRC2012_srcfolder,'val');
org_im_folder = fullfile(output_folder,'test','org');
[t1,t2,t3] = mkdir(org_im_folder);
src_lst = load('val_subset_file_list.mat');

if FULLVALSET
    %********* Prepare the test images from the full validation set *******
    % Generate occlusion-free test images by resizing short side to 256
    % and cropping at the center to 224x224
    src_lst.files = [];
    
    % Update the files in src_lst according to the files in each subfolder 
    % of src_rtfolder
    for i = 1:numel(src_lst.folders)
        file_lst = dir(fullfile(src_rtfolder,src_lst.folders{i},'*.JPEG'));
        for j = 1:numel(file_lst)
            src_lst.files(i,1).lst{j,1} = file_lst(j).name;
        end
    end
    resize_crop(src_lst,src_rtfolder,org_im_folder,256,224);
else
    %** Prepare the test images from a small subset of the validation set *
    % Generate occlusion-free test images by resizing short side to 256
    % and cropping at the center to 224x224    
    resize_crop(src_lst,src_rtfolder,org_im_folder,256,224); 
end

% Generate occluded test images by superimposing occlusion patches
ORs = [10,20];                  % two occlusion ratios: 10% and 20%
occ_im_rtfolder = fullfile(output_folder,'test','occ');
[t1,t2,t3] = mkdir(occ_im_rtfolder);
gen_occ_im(patches,org_im_folder,occ_im_rtfolder,ORs);


