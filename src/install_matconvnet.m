clear all 
close all

dst_folder = 'matconvnet';

% Download MatConvNet package and extract files
url = 'https://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta24.tar.gz';
o = weboptions('CertificateFilename','');
websave('matconvnet.tar.gz',url,o);
filenames = untar('matconvnet.tar.gz','.');
delete('matconvnet.tar.gz');

% Move the files to dst_folder
movefile(filenames{2},dst_folder);

% Compile matlab interface
cd(dst_folder);
addpath('matlab');
vl_compilenn('enableGpu', true); % If you have a GPU.
% vl_compilenn                   % If you do not have a GPU. 
                                 % See "https://www.vlfeat.org/matconvnet/install/#compiling" for more options

cd('..');

% Download pre-trained ResNet152 model
url = 'https://www.vlfeat.org/matconvnet/models/imagenet-resnet-152-dag.mat';
o = weboptions('CertificateFilename','');
websave(fullfile('cnn_models','imagenet-resnet-152-dag.mat'),url,o);