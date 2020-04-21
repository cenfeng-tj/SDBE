clear all 
close all

dst_folder = 'liblinear';

% Download linear SVM package and extract files
url = 'http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/liblinear.cgi?+http://www.csie.ntu.edu.tw/~cjlin/liblinear+tar.gz';
o = weboptions('CertificateFilename','');
websave('liblinear.tar.gz',url,o)
filenames = untar('liblinear.tar.gz','.');
delete('liblinear.tar.gz');

% Move the files to dst_folder
movefile(filenames{1},dst_folder);

% Compile matlab interface
cd(fullfile(dst_folder,'matlab'));
make

cd(fullfile('..','..'));