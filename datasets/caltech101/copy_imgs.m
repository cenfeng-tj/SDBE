function copy_imgs(src_lst,src_rtfolder,dst_rtfolder)
% Copy the files in src_lst from src_rtfolder to dst_rtfolder
% src_rtfolder: the folder containing the class folders of original images.
% dst_rtfolder: the folder containing the class folders used to store the resized
%               and cropped images.

for ii = 1:numel(src_lst.folders)
    im_lst = src_lst.files(ii).lst;
    dst_folder = fullfile(dst_rtfolder,src_lst.outfolders{ii});
    [status,msg,msgId] = mkdir(dst_folder);
    
    for jj = 1:numel(im_lst)
        copyfile(fullfile(src_rtfolder,src_lst.folders{ii},im_lst{jj}),fullfile(dst_folder,im_lst{jj}));        
    end    
end


