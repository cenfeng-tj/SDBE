function resize_crop(src_lst,src_rtfolder,dst_rtfolder,R_SZ,C_SZ)
% First, convert one channel images into three channel images.
% Then, resize the original image to short side equal to R_SZ, and crop
% a C_SZ x C_SZ patch at the center of the resized image.
% src: the folder containing the class folders of original images.
% dst: the folder containing the class folders used to store the resized
%      and cropped images.
% R_SZ: the resizing length of the image short side
% C_SZ: the side length of the cropped patch

cls_lst = src_lst.folders;

for ii = 1:numel(cls_lst)
    im_lst = src_lst.files(ii).lst;
    dst_folder = fullfile(dst_rtfolder,cls_lst{ii});
    [status,msg,msgId] = mkdir(dst_folder);
    
    for jj = 1:numel(im_lst)
        im = imread(fullfile(src_rtfolder,cls_lst{ii},im_lst{jj}));
        im_sz = size(im);
        
        % Convert one channel image to three channel image
        if size(im,3)==1
            im(:,:,2) = im(:,:,1);
            im(:,:,3) = im(:,:,1);
        end
        
        % Resize the image
        if im_sz(1) >= im_sz(2)
            h = round(im_sz(1)/im_sz(2).*R_SZ);
            w = R_SZ;
        else
            h = R_SZ;
            w = round(im_sz(2)/im_sz(1).*R_SZ);
        end
        im = imresize(im,[h,w]);
        
        % Crop the image
        x_min = ceil((w-C_SZ)/2);   % Central crop
        y_min = ceil((h-C_SZ)/2);   % Central crop
        h_c = C_SZ-1;
        h_w = C_SZ-1;
        rect = [x_min, y_min, h_w,h_c];        
        im_c = imcrop(im,rect);
        
        imwrite(im_c,fullfile(dst_folder,im_lst{jj}));
    end    
end


