function extract_dfv2folder(im_rtfolder, dfv_rtfolder, net, model_type, USEGPU)


cls_lst = dir(im_rtfolder) ; cls_lst = cls_lst(3:end) ;
for ii = 1:numel(cls_lst)
    cls_folder = fullfile(im_rtfolder,cls_lst(ii).name) ;
    im_lst = dir(cls_folder); im_lst = im_lst(3:end) ;
    
    dfv_folder = fullfile(dfv_rtfolder,cls_lst(ii).name);
    [status,msg,msgid] = mkdir(dfv_folder) ;
    
    for jj = 1:numel(im_lst)
        % fetch an image
        im = imread(fullfile(cls_folder,im_lst(jj).name)) ;
        
        % preprocess the image
        im = single(im) ; % pixel value in [0,255].
        if size(im,3)==1
            im(:,:,2) = im(:,:,1) ;
            im(:,:,3) = im(:,:,1) ;
        end
        im_sz = size(im);
        if any(im_sz(1:2) ~= net.meta.normalization.imageSize(1:2))
            im = imresize(im, net.meta.normalization.imageSize(1:2)) ;
        end
        im = bsxfun(@minus, im, net.meta.normalization.averageImage) ;
        
        % extract DFV
        if strcmp(model_type, 'ResNet')||strcmp(model_type, 'GoogLeNet')
            if USEGPU
                net.eval({'data', gpuArray(im)});
            else
                net.eval({'data', im});
            end
            if strcmp(model_type, 'ResNet')
                DFV = net.vars(net.getVarIndex('pool5')).value ;
            else
                DFV = net.vars(net.getVarIndex('cls3_pool')).value ;
            end
            if USEGPU
                DFV = squeeze(gather(DFV)) ;
            else
                DFV = squeeze(DFV);
            end            
        else
            display('Unrecognized networks') ;
        end
        
        % save DFV
        s = regexp(im_lst(jj).name,'\.','split');
        save(fullfile(dfv_folder, [im_lst(jj).name(1:end-length(s{end})) 'mat']), 'DFV');
    end
end

