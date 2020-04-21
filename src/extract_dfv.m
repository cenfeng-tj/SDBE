function v = extract_dfv(im_file,net,model_type,USEGPU)


% read image
im = imread(im_file) ;

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
        v = net.vars(net.getVarIndex('pool5')).value ;
    else
        v = net.vars(net.getVarIndex('cls3_pool')).value ;
    end
    if USEGPU
        v = squeeze(gather(v)) ;
    else
        v = squeeze(v);
    end
else
    display('Unrecognized networks') ;
end
        


