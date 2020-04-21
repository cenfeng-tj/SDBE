function gen_occ_im(patches,org_im_folder,occ_im_rtfolder,ORs)

% The input occlusion-free images need to be 224x224.
% For the ILSVRC dataset, resize the image with the short side equal to 256,
% and then, crop the image into 224x224 in advance.

list = dir(org_im_folder); list = list(3:end);

for jj = 1:numel(ORs)
    OR = ORs(jj);
    occ_im_folder = fullfile(occ_im_rtfolder,sprintf('%d',OR));
    
    for kk = 1:numel(list)
        class = list(kk).name;
        im_list = dir(fullfile(org_im_folder,class,'*.JPEG'));
        occ_im_folder_class = fullfile(occ_im_folder,class);
        [status,msg,msgId] = mkdir(occ_im_folder_class);
        
        for ll = 1:numel(patches.cls)
            for mm = 1:numel(patches.cls(ll).ptch)
                idx = 0;
                for idxOR = 1:numel(patches.cls(ll).ptch(mm).occ)
                    if patches.cls(ll).ptch(mm).occ(idxOR).OR == OR
                        idx = idxOR;
                        break;
                    end
                end
                
                for nn = 1:numel(im_list)
                    org_im = imread(fullfile(org_im_folder,class,im_list(nn).name));
                    
                    pos = patches.cls(ll).ptch(mm).occ(idx).cntpos;
                    occ_im = superimpose_occlusion(org_im, ...
                        patches.cls(ll).ptch(mm).img, ...
                        pos, ...
                        patches.cls(ll).ptch(mm).img_mask, ...
                        patches.cls(ll).ptch(mm).occ(idx).scale ...
                        );
                    im_file = regexp(im_list(nn).name,'\.','split');
                    occ_im_file = sprintf('%s-%s-%d-%d-%s.%s',im_file{1},patches.cls(ll).idName, mm,OR,'cnt',im_file{2});
                    imwrite(occ_im,fullfile(occ_im_folder_class,occ_im_file));
                    
                    for i = 1:numel(patches.cls(ll).ptch(mm).occ(idx).rndpos)
                        pos = patches.cls(ll).ptch(mm).occ(idx).rndpos(i);
                        occ_im = superimpose_occlusion(org_im, ...
                            patches.cls(ll).ptch(mm).img, ...
                            pos, ...
                            patches.cls(ll).ptch(mm).img_mask, ...
                            patches.cls(ll).ptch(mm).occ(idx).scale ...
                            );
                        occ_im_file = sprintf('%s-%s-%d-%d-%s-%d.%s',im_file{1},patches.cls(ll).idName, mm,OR,'rnd',i,im_file{2});
                        imwrite(occ_im,fullfile(occ_im_folder_class,occ_im_file));
                    end
                end
            end
        end
    end
end

end

function im = superimpose_occlusion(im,ptch,pos,msk,scale)
[H,W,C] = size(im);
[m_ptch,n_ptch,k_ptch] = size(ptch);
up_scale = 1./scale;
im_r = imresize(im,up_scale);
[r_H,r_W,r_C] = size(im_r);

y_s = max(floor(pos.h.*up_scale),1);
x_s = max(floor(pos.w.*up_scale),1);
y_e = min(y_s+m_ptch-1, r_H);
x_e = min(x_s+n_ptch-1, r_W);
ptch = ptch(1:y_e-y_s+1,1:x_e-x_s+1,:);
msk = msk(1:y_e-y_s+1,1:x_e-x_s+1,:);

im_ptch = im_r(y_s:y_e,x_s:x_e,:);
im_ptch(msk==1) = ptch(msk==1);
im_r(y_s:y_e,x_s:x_e,:)=im_ptch;

im = imresize(im_r,[H,W]);
end
