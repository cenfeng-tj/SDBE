function B = construct_OED(occ_dfv_rtfolder,org_dfv_rtfolder)

cls_lst = dir(occ_dfv_rtfolder); cls_lst = cls_lst(3:end);

idx = 1;
for ii = 1:numel(cls_lst)    
    org_dfv_lst = dir(fullfile(org_dfv_rtfolder,cls_lst(ii).name,'*.mat'));
    idx_org = 1;
    for jj = 1:numel(org_dfv_lst)
        s = regexp(org_dfv_lst(jj).name,'\.','split');
        org_name{idx_org} = s{1};
        load(fullfile(org_dfv_rtfolder,cls_lst(ii).name,org_dfv_lst(jj).name));
        wf(:,idx_org) = DFV;
        idx_org = idx_org + 1;
    end
    
    occ_dfv_lst = dir(fullfile(occ_dfv_rtfolder,cls_lst(ii).name,'*.mat'));
    for jj = 1:numel(occ_dfv_lst)
        s = regexp(occ_dfv_lst(jj).name,'-','split');
        load(fullfile(occ_dfv_rtfolder,cls_lst(ii).name,occ_dfv_lst(jj).name));
        for kk = 1:numel(org_name)
            if strcmp(s{1},org_name{kk})
                B(:,idx) = DFV - wf(:,kk);
                idx = idx + 1;
                break;
            end
        end
    end
end

