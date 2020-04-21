function varargout = construct_CD(dfv_rtfolder)

nout = nargout;
cls_lst = dir(dfv_rtfolder); cls_lst = cls_lst(3:end);
idx = 1;
for ii = 1:numel(cls_lst)
    if cls_lst(ii).isdir        
        dfv_lst = dir(fullfile(dfv_rtfolder,cls_lst(ii).name,'*.mat'));
        for jj = 1:numel(dfv_lst)
            load(fullfile(dfv_rtfolder,cls_lst(ii).name,dfv_lst(jj).name));            
            A(:,idx) = DFV;
            label{idx} = cls_lst(ii).name;
            idx = idx+1;
        end
    end
end

varargout{1} = A;
if nout == 2
    varargout{2} = label;
end
    
