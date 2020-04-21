function net = get_resnet(model_file,USEGPU)

net = dagnn.DagNN.loadobj(load(model_file)) ;
net.removeLayer('prob');
net.removeLayer('fc1000');

% remove loss and dropout layers
dagRemoveLayersOfType(net, 'dagnn.Loss') ;
dagRemoveLayersOfType(net, 'dagnn.DropOut') ;
% remove Batch Norm layers
dagMergeBatchNorm(net) ;
dagRemoveLayersOfType(net, 'dagnn.BatchNorm') ;
if USEGPU
    % move net.params.value to gpu
    net.move('gpu');
end
net.mode = 'test' ;

end

% -------------------------------------------------------------------------
function layers = dagFindLayersWithOutput(net, outVarName)
% -------------------------------------------------------------------------
layers = {} ;
for l = 1:numel(net.layers)
  if any(strcmp(net.layers(l).outputs, outVarName))
    layers{1,end+1} = net.layers(l).name ;
  end
end
end
% -------------------------------------------------------------------------
function layers = dagFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = [] ;
for l = 1:numel(net.layers)
    if isa(net.layers(l).block, type)
        layers{1,end+1} = net.layers(l).name ;
    end
end
end

% -------------------------------------------------------------------------
function dagRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
names = dagFindLayersOfType(net, type) ;
for i = 1:numel(names)
    layer = net.layers(net.getLayerIndex(names{i})) ;
    net.removeLayer(names{i}) ;
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end
end
% -------------------------------------------------------------------------
function dagMergeBatchNorm(net)
% -------------------------------------------------------------------------
names = dagFindLayersOfType(net, 'dagnn.BatchNorm') ;
for name = names
    name = char(name) ;
    layer = net.layers(net.getLayerIndex(name)) ;
    
    % merge into previous conv layer
    playerName = dagFindLayersWithOutput(net, layer.inputs{1}) ;
    playerName = playerName{1} ;
    playerIndex = net.getLayerIndex(playerName) ;
    player = net.layers(playerIndex) ;
    if ~isa(player.block, 'dagnn.Conv')
        error('Batch normalization cannot be merged as it is not preceded by a conv layer.') ;
    end
    
    % if the convolution layer does not have a bias,
    % recreate it to have one
    if ~player.block.hasBias
        block = player.block ;
        block.hasBias = true ;
        net.renameLayer(playerName, 'tmp') ;
        net.addLayer(playerName, ...
            block, ...
            player.inputs, ...
            player.outputs, ...
            {player.params{1}, sprintf('%s_b',playerName)}) ;
        net.removeLayer('tmp') ;
        playerIndex = net.getLayerIndex(playerName) ;
        player = net.layers(playerIndex) ;
        biases = net.getParamIndex(player.params{2}) ;
        net.params(biases).value = zeros(block.size(4), 1, 'single') ;
    end
    
    filters = net.getParamIndex(player.params{1}) ;
    biases = net.getParamIndex(player.params{2}) ;
    multipliers = net.getParamIndex(layer.params{1}) ;
    offsets = net.getParamIndex(layer.params{2}) ;
    moments = net.getParamIndex(layer.params{3}) ;
    
    [filtersValue, biasesValue] = mergeBatchNorm(...
        net.params(filters).value, ...
        net.params(biases).value, ...
        net.params(multipliers).value, ...
        net.params(offsets).value, ...
        net.params(moments).value) ;
    
    net.params(filters).value = filtersValue ;
    net.params(biases).value = biasesValue ;
end
end
% -------------------------------------------------------------------------
function [filters, biases] = mergeBatchNorm(filters, biases, multipliers, offsets, moments)
% -------------------------------------------------------------------------
% wk / sqrt(sigmak^2 + eps)
% bk - wk muk / sqrt(sigmak^2 + eps)
a = multipliers(:) ./ moments(:,2) ;
b = offsets(:) - moments(:,1) .* a ;
biases(:) = biases(:) + b(:) ;
sz = size(filters) ;
numFilters = sz(4) ;
filters = reshape(bsxfun(@times, reshape(filters, [], numFilters), a'), sz) ;
end
