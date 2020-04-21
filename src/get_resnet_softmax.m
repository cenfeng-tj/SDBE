function net = get_resnet_softmax(model_file,USEGPU)

net = dagnn.DagNN.loadobj(load(model_file)) ;
NN = numel(net.layers);
for ii = 1:NN-2
    net.removeLayer(net.layers(1).name);
end
net.meta.inputs.name='pool5';
net.meta.inputs.size = [1,1,2048,1];
net.meta.normalization = [];
if USEGPU
    % move net.params.value to gpu
    net.move('gpu');
end
net.mode = 'test' ;