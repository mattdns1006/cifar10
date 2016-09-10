local function MSRinit(net)
	local function init(name)
	for k,v in pairs(net:findModules(name)) do
	local n = v.kW*v.kH*v.nOutputPlane
	v.weight:normal(0,math.sqrt(2/n))
	v.bias:zero()
	end
	end
	init'nn.SpatialConvolution'
end

function makeModel()
	local model = nn.Sequential()
	nFeats = 24 
	for i = 1, 4 do

		nOut = nFeats + 24

		if i == 1 then 
			model:add(nn.SpatialConvolution(3,nOut,3,3,1,1,1,1))
		else
			model:add(nn.SpatialConvolution(nFeats,nOut,3,3,1,1,1,1))

		end
		model:add(nn.SpatialBatchNormalization(nOut))
		model:add(nn.ReLU())
		model:add(nn.SpatialConvolution(nOut,nOut,3,3,2,2,1,1))
		model:add(nn.SpatialBatchNormalization(nOut))
		model:add(nn.ReLU())
		model:add(nn.Dropout(0.1*i))

		nFeats = nOut
	end


	local n = nOut*2*2
	model:add(nn.View(n))
	model:add(nn.Linear(n,nClasses))

	model:cuda()
	MSRinit(model)
	return model

end
