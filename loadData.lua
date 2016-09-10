require "image"
require "xlua"

data = {}
data.__index = data

function data.init()
	local self = {}
	self.classNames = {"plane1","car2","bird3","cat4","deer5","dog6","amphibian7","horse8","boat9","truck10"}
	self.currentFile = 0
	setmetatable(self,data)
	return self
end

function data:getNames(labels)
	local out = {}
	for i = 1, labels:size(1) do
		table.insert(out,self.classNames[labels[i]:long()[1]])
	end
	return out
end


function data:getNextFile(trainOrTest)

	local path 
	assert(trainOrTest,"please enter 'train' or 'test'")
	if trainOrTest == "train" then  
		if self.currentFile == 5 then self.currentFile = 1 else self.currentFile = self.currentFile + 1 end
		path = string.format("/Users/matt/kaggle/cifar10/cifar-10-batches-bin/data_batch_%d.bin",self.currentFile)
	elseif trainOrTest == "test" then
		path = "/Users/matt/kaggle/cifar10/cifar-10-batches-bin/test_batch.bin"
	end
	
	local dataBatch = torch.ByteTensor(torch.ByteStorage(path))
	local nObs = 1e4
	local nBytes = 3*32*32
	local n = dataBatch:size(1)
	self.X = torch.DoubleTensor(nObs,nBytes)
	self.Y = torch.DoubleTensor(nObs,1)

	local i, from
	for i = 0,nObs-1 do
		from = i*(nBytes+1) + 1
		self.Y[i+1] = dataBatch:narrow(1,from,1) + 1 
		self.X[i+1] = dataBatch:narrow(1,from+1,nBytes) 
	end

	self.X:resize(nObs,3,32,32)
	self.names = self:getNames(self.Y)

	collectgarbage()

end






