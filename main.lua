require "optim"
require "nn"
require "cunn"
require "optim"
require "gnuplot"
lapp = require "pl.lapp"

c = require 'trepl.colorize'
dofile("loadData.lua")
dofile("models.lua")
dofile("display.lua")

opt = lapp[[
	-l --load    	   (default true)
	-r --run           (default false)
	--learningRate     (default 1)
]]

nClasses = 10
feed = data.init()
criterion = nn.CrossEntropyCriterion():cuda()

if opt.l == true then print("Loading model"); model = torch.load("nn.model") else print("New model"); model = makeModel() end
parameters, gradParameters = model:getParameters()
trainCm, testCm = optim.ConfusionMatrix(10), optim.ConfusionMatrix(10)
batchSize = 200
lr = 1

function trainEpoch()

	model:training()
	epoch = epoch or 1
	lr = (lr - 0.02) 
	optimState = {learningRate = lr, weightDecay = 0.0005, momentum = 0.9, learningRateDecay = 1e-7}

	print("Training epoch number ", epoch, optimState)

	trTimer = torch.Timer()
	local losses = {}

	while true do
		feed:getNextFile("train")
		nObs = feed.Y:size(1)
		print(string.format("Training batch %d out of 5",feed.currentFile))
		indices = torch.randperm(nObs):long():split(batchSize)

		--for i = 1, nObs,batchSize do
		for k,v in ipairs(indices) do 

			local X,Y, YPred, dLdO
			X = feed.X:index(1,v):cuda()
			Y = feed.Y:index(1,v):cuda()
			YPred = model:forward(X)

			feval = function(x) 
				if x~= parameters then parameters:copy(x) end
				gradParameters:zero()
				trainCm:batchAdd(YPred,Y)
				loss = criterion:forward(YPred,Y)
				table.insert(losses,loss)
				dLdO = criterion:backward(YPred,Y)
				model:backward(X,dLdO)
				return loss, gradParameters
			end
			display("train",X,Y,YPred)
			xlua.progress(k,#indices)
			optim.sgd(feval,parameters,optimState)

			collectgarbage()

		end

		if feed.currentFile == 5 then
			break
		end
		print(trainCm)

	end
	meanLoss = torch.Tensor(losses):mean()
	acc = trainCm.mat:diag():sum()/trainCm.mat:sum()*100

	epoch = epoch + 1
	print(string.format("Finished train epoch no %d taking %f seconds.",epoch,trTimer:time().real))
	print(string.format("Mean train loss = %s and accuracy = %s .",c.Blue(meanLoss),c.Red(acc)))

	print("Saving")
	torch.save("nn.model",model)
	trainCm:zero()
	trTimer:reset()
	return meanLoss, acc/100
end

function testEpoch()
	print("Testing epoch number \n\n", epoch)
	feed:getNextFile("test")
	model:evaluate()
	nObs = feed.Y:size(1)
	losses = {}
	for i = 1, nObs,batchSize do
		local X,Y, YPred
		X = feed.X:narrow(1,i,batchSize)
		Y = feed.Y:narrow(1,i,batchSize)
		X = X:cuda()
		Y = Y:cuda()
		YPred = model:forward(X) 
		display("test",X,Y,YPred)
		loss = criterion:forward(YPred,Y)
		testCm:batchAdd(YPred,Y)
		table.insert(losses,loss)
	end
	print(testCm)
	acc = testCm.totalValid*100
	meanLoss = torch.Tensor(losses):mean()
	print(string.format("Mean test loss = %s and accuracy = %s .",c.Blue(meanLoss),c.Red(acc)))
	testCm:zero()
	return meanLoss, acc/100
end

function run() 
	print(model)
	nEpochs = 10
	x = torch.range(1,nEpochs)
	trLosses,trAccs = torch.zeros(nEpochs), torch.zeros(nEpochs)
	teLosses,teAccs = torch.zeros(nEpochs), torch.zeros(nEpochs)
	for i=1,10 do
		trLosses[i], trAccs[i] = trainEpoch()
		teLosses[i], teAccs[i] = testEpoch()
		gnuplot.plot({'Train losses', x,trLosses,'-'},{'Train acc', x,trAccs,'-'},{'Test losses', x,teLosses,'-'},{'Test acc', x,teAccs,'-'})
		gnuplot.axis{1,i,0,2}
	end

end

if opt.r == true then run() end

