classNames = {"plane1","car2","bird3","cat4","deer5","dog6","amphibian7","horse8","boat9","truck10"}

function display(trainOrTest,X,Y,YPred)
	rObsDis = torch.random(batchSize)
	xDis = X:narrow(1,rObsDis,1)
	yDis = Y:narrow(1,rObsDis,1)
	yPredDis = YPred:narrow(1,rObsDis,1)
	if imgTr == nil then
		local initPic = torch.rand(32,32)
		imgTr = image.display{image=initPic,zoom=10}
		imgTe = image.display{image=initPic,zoom=10}
	end
	_, argmax = torch.max(yPredDis,2)
	local title = string.format("Pred = %d (%s),Truth = %d (%s)",argmax:squeeze(),
					classNames[argmax:squeeze()],yDis:squeeze(),classNames[yDis:squeeze()])
	if trainOrTest == "train" then 
		image.display{image=xDis,win=imgTr,legend=title}
	elseif trainOrTest == "test" then
		image.display{image=xDis,win=imgTe,legend=title}
	end
end
