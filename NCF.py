import pandas as pd
import numpy as np
import math
import time
import json
import collections
import re

import torch
from torch import nn
from torch.autograd import Variable


class myAutoencoder2(nn.Module):
	def __init__(self, NumHidden,inputSize,outputSize):
		super(myAutoencoder2, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(inputSize, NumHidden[0]),# bias=False),
			nn.ReLU(True),
			nn.Linear(NumHidden[0], NumHidden[1]))# bias=False),
			#nn.ReLU(True),
			#nn.Linear(NumHidden[1], NumHidden[2]),
			#nn.ReLU(True),
			#nn.Linear(NumHidden[2], NumHidden[3]))#,
			#nn.ReLU(True),
			#nn.Linear(NumHidden[3], NumHidden[4]))
		self.decoder = nn.Sequential(
			#nn.Linear(NumHidden[4], NumHidden[3]),
			#nn.ReLU(True),
			#nn.Linear(NumHidden[3], NumHidden[2]),
			#nn.ReLU(True),
			#nn.Linear(NumHidden[2], NumHidden[1]),
			#nn.ReLU(True),
			nn.Linear(NumHidden[1], NumHidden[0]),# bias=False),
			nn.ReLU(True),
			nn.Linear(NumHidden[0], outputSize),# bias=False),
			nn.Tanh())

	def forward(self, x):
		outputgrade = self.encoder(x)
		return outputgrade, self.decoder(outputgrade)

class FClinear(nn.Module):
	def __init__(self, inputSize):
		super(FClinear, self).__init__()
		self.ff5 = nn.Sequential(
			nn.Linear(inputSize, 1))
	def forward(self,x1):
		x = self.ff5(x1)
		return x

class RMSE_NCF:
	def __init__(self, dataset, args):
		print('In class RMSE_NCF')
		self.dataset = dataset
		self.args = args
		oldtime = time.time()
		if self.args.ifRead == 1:
			self.userVec, self.itemVec, self.instrVec, self.trainingSet, self.testSet = self.dataset.read_generate_vector_set_ncf(self.args.userFile, \
																														self.args.itemFile, \
																														self.args.instrFile, \
																														self.args.trainFile, \
																														self.args.testFile)
		else:
			self.userVec, self.itemVec, self.instrVec, self.trainingSet, self.testSet = self.dataset.generate_vector_set_ncf()
			self.dataset.write_generate_vector_set_ncf(self.args.userFile, \
															self.args.itemFile, \
															self.args.instrFile, \
															self.args.trainFile, \
															self.args.testFile, \
															self.userVec, \
															self.itemVec, \
															self.instrVec, \
															self.trainingSet, \
															self.testSet)
		with open(self.args.paraDictPath) as json_file:
			self.paraDict = json.load(json_file)


	def train(self):
		paraDict = self.paraDict
		trainingSet, testSet = self.trainingSet, self.testSet
		N, M, L = len(self.userVec), len(self.itemVec), len(self.instrVec)
		K = paraDict['K']
		k=[10,10,10]
		maeVec, pct0Vec, pct1Vec, pct2Vec = [],[],[],[]
		stdID, crsID, instrID = np.zeros((N, N),dtype=np.float32),np.zeros((M, M),dtype=np.float32),np.zeros((L, L),dtype=np.float32)
		np.fill_diagonal(stdID, 1)
		np.fill_diagonal(crsID, 1)
		np.fill_diagonal(instrID, 1)
		stdID, crsID, instrID = Variable(torch.from_numpy(stdID)),Variable(torch.from_numpy(crsID)), Variable(torch.from_numpy(instrID))
		l1 = nn.Linear(N,k[0])
		l2 = nn.Linear(M,k[1])
		l3 = nn.Linear(L,k[2])
		criterion = nn.MSELoss()
		model = myAutoencoder2([10,1], sum(k),sum(k))
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
		optimizerl1 = torch.optim.Adam(l1.parameters(), lr=1e-4, weight_decay=1e-5)
		optimizerl2 = torch.optim.Adam(l2.parameters(), lr=1e-4, weight_decay=1e-5)
		optimizerl3 = torch.optim.Adam(l3.parameters(), lr=1e-4, weight_decay=1e-5)
		oldmae = 10
		maeCt = 0
		for x in range(paraDict['maxIter']):
			oldtime = time.time()
			predGrdVec, trueGrdVec = [], []
			FFInputCrs,FFInputCoCrs,targetGrd = [],[],[]
			for std in list(trainingSet.keys()):
				tempStdVec = trainingSet[std]
				std = float(std)
				std = self.userVec.index(std)
				for tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					#
					#  go through training samples
					#
					for crsCode, insCode, grd in tempTermVec:
						crsj = self.itemVec.index(crsCode)
						instrl = self.instrVec.index(insCode)
						inputVec = torch.cat((l1(stdID[std]),l2(crsID[crsj]),l3(instrID[instrl])), 0)
						outputgrade, output = model(inputVec)
						target = Variable(torch.from_numpy(np.asarray([grd])))
						target = target.float()
						inputVec = inputVec.float()
						loss=criterion(outputgrade, target)
						optimizer.zero_grad()
						optimizerl1.zero_grad()
						optimizerl2.zero_grad()
						optimizerl3.zero_grad()
						loss.backward()
						optimizer.step()
						optimizerl1.step()
						optimizerl2.step()
						optimizerl3.step()
			mae, rmse, pct0, pct1, pct2 = self.testNCFwithInstr(model,l1,l2,l3,stdID,crsID,instrID)
			print("NCF -  %d th iter:	rmse & mae   pct0 & pct1 & pct2:		%.6f  &  %.6f		  %.6f  &  %.6f  &  %.6f"%(x+1,rmse,mae, pct0, pct1, pct2))
			maeVec.append(mae)
			pct0Vec.append(pct0)
			pct1Vec.append(pct1)
			pct2Vec.append(pct2)
			if mae >= oldmae:
				if maeCt < 3:
					maeCt+=1
					oldmae = mae
				else:
					break
			else:
				oldmae = mae
				maeCt=0
		minIndex = maeVec.index(min(maeVec))
		print('NCF  %.3f  &  %.3f  &  %.3f  &  %.3f\n'%(maeVec[minIndex],pct0Vec[minIndex],pct1Vec[minIndex],pct2Vec[minIndex]))


	def testNCFwithInstr(self, model,l1,l2,l3,stdID,crsID,instrID):
		trainingSet, testSet = self.trainingSet, self.testSet
		paraDict = self.paraDict
		groundTruth, predictResult = [], []
		for std in list(testSet.keys()):
			tempStdVec = testSet[std]
			std = float(std)
			std = self.userVec.index(std)
			for tempTerm in list(tempStdVec.keys()):
				tempTermVec = tempStdVec[tempTerm]
				for crsCode, insCode, testGrd in tempTermVec:
					crsj = self.itemVec.index(crsCode)
					instrl = self.instrVec.index(insCode)
					inputVec = torch.cat((l1(stdID[std]),l2(crsID[crsj]),l3(instrID[instrl])), 0)
					outputgrade, output = model(inputVec)
					outputgrade = outputgrade[0]
					outputgrade = outputgrade.data.numpy()
					predictResult.append(outputgrade)
					groundTruth.append(testGrd)
		predG, trueG = np.asarray(predictResult), np.asarray(groundTruth)
		return self.testRMSE(predG, trueG)

	def testRMSE(self, predGrdVec, trueGrdVec):
		if len(predGrdVec) == 0:
			return 0,0,0,0,0
		#
		predGrdVec[predGrdVec<0] = 0
		predGrdVec[predGrdVec>4] = 4
		mae = np.asarray(list(map(lambda x: math.fabs(x), trueGrdVec-predGrdVec)))
		maeSD = np.std(mae)
		mae = sum(mae)/len(mae)
		rmse = np.asarray(list(map(lambda x: math.pow(x, 2), trueGrdVec-predGrdVec)))
		rmse = math.sqrt(sum(rmse)/len(rmse))
		#
		result = abs(trueGrdVec-predGrdVec)
		l = len(result)
		pct0 = (float)((result <= 0.5*0.34).sum())/(float)(l)
		pct1 = (float)((result <= 0.34).sum())/(float)(l)
		pct2 = (float)((result <= 2*0.34).sum())/(float)(l)
		return rmse, mae, pct0, pct1, pct2
