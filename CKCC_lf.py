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


class FC(nn.Module):
	def __init__(self, inputSize, hiddenSize):
		super(FC, self).__init__()
		self.ff5 = nn.Sequential(
			nn.Linear(inputSize, hiddenSize[0]),
			nn.ReLU(True),
			nn.Linear(hiddenSize[0], hiddenSize[1]))
			#nn.Tanh())
			#nn.ReLU(True),
			#nn.Linear(hiddenSize[1], hiddenSize[2]))
			#nn.ReLU(True),
			#nn.Linear(hiddenSize[2], hiddenSize[3]))
			#nn.ReLU(True),
			#nn.Linear(hiddenSize[3], hiddenSize[4]))
	def forward(self,x1):
		x = self.ff5(x1)
		return x

class FClinear(nn.Module):
	def __init__(self, inputSize):
		super(FClinear, self).__init__()
		self.ff5 = nn.Sequential(
			nn.Linear(inputSize, 1))
	def forward(self,x1):
		x = self.ff5(x1)
		return x

class RMSE_CKCC_lf:
	def __init__(self, dataset, args):
		print('In class RMSE_CKCC')
		self.dataset = dataset
		self.args = args
		self.lf=self.args.lf
		if self.args.ifRead == 1:
			self.userVec, self.itemVec, self.trainingSet, self.testSet = self.dataset.read_generate_vector_set_ck(self.args.userFile, \
																												  self.args.itemFile, \
																												  self.args.trainFile, \
																												  self.args.testFile)
		else:
			self.userVec, self.itemVec, self.trainingSet, self.testSet = self.dataset.generate_vector_set_ck()
			self.dataset.write_generate_vector_set_ck(self.args.userFile, \
													  self.args.itemFile, \
													  self.args.trainFile, \
													  self.args.testFile, \
													  self.userVec, \
													  self.itemVec, \
													  self.trainingSet, \
													  self.testSet)
		with open(self.args.paraDictPath) as json_file:
			self.paraDict = json.load(json_file)
		#
		self.criterion = nn.MSELoss()
		if self.args.coCrsSum == 1:
			inputSize = self.paraDict['inputSize']*2
		else:
			inputSize = self.paraDict['inputSize']
		if self.args.islinear == 0:
			self.model = FC(inputSize, self.paraDict['hiddenSize'])
		else:
			self.model = FClinear(inputSize)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-3)

	def train(self):
		paraDict = self.paraDict
		trainingSet, testSet = self.trainingSet, self.testSet
		K = paraDict['K']
		print('K - ',K)
		N, M = len(self.userVec), len(self.itemVec)
		R = np.random.rand(M,K) # previous
		Q = np.random.rand(M,K) # current
		bs = [0 for i in range(N)]
		bc = [0 for i in range(M)]

		oldmae, oldrmse = 100, 100
		maeCt = 0
		maeVec, pct0Vec, pct1Vec, pct2Vec = [],[],[],[]
		testpredGrdGrp_crs_vec = []
		testpredGrdGrp_mjr_vec = []

		for xx in range(paraDict['maxIter']):
			oldtime = time.time()
			predGrdVec, trueGrdVec = [], []
			FFInputCrs,FFInputCoCrs,targetGrd = [],[],[]
			for std in list(trainingSet.keys()):
				tempStdVec = trainingSet[std]
				std = float(std)
				std = self.userVec.index(std)
				for tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					train = tempTermVec[0]
					test = tempTermVec[1]
					#
					#  build input -- tempR
					#
					crsjSet = np.array([self.itemVec.index(crsCode) for crsCode,_ in train])
					crsjGrd = np.array([grd for _,grd in train])
					tempCtR = len(crsjSet)
					tempR = R[crsjSet]
					tempR = np.array([tempR[i] * crsjGrd[i] for i in range(len(train))])
					tempR = tempR.sum(0)
					tempR = tempR/tempCtR
					#
					#  go through test samples
					#
					for crsCode, testGrd in test:
						crs = self.itemVec.index(crsCode)
						sum1=testGrd-np.dot(tempR,Q[crs])-paraDict['isbs']*bs[std]-paraDict['isbc']*bc[crs]
						sum1 = (float)(sum1)/(float)(self.lf)
						#
						#  Build Deep input
						#
						coCrs = np.zeros(K)
						coCrsSum = np.zeros(K)
						for coCrsCode, coCrsGrd in test:
							if coCrsCode == crsCode:
								continue
							coCrsCodeLoc = self.itemVec.index(coCrsCode)
							coCrs += abs(Q[coCrsCodeLoc] - Q[crs])
							coCrsSum += Q[coCrsCodeLoc]
						coCrs = Variable(torch.from_numpy(coCrs))
						coCrs = coCrs.float()
						coCrsSum = Variable(torch.from_numpy(coCrsSum))
						coCrsSum = coCrsSum.float()
						curCrs = Variable(torch.from_numpy(Q[crs]))
						curCrs = curCrs.float()
						if self.args.coCrsSum == 1:
							coCrs = torch.cat((coCrs,coCrsSum))
						FFInputCrs.append(curCrs)
						FFInputCoCrs.append(coCrs)
						targetGrd.append(sum1)
						#
						#  train CK
						#
						leaveDMout = deepOutput[deepOutputCt] if xx>0 else 0
						deepOutputCt = deepOutputCt+1 if xx>0 else 0
						sum1=testGrd-np.dot(tempR,Q[crs])-paraDict['isbs']*bs[std]-paraDict['isbc']*bc[crs]-self.lf*leaveDMout
						tempQ = Q[crs]-paraDict['lr']*((-1)*tempR*sum1+paraDict['l2']*Q[crs]+paraDict['l1'])
						for index_j in range(len(train)):
							crsj = crsjSet[index_j]
							trainGrd = crsjGrd[index_j]
							R[crsj] = R[crsj] - paraDict['lr']*(((-1) * sum1 * Q[crs] * trainGrd/tempCtR) + paraDict['l2']*R[crsj]+paraDict['l1'])
							#R[crsj] = [x if x >= 0 else 0 for x in R[crsj]]
						#Q[crs] = [x if x >= 0 else 0 for x in tempQ]
						bs[std] = bs[std]-paraDict['lr_bias']*((-1)*paraDict['isbs']*sum1+paraDict['l2']*bs[std]+paraDict['l1'])
						bc[crs] = bc[crs]-paraDict['lr_bias']*((-1)*paraDict['isbc']*sum1+paraDict['l2']*bc[crs]+paraDict['l1'])
						#
						#  calculate training error
						#
						tempR = R[crsjSet]
						tempR = np.array([tempR[i] * crsjGrd[i] for i in range(len(train))])
						tempR = tempR.sum(0)
						tempR = tempR/tempCtR
						predGrd = np.dot(tempR,Q[crs]) + paraDict['isbs']*bs[std] + paraDict['isbc']*bc[crs] + leaveDMout
						predGrdVec.append(predGrd)
						trueGrdVec.append(testGrd)
			#
			# train deep model
			#
			lenGrd = len(trueGrdVec)
			iterBatch = int(lenGrd / paraDict['FFbatchSize'])
			lastIterBatchSize = lenGrd % paraDict['FFbatchSize']
			FFInputCrs = np.array([a.data.numpy() for a in FFInputCrs])
			FFInputCoCrs = np.array([a.data.numpy() for a in FFInputCoCrs])
			FFInputCrs = Variable(torch.from_numpy(FFInputCrs))
			FFInputCoCrs = Variable(torch.from_numpy(FFInputCoCrs))
			targetGrd = Variable(torch.from_numpy(np.array(targetGrd, dtype=np.float32)))
			deepOutput = []
			for x in range(iterBatch):
				deepInput = FFInputCoCrs[x*paraDict['FFbatchSize']:(x+1)*paraDict['FFbatchSize']]
				predGrd = self.model(deepInput)
				loss =  self.criterion(predGrd, targetGrd[x*paraDict['FFbatchSize']:(x+1)*paraDict['FFbatchSize']])
				self.optimizer.zero_grad()
				loss.backward(retain_graph=True)
				self.optimizer.step()
				predGrd = predGrd.view(1,-1)
				predGrd = predGrd.data.numpy().tolist()[0]
				deepOutput += predGrd
			if lastIterBatchSize>0:
				deepInput = FFInputCoCrs[(x+1)*paraDict['FFbatchSize']:]
				predGrd = self.model(deepInput)
				loss =  self.criterion(predGrd, targetGrd[(x+1)*paraDict['FFbatchSize']:])
				self.optimizer.zero_grad()
				loss.backward(retain_graph=True)
				self.optimizer.step()
				predGrd = predGrd.view(1,-1)
				predGrd = predGrd.data.numpy().tolist()[0]
				deepOutput += predGrd
			deepOutputCt = 0
			#
			predGrdVec, trueGrdVec = np.array(predGrdVec), np.array(trueGrdVec)
			testrmse, testmae, testpct0, testpct1, testpct2,testpredGrdGrp_crs, testpredGrdGrp_mjr = self.testCKCC(R,Q,bs,bc)
			#
			trainrmse, trainmae, trainpct0, trainpct1, trainpct2 = self.testRMSE(predGrdVec, trueGrdVec)
			print(' ACK - ',xx,' time -- %.6f   '%(time.time()-oldtime),'train --  mae -- %.6f'%(trainmae), ' ///   test --  mae -- %.6f'%(testmae), '  pct0 -- %.6f'%(testpct0),'  pct1 -- %.6f'%(testpct1),'  pct2 -- %.6f'%(testpct2))
			maeVec.append(testmae)
			pct0Vec.append(testpct0)
			pct1Vec.append(testpct1)
			pct2Vec.append(testpct2)
			testpredGrdGrp_crs_vec.append(testpredGrdGrp_crs)
			testpredGrdGrp_mjr_vec.append(testpredGrdGrp_mjr)
			if testmae >= oldmae:
				if maeCt < 3:
					maeCt+=1
					oldmae = testmae
				else:
					break
			else:
				oldmae = testmae
				maeCt=0
		minIndex = maeVec.index(min(maeVec))
		testpredGrdGrp_crs = testpredGrdGrp_crs_vec[minIndex]
		testpredGrdGrp_mjr = testpredGrdGrp_mjr_vec[minIndex]
		print(minIndex)
		print('CKCC  %.3f  &  %.3f  &  %.3f  &  %.3f'%(maeVec[minIndex],pct0Vec[minIndex],pct1Vec[minIndex],pct2Vec[minIndex]))
		print('CKCC  %.3f  &  %.3f  &  %.3f  &  %.3f'%(maeVec[minIndex],pct0Vec[minIndex],pct1Vec[minIndex],pct2Vec[minIndex]),file=open(self.args.logFile, "a"))
		print('\n')
		print('testpredGrdGrp_crs - ')
		print('testpredGrdGrp_crs - ',file=open(self.args.logFile, "a"))
		self.printDict(testpredGrdGrp_crs)
		print('testpredGrdGrp_mjr - ')
		print('testpredGrdGrp_mjr - ',file=open(self.args.logFile, "a"))
		self.printDict(testpredGrdGrp_mjr)

	def printDict(self,inputDict):
		keyVec = list(inputDict.keys())
		keyVec.sort()
		for tempKey in keyVec:
			print('%d  &  %.3f  &  %.3f  &  %.3f  &  %.3f'%(tempKey, inputDict[tempKey][0][1],inputDict[tempKey][0][2],inputDict[tempKey][0][3],inputDict[tempKey][0][4]))
			print('%d  &  %.3f  &  %.3f  &  %.3f  &  %.3f'%(tempKey, inputDict[tempKey][0][1],inputDict[tempKey][0][2],inputDict[tempKey][0][3],inputDict[tempKey][0][4]),file=open(self.args.logFile, "a"))

	def testCKCC(self,R,Q,bs,bc):
		model = self.model
		testSet = self.testSet
		paraDict = self.paraDict
		testpredGrdVec, testtrueGrdVec = [], []
		testpredGrdGrp_crs = collections.defaultdict(list)
		testpredGrdGrp_mjr = collections.defaultdict(list)
		testtrueGrdGrp_crs = collections.defaultdict(list)
		testtrueGrdGrp_mjr = collections.defaultdict(list)
		for std in list(testSet.keys()):
			tempStdVec = testSet[std]
			std = float(std)
			std = self.userVec.index(std)
			for tempTerm in list(tempStdVec.keys()):
				tempTermVec = tempStdVec[tempTerm]
				train = tempTermVec[0]
				test = tempTermVec[1]
				#
				#  count courses
				#
				numCrs = len(test)
				#
				#  count majors
				#
				crsCodeVec = [a[0] for a in test]
				digitStartVec = [re.search("\d", x).start() for x in crsCodeVec]
				crsMajorVec = [crsCodeVec[i][:digitStartVec[i]] for i in range(len(crsCodeVec))]
				crsMajorVec = list(set(crsMajorVec))
				numMjr = len(crsMajorVec)
				#
				#  build input -- tempR
				#
				crsjSet = np.array([self.itemVec.index(crsCode) for crsCode,_ in train])
				crsjGrd = np.array([grd for _,grd in train])
				tempCtR = len(crsjSet)
				tempR = R[crsjSet]
				tempR = np.array([tempR[i] * crsjGrd[i] for i in range(len(train))])
				tempR = tempR.sum(0)
				tempR = tempR/tempCtR
				for crsCode, testGrd in test:
					crs = self.itemVec.index(crsCode)
					#
					#  Build Deep input
					#
					coCrs = np.zeros(paraDict['K'])
					coCrsSum = np.zeros(paraDict['K'])
					for coCrsCode, coCrsGrd in test:
						if coCrsCode == crsCode:
							continue
						coCrsCodeLoc = self.itemVec.index(coCrsCode)
						coCrs += abs(Q[coCrsCodeLoc] - Q[crs])
						coCrsSum += Q[coCrsCodeLoc]
					coCrs = Variable(torch.from_numpy(coCrs))
					coCrs = coCrs.float()
					coCrsSum = Variable(torch.from_numpy(coCrsSum))
					coCrsSum = coCrsSum.float()
					curCrs = Variable(torch.from_numpy(Q[crs]))
					curCrs = curCrs.float()
					if self.args.coCrsSum == 1:
						coCrs = torch.cat((coCrs,coCrsSum))
					#
					#  test
					#
					predGrd = model(coCrs)
					predGrd = predGrd.data.numpy()[0]
					predGrd_1 = predGrd
					predGrd = np.dot(tempR,Q[crs]) + paraDict['isbs']*bs[std] + paraDict['isbc']*bc[crs] + self.lf*predGrd
					print(predGrd_1, ' ',predGrd)
					testpredGrdVec.append(predGrd)
					testtrueGrdVec.append(testGrd)
					testpredGrdGrp_crs[numCrs].append(predGrd)
					testpredGrdGrp_mjr[numMjr].append(predGrd)
					testtrueGrdGrp_crs[numCrs].append(testGrd)
					testtrueGrdGrp_mjr[numMjr].append(testGrd)
		for tempKey in list(testpredGrdGrp_crs.keys()):
			testpredGrdGrp_crs[tempKey] = [self.testRMSE(np.array(testpredGrdGrp_crs[tempKey]), \
														 np.array(testtrueGrdGrp_crs[tempKey]))]
		for tempKey in list(testpredGrdGrp_mjr.keys()):
			testpredGrdGrp_mjr[tempKey] = [self.testRMSE(np.array(testpredGrdGrp_mjr[tempKey]), \
														 np.array(testtrueGrdGrp_mjr[tempKey]))]
		rmse, mae, pct0, pct1, pct2 = self.testRMSE(np.array(testpredGrdVec), np.array(testtrueGrdVec))
		return rmse, mae, pct0, pct1, pct2, testpredGrdGrp_crs, testpredGrdGrp_mjr

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
