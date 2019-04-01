import pandas as pd
import numpy as np
import math
import time
import json
import collections
import re

class RMSE_ALE:
	def __init__(self, dataset, args):
		print('In class RMSE_ALE')
		self.dataset = dataset
		self.args = args
		self.lstdTerm = 12
		oldtime = time.time()
		if self.args.ifRead == 1:
			self.userVec, self.itemVec, self.instrVec, self.trainingSet, self.testSet = self.dataset.read_generate_vector_set_ale(self.args.userFile, \
																														self.args.itemFile, \
																														self.args.instrFile, \
																														self.args.trainFile, \
																														self.args.testFile)
		else:
			self.userVec, self.itemVec, self.instrVec, self.trainingSet, self.testSet = self.dataset.generate_vector_set_ale()
			self.dataset.write_generate_vector_set_ale(self.args.userFile, \
															self.args.itemFile, \
															self.args.instrFile, \
															self.args.trainFile, \
															self.args.testFile, \
															self.userVec, \
															self.itemVec, \
															self.instrVec, \
															self.trainingSet, \
															self.testSet)
		print("Prepare data: ",time.time()-oldtime)
		print("\n\n")
		with open(self.args.paraDictPath) as json_file:
			self.paraDict = json.load(json_file)


	def train(self):
		paraDict = self.paraDict
		trainingSet, testSet = self.trainingSet, self.testSet
		N, M, L = len(self.userVec), len(self.itemVec), len(self.instrVec)
		K = paraDict['K']

		R = np.random.rand(M,K) # cumulative knowledge
		P = np.random.rand(N,K) # global
		Q = np.random.rand(M,K)
		S = np.random.rand(L,K)
		T = np.random.rand(self.lstdTerm,K)

		bs = np.zeros(N)
		bc = np.zeros(M)
		bi = np.zeros(L)

		stdTermStart = np.zeros(self.lstdTerm)

		oldmae, oldrmse = 100, 100
		maeCt = 0
		maeVec, pct0Vec, pct1Vec, pct2Vec = [],[],[],[]
		testpredGrdGrp_crs_vec = []
		testpredGrdGrp_mjr_vec = []

		for x in range(paraDict['maxIter']):
			oldtime = time.time()
			predGrdVec, trueGrdVec = [], []
			FFInputCrs,FFInputCoCrs,targetGrd = [],[],[]
			for std in list(trainingSet.keys()):
				tempStdVec = trainingSet[std]
				std = float(std)
				i = self.userVec.index(std)
				enrollTerms = list(tempStdVec.keys())
				for tempTermID in range(len(enrollTerms)):
					tempTerm = enrollTerms[tempTermID]
					tempTermVec = tempStdVec[tempTerm]
					train = tempTermVec[0]
					test = tempTermVec[1]
					#
					#  build input -- tempR
					#
					crsjSet = np.array([self.itemVec.index(crsCode) for crsCode,_,_ in train])
					crsjGrd = np.array([grd for _,_,grd in train])
					tempCtR = len(crsjSet)
					tempR = R[crsjSet]
					tempR = np.array([tempR[ii] * crsjGrd[ii] for ii in range(len(train))])
					tempR = tempR.sum(0)
					sum1 = tempR/tempCtR
					#
					for crsCode, insCode, testGrd in test:
						j = self.itemVec.index(crsCode)
						l = self.instrVec.index(insCode)
						truthGrade = testGrd+0.0001
						k = tempTermID
						stdTermStart[k] = 1
						tempSum = truthGrade - paraDict['w1']*np.dot((paraDict['isT']*T[k]+sum1),(Q[j]+paraDict['isS']*S[l]))-paraDict['w2']*np.dot(P[i],Q[j])-paraDict['isbs']*bs[i]-paraDict['isbc']*bc[j]
						tempQ = Q[j]-paraDict['lr']*((-1)*(paraDict['w1']*(sum1+paraDict['isT']*T[k])+paraDict['w2']*P[i])*tempSum+paraDict['l2']*Q[j]+paraDict['l1'])
						tempS = S[l]-paraDict['lr']*((-1)*paraDict['w1']*paraDict['isS']*(sum1+paraDict['isT']*T[k])*tempSum+paraDict['l2']*S[l]+paraDict['l1'])
						tempT = T[k]-paraDict['lr']*((-1)*paraDict['w1']*paraDict['isT']*(Q[j]+paraDict['isS']*S[l])*tempSum+paraDict['l2']*T[k]+paraDict['l1'])
						tempP = P[i]-paraDict['lr']*((-1)*paraDict['w2']*Q[j]*tempSum+paraDict['l2-r']*P[i]+paraDict['l1-r'])
						for index_j in range(len(train)):
							crsj = crsjSet[index_j]
							trainGrd = crsjGrd[index_j]
							R[crsj] = R[crsj] - paraDict['lr']*((-1) * tempSum * paraDict['isP'] * paraDict['w1']*(Q[j]+paraDict['isS']*S[l]) * trainGrd/tempCtR + paraDict['l2']*R[crsj]+paraDict['l1'])
						bs[i] = bs[i]-paraDict['lr_bias']*((-1)*paraDict['isbs']*tempSum+paraDict['l2']*bs[i]+paraDict['l1'])
						bc[j] = bc[j]-paraDict['lr_bias']*((-1)*paraDict['isbc']*tempSum+paraDict['l2']*bc[j]+paraDict['l1'])
						Q[j] = tempQ
						P[i] = tempP
						S[l], T[k] = tempS, tempT
			mae, rmse, pct0, pct1, pct2 = self.testBSCKRM(stdTermStart, P, Q, S, T, R,bs,bc)
			#if mae>oldmae:
			#	break
			print("ALE  %d th iter:    rmse & mae   pct0 & pct1 & pct2:        %.3f  &  %.3f  &      %.3f  &  %.3f  &  %.3f"%(x+1,rmse,mae, pct0, pct1, pct2))
			oldmae, oldrmse = mae, rmse
		mae, rmse, pct0, pct1, pct2 = self.testBSCKRM(stdTermStart, P, Q, S, T, R,bs,bc)
		print("ALE \n\nrmse & mae    pct0 & pct1 & pct2:	%.3f	&	%.3f		&	%.3f	&	%.3f	&	%.3f\n\n"%(rmse,mae, pct0, pct1, pct2))

	def testBSCKRM(self,stdTermStart, P, Q, S, T, R,bs,bc):
		paraDict = self.paraDict
		trainingSet, testSet = self.trainingSet, self.testSet
		N, M, L = len(self.userVec), len(self.itemVec), len(self.instrVec)
		testpredGrdVec, testtrueGrdVec = [], []
		trueG, predG = [],[]
		for std in list(testSet.keys()):
			tempStdVec = testSet[std]
			std = float(std)
			curStdInd = self.userVec.index(std)
			tempTermID = len(trainingSet[std])
			for tempTerm in list(tempStdVec.keys()):
				tempTermVec = tempStdVec[tempTerm]
				train = tempTermVec[0]
				test = tempTermVec[1]
				#
				#  count courses
				#
				numCrs = len(test)
				#
				#  build input -- tempR
				#
				crsjSet = np.array([self.itemVec.index(crsCode) for crsCode,_,_ in train])
				crsjGrd = np.array([grd for _,_,grd in train])
				tempCtR = len(crsjSet)
				tempR = R[crsjSet]
				tempR = np.array([tempR[ii] * crsjGrd[ii] for ii in range(len(train))])
				tempR = tempR.sum(0)
				sum1 = tempR/tempCtR
				for crsCode, insCode, testGrd in test:
					truthGrade = testGrd+0.0001
					k = tempTermID
					stdTermStart[k] = 1
					curCrsInd = self.itemVec.index(crsCode)
					curInstrInd = self.instrVec.index(insCode)
					stdTerm = tempTermID
					#
					if stdTermStart[stdTerm] == 0:
						continue
					partLocal = paraDict['w1']*np.dot((stdTermStart[stdTerm]*paraDict['isT']*T[stdTerm]+sum1),(Q[curCrsInd]+paraDict['isS']*S[curInstrInd]))
					partGlobal = paraDict['w2']*np.dot(P[curStdInd],Q[curCrsInd])
					partKnowledge=paraDict['w1']*np.dot(sum1,Q[curCrsInd])
					predGrade = partLocal+partGlobal+paraDict['isbs']*bs[curStdInd]+paraDict['isbc']*bc[curCrsInd]
					if predGrade>4.0001:
						predGrade=4.0001
					if predGrade<0.0001:
						predGrade=0.0001
					trueG.append(testGrd)
					predG.append(predGrade)
		trueG = np.asarray(trueG)
		predG = np.asarray(predG)
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
