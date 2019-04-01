import pandas as pd
import numpy as np
import math
import time
import json
import collections
import re

class RMSE_MF:
	def __init__(self, dataset, args):
		print('In class RMSE_MF')
		self.dataset = dataset
		self.args = args
		oldtime = time.time()
		if self.args.ifRead == 1:
			self.userVec, self.itemVec, self.trainingSet, self.testSet = self.dataset.read_generate_vector_set_temporal(self.args.userFile, \
																														self.args.itemFile, \
																														self.args.trainFile, \
																														self.args.testFile)
		else:
			self.userVec, self.itemVec, self.trainingSet, self.testSet = self.dataset.generate_vector_set_temporal()
			self.dataset.write_generate_vector_set_temporal(self.args.userFile, \
															self.args.itemFile, \
															self.args.trainFile, \
															self.args.testFile, \
															self.userVec, \
															self.itemVec, \
															self.trainingSet, \
															self.testSet)
		print("Prepare data: ",time.time()-oldtime)
		print("\n\n")
		'''
		if self.args.ifRead == 1:
			self.userVec, self.itemVec, self.trainingSet, self.testSet = self.dataset.read_generate_vector_set_mf(self.args.userFile, \
																												  self.args.itemFile, \
																												  self.args.trainFile, \
																												  self.args.testFile)
		else:
			self.userVec, self.itemVec, self.trainingSet, self.testSet = self.dataset.generate_vector_set_mf()
			self.dataset.write_generate_vector_set_mf(self.args.userFile, \
													  self.args.itemFile, \
													  self.args.trainFile, \
													  self.args.testFile, \
													  self.userVec, \
													  self.itemVec, \
													  self.trainingSet, \
													  self.testSet)
		'''
		with open(self.args.paraDictPath) as json_file:
			self.paraDict = json.load(json_file)

	def train(self):
		paraDict = self.paraDict
		trainingSet, testSet = self.trainingSet, self.testSet
		N, M = len(self.userVec), len(self.itemVec)
		K = paraDict['K']
		P = np.random.rand(N,K)
		Q = np.random.rand(M,K)
		bs = np.zeros(N)
		bc = np.zeros(M)

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
				std = self.userVec.index(std)
				for tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					#
					#  go through training samples
					#
					for crsCode, grd in tempTermVec:
						crs = self.itemVec.index(crsCode)
						sum1=grd-np.dot(P[std],Q[crs])-paraDict['isbs']*bs[std]-paraDict['isbc']*bc[crs]
						tempP = P[std]-paraDict['lr']*((-1)*Q[crs]*sum1+paraDict['l2']*P[std]+paraDict['l1'])
						tempQ = Q[crs]-paraDict['lr']*((-1)*P[std]*sum1+paraDict['l2']*Q[crs]+paraDict['l1'])
						P[std], Q[crs] = tempP, tempQ
						bs[std] = bs[std]-paraDict['lr_bias']*((-1)*paraDict['isbs']*sum1+paraDict['l2']*bs[std]+paraDict['l1'])
						bc[crs] = bc[crs]-paraDict['lr_bias']*((-1)*paraDict['isbc']*sum1+paraDict['l2']*bc[crs]+paraDict['l1'])
			rmse, mae, pct0, pct1, pct2, testpredGrdGrp_crs, testpredGrdGrp_mjr = self.testMF(P, Q, bs, bc)
			print("MF -  %d th iter:	rmse & mae   pct0 & pct1 & pct2:		%.6f  &  %.6f		  %.6f  &  %.6f  &  %.6f"%(x+1,rmse,mae, pct0, pct1, pct2))
			maeVec.append(mae)
			pct0Vec.append(pct0)
			pct1Vec.append(pct1)
			pct2Vec.append(pct2)
			testpredGrdGrp_crs_vec.append(testpredGrdGrp_crs)
			testpredGrdGrp_mjr_vec.append(testpredGrdGrp_mjr)
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
		testpredGrdGrp_crs = testpredGrdGrp_crs_vec[minIndex]
		testpredGrdGrp_mjr = testpredGrdGrp_mjr_vec[minIndex]

		if self.args.ifFTF == 1:
			print("student group -- FTF")
		else:
			print("student group -- TR")
		print(self.args.majorVec)
		print(minIndex)
		print('MF  %.3f  &  %.3f  &  %.3f  &  %.3f\n'%(maeVec[minIndex],pct0Vec[minIndex],pct1Vec[minIndex],pct2Vec[minIndex]))
		print('MF  %.3f  &  %.3f  &  %.3f  &  %.3f\n'%(maeVec[minIndex],pct0Vec[minIndex],pct1Vec[minIndex],pct2Vec[minIndex]),file=open(self.args.logFile, "a"))
		print('testpredGrdGrp_crs - ')
		print('testpredGrdGrp_crs - ',file=open(self.args.logFile, "a"))
		self.printDict(testpredGrdGrp_crs)
		print('testpredGrdGrp_mjr - ')
		print('testpredGrdGrp_mjr - ',file=open(self.args.logFile, "a"))
		self.printDict(testpredGrdGrp_mjr)
		print("\n\n\n")

	def printDict(self,inputDict):
		keyVec = list(inputDict.keys())
		keyVec.sort()
		for tempKey in keyVec:
			print('%d  &  %.3f  &  %.3f  &  %.3f  &  %.3f'%(tempKey, inputDict[tempKey][0][1],inputDict[tempKey][0][2],inputDict[tempKey][0][3],inputDict[tempKey][0][4]))
			print('%d  &  %.3f  &  %.3f  &  %.3f  &  %.3f'%(tempKey, inputDict[tempKey][0][1],inputDict[tempKey][0][2],inputDict[tempKey][0][3],inputDict[tempKey][0][4]),file=open(self.args.logFile, "a"))


	def testMF(self,P,Q,bs,bc):
		trainingSet, testSet = self.trainingSet, self.testSet
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
				#
				#  count courses
				#
				numCrs = len(tempTermVec)
				#
				#  count majors
				#
				crsCodeVec = [a[0] for a in tempTermVec]
				digitStartVec = [re.search("\d", x).start() for x in crsCodeVec]
				crsMajorVec = [crsCodeVec[i][:digitStartVec[i]] for i in range(len(crsCodeVec))]
				crsMajorVec = list(set(crsMajorVec))
				numMjr = len(crsMajorVec)
				#
				#  build input -- tempR
				#
				for crsCode, testGrd in tempTermVec:
					crs = self.itemVec.index(crsCode)
					predGrd = np.dot(P[std],Q[crs]) + paraDict['isbs']*bs[std] + paraDict['isbc']*bc[crs]
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


	def original_train(self):
		paraDict = self.paraDict
		trainingSet, testSet = self.trainingSet, self.testSet
		N, M = len(self.userVec), len(self.itemVec)
		K = paraDict['K']
		P = np.random.rand(N,K)
		Q = np.random.rand(M,K)
		bs = np.zeros(N)
		bc = np.zeros(M)

		oldmae, oldrmse = 100, 100
		maeCt = 0
		maeVec, pct0Vec, pct1Vec, pct2Vec = [],[],[],[]

		for x in range(paraDict['maxIter']):
			for l in range(len(trainingSet)):
				std = self.userVec.index(trainingSet[l][0])
				crs = self.itemVec.index(trainingSet[l][1])
				grd = trainingSet[l][2]

				sum1=grd-np.dot(P[std],Q[crs])-paraDict['isbs']*bs[std]-paraDict['isbc']*bc[crs]
				tempP = P[std]-paraDict['lr']*((-1)*Q[crs]*sum1+paraDict['l2']*P[std]+paraDict['l1'])
				tempQ = Q[crs]-paraDict['lr']*((-1)*P[std]*sum1+paraDict['l2']*Q[crs]+paraDict['l1'])
				P[std], Q[crs] = tempP, tempQ
				bs[std] = bs[std]-paraDict['lr_bias']*((-1)*paraDict['isbs']*sum1+paraDict['l2']*bs[std]+paraDict['l1'])
				bc[crs] = bc[crs]-paraDict['lr_bias']*((-1)*paraDict['isbc']*sum1+paraDict['l2']*bc[crs]+paraDict['l1'])

			rmse, mae, pct0, pct1, pct2 = self.original_testMF(P, Q, bs, bc)
			print(" %d th iter:	rmse & mae   pct0 & pct1 & pct2:		%.6f  &  %.6f		  %.6f  &  %.6f  &  %.6f"%(x+1,rmse,mae, pct0, pct1, pct2))
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
		print(minIndex)
		print('MF  %.3f  &  %.3f  &  %.3f  &  %.3f\n'%(maeVec[minIndex],pct0Vec[minIndex],pct1Vec[minIndex],pct2Vec[minIndex]))


	def original_testMF(self, P, Q, bs, bc):
		trainingSet, testSet = self.trainingSet, self.testSet
		paraDict = self.paraDict
		trueGrd=[]
		predGrd=[]
		for l in range(len(testSet)):
			std = self.userVec.index(testSet[l][0])
			crs = self.itemVec.index(testSet[l][1])
			trueGrd.append(testSet[l][2]+0.0001)
			predGrd.append(np.dot(P[std],Q[crs])+paraDict['isbs']*bs[std]+paraDict['isbc']*bc[crs])
		trueGrd = np.asarray(trueGrd)
		predGrd = np.asarray(predGrd)
		return self.testRMSE(predGrd, trueGrd)

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
