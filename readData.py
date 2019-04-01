import pandas as pd
import numpy as np
import re
import math
import datetime
import matplotlib.pyplot as plt
import collections
import json
import codecs

#
#  args:
#	- dataPath
#	- majorVec
#	- trainingTerms
#	- testTerms
#	- model
#	- paraDictPath
#	- islinear
#

class Dataset:
	def __init__(self, args):
		self.args = args
		self.majorVec = self.args.majorVec
		self.dataset = pd.read_csv(self.args.dataPath)
		self.dataDict = collections.defaultdict(list)
		with open(self.args.paraDictPath) as json_file:
			self.paraDict = json.load(json_file)

	def prepareDataDict(self):
		L = len(self.dataset)
		for i in range(L):
			stdCode = self.dataset['stdID'][i]
			termNumber = str(self.dataset['TERMBNR'][i])
			termMajor = self.dataset['TMAJOR'][i]
			crsCode = self.dataset['crsID'][i]
			insCode = self.dataset['instrID'][i]
			grd = self.dataset['grade'][i]
			if termMajor in self.majorVec:
				if self.dataDict[stdCode] == []:
					self.dataDict[stdCode] = collections.defaultdict(list)
				self.dataDict[stdCode][termNumber].append([crsCode,insCode,grd])

	def generate_vector_set_mf(self):
		ctsample = 0
		self.prepareDataDict()
		userVec, itemVec = [],[]
		trainingSet, testSet = [], []
		trainCt, testCt = 0,0
		for tempStd in list(self.dataDict.keys()):
			tempStdVec = self.dataDict[tempStd]
			for tempTerm in list(tempStdVec.keys()):
				if tempTerm in self.args.trainingTerms:
					tempTermVec = tempStdVec[tempTerm]
					tempLen = len(tempTermVec)
					for j in range(tempLen):
						crsCode = tempTermVec[j][0]
						grd = tempTermVec[j][2]
						trainingSet.append([tempStd,crsCode,grd])
						userVec.append(tempStd)
						ctsample += 1
		userVec = list(set(userVec))
		itemVec = list(set(itemVec))
		for tempStd in list(self.dataDict.keys()):
			if tempStd not in userVec:
				continue
			tempStdVec = self.dataDict[tempStd]
			for tempTerm in list(tempStdVec.keys()):
				if tempTerm in self.args.testTerms:
					tempTermVec = tempStdVec[tempTerm]
					tempLen = len(tempTermVec)
					for j in range(tempLen):
						crsCode = tempTermVec[j][0]
						grd = tempTermVec[j][2]
						if crsCode not in itemVec:
							continue
						testSet.append([tempStd,crsCode,grd])
						ctsample+=1
		print(ctsample)
		print(len(userVec))
		print(len(itemVec))
		return userVec, itemVec, trainingSet, testSet

	def write_generate_vector_set_mf(self,userFile,itemFile,trainFile,testFile,userVec, itemVec, trainingSet, testSet):
		np.savetxt(userFile, userVec, delimiter=",",fmt="%s")
		np.savetxt(itemFile, itemVec, delimiter=",",fmt="%s")
		np.savetxt(trainFile, trainingSet, delimiter=",",fmt="%s")
		np.savetxt(testFile, testSet, delimiter=",",fmt="%s")

	def read_generate_vector_set_mf(self,userFile,itemFile,trainFile,testFile):
		userVec = pd.read_csv(userFile,header=None)
		userVec = userVec.values.tolist()
		userVec = [a[0] for a in userVec]
		#
		itemVec = pd.read_csv(itemFile,header=None)
		itemVec = itemVec.values.tolist()
		itemVec = [a[0] for a in itemVec]
		#
		trainingSet = pd.read_csv(trainFile,header=None)
		testSet = pd.read_csv(testFile,header=None)
		return userVec, itemVec, np.array(trainingSet), np.array(testSet)



	def generate_vector_set_temporal(self):
		self.prepareDataDict()
		userVec, itemVec = [],[]
		coursePrev, courseCurr = [],[]
		trainingSet = collections.defaultdict(list)
		testSet = collections.defaultdict(list)
		for tempStd in list(self.dataDict.keys()):
			tempStdVec = self.dataDict[tempStd]
			# cold start for student
			if len(tempStdVec) < 2:
				continue
			for tempTerm in self.args.trainingTerms:
				if tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					tempCurVec = [[crsCode, grd] for crsCode,_,grd in tempTermVec]
					if len(tempCurVec) == 1:
						continue
					userVec.append(tempStd)
					itemVec += [crsCode for crsCode,_,_ in tempTermVec]
					if trainingSet[tempStd] == []:
						trainingSet[tempStd] = collections.defaultdict(list)
					trainingSet[tempStd][tempTerm] = tempCurVec[:]
				userVec = list(set(userVec))
				itemVec = list(set(itemVec))

		for tempStd in list(self.dataDict.keys()):
			if tempStd not in userVec:
				continue
			tempStdVec = self.dataDict[tempStd]
			for tempTerm in self.args.testTerms:
				if tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					tempCurVec = [[crsCode, grd] for crsCode,_,grd in tempTermVec if crsCode in itemVec]
					if len(tempCurVec) <= 1:
						continue
					if testSet[tempStd] == []:
						testSet[tempStd] = collections.defaultdict(list)
					testSet[tempStd][tempTerm] = tempCurVec[:]
		# delete keys with []
		for tempKey in list(trainingSet.keys()):
			if trainingSet[tempKey] == []:
				del trainingSet[tempKey]
		for tempKey in list(testSet.keys()):
			if tempKey not in list(trainingSet.keys()):
				del testSet[tempKey]
		return userVec, itemVec, trainingSet, testSet

	def write_generate_vector_set_temporal(self,userFile,itemFile,trainFile,testFile,userVec, itemVec, trainingSet, testSet):
		np.savetxt(userFile, userVec, delimiter=",",fmt="%s")
		np.savetxt(itemFile, itemVec, delimiter=",",fmt="%s")
		#
		fh = codecs.open(trainFile,"w")
		saveJson = json.dumps(trainingSet)
		fh.write(saveJson)
		fh.close()
		#
		fh = codecs.open(testFile,"w")
		saveJson = json.dumps(testSet)
		fh.write(saveJson)
		fh.close()

	def read_generate_vector_set_temporal(self,userFile,itemFile,trainFile,testFile):
		userVec = pd.read_csv(userFile,header=None)
		userVec = userVec.values.tolist()
		userVec = [a[0] for a in userVec]
		#
		itemVec = pd.read_csv(itemFile,header=None)
		itemVec = itemVec.values.tolist()
		itemVec = [a[0] for a in itemVec]
		#
		with open(trainFile) as json_file:
			trainingSet = json.load(json_file)
		with open(testFile) as json_file:
			testSet = json.load(json_file)
		return userVec, itemVec, trainingSet, testSet




	def generate_vector_set_ncf(self):
		self.prepareDataDict()
		userVec, itemVec = [],[]
		coursePrev, courseCurr = [],[]
		instrVec = []
		trainingSet = collections.defaultdict(list)
		testSet = collections.defaultdict(list)
		for tempStd in list(self.dataDict.keys()):
			tempStdVec = self.dataDict[tempStd]
			# cold start for student
			if len(tempStdVec) < 2:
				continue
			for tempTerm in self.args.trainingTerms:
				if tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					tempCurVec = tempTermVec
					if len(tempCurVec) == 1:
						continue
					userVec.append(tempStd)
					itemVec += [crsCode for crsCode,_,_ in tempTermVec]
					instrVec += [insCode for _,insCode,_ in tempTermVec]
					if trainingSet[tempStd] == []:
						trainingSet[tempStd] = collections.defaultdict(list)
					trainingSet[tempStd][tempTerm] = tempCurVec[:]
				userVec = list(set(userVec))
				itemVec = list(set(itemVec))
				instrVec = list(set(instrVec))

		for tempStd in list(self.dataDict.keys()):
			if tempStd not in userVec:
				continue
			tempStdVec = self.dataDict[tempStd]
			for tempTerm in self.args.testTerms:
				if tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					tempCurVec = [[crsCode, insCode, grd] for crsCode,insCode,grd in tempTermVec if crsCode in itemVec and insCode in instrVec]
					if len(tempCurVec) <= 1:
						continue
					if testSet[tempStd] == []:
						testSet[tempStd] = collections.defaultdict(list)
					testSet[tempStd][tempTerm] = tempCurVec[:]
		# delete keys with []
		for tempKey in list(trainingSet.keys()):
			if trainingSet[tempKey] == []:
				del trainingSet[tempKey]
		for tempKey in list(testSet.keys()):
			if tempKey not in list(trainingSet.keys()):
				del testSet[tempKey]
		return userVec, itemVec, instrVec, trainingSet, testSet

	def write_generate_vector_set_ncf(self,userFile,itemFile,instrFile,trainFile,testFile,userVec, itemVec, instrVec, trainingSet, testSet):
		np.savetxt(userFile, userVec, delimiter=",",fmt="%s")
		np.savetxt(itemFile, itemVec, delimiter=",",fmt="%s")
		np.savetxt(instrFile, instrVec, delimiter=",",fmt="%s")
		#
		fh = codecs.open(trainFile,"w")
		saveJson = json.dumps(trainingSet)
		fh.write(saveJson)
		fh.close()
		#
		fh = codecs.open(testFile,"w")
		saveJson = json.dumps(testSet)
		fh.write(saveJson)
		fh.close()

	def read_generate_vector_set_ncf(self,userFile,itemFile,instrFile,trainFile,testFile):
		userVec = pd.read_csv(userFile,header=None)
		userVec = userVec.values.tolist()
		userVec = [a[0] for a in userVec]
		#
		itemVec = pd.read_csv(itemFile,header=None)
		itemVec = itemVec.values.tolist()
		itemVec = [a[0] for a in itemVec]
		#
		instrVec = pd.read_csv(instrFile,header=None)
		instrVec = instrVec.values.tolist()
		instrVec = [a[0] for a in instrVec]
		#
		with open(trainFile) as json_file:
			trainingSet = json.load(json_file)
		with open(testFile) as json_file:
			testSet = json.load(json_file)
		return userVec, itemVec, instrVec, trainingSet, testSet


	def generate_vector_set_ale(self):
		self.prepareDataDict()
		paraDict = self.paraDict
		userVec, itemVec = [],[]
		instrVec = []
		coursePrev, courseCurr = [],[]
		trainingSet = collections.defaultdict(list)
		testSet = collections.defaultdict(list)
		for tempStd in list(self.dataDict.keys()):
			tempStdVec = self.dataDict[tempStd]
			# cold start for student
			if len(tempStdVec) < 3:
				continue
			tempHistVec = []
			tempLastVec = []
			firstMark = 0
			for tempTerm in self.args.trainingTerms:
				if tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					if firstMark == 0:
						firstMark = 1
						tempLastVec = tempTermVec
					else:
						tempHistVec += tempLastVec
						tempHistVec = [[crsCode, insCode, math.exp(-paraDict['decayRate']) * value] for crsCode,insCode,value in tempHistVec]
						tempLastVec = tempTermVec
						if len(tempLastVec) == 1:
							continue
						courseCurr += [crsCode for crsCode,_,_ in tempTermVec]
						instrVec += [insCode for _,insCode,_ in tempTermVec]
						coursePrev += [crsCode for crsCode,_,_ in tempHistVec]
						if trainingSet[tempStd] == []:
							trainingSet[tempStd] = collections.defaultdict(list)
						trainingSet[tempStd][tempTerm].append(tempHistVec[:])
						trainingSet[tempStd][tempTerm].append(tempLastVec[:])
						userVec.append(tempStd)
				courseCurr = list(set(courseCurr))
				coursePrev = list(set(coursePrev))
				instrVec = list(set(instrVec))
			# no training samples for this student
			if trainingSet[tempStd] == []:
				continue
			for tempTerm in self.args.testTerms:
				if tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					tempHistVec += tempLastVec
					tempHistVec = [[crsCode, insCode, math.exp(-paraDict['decayRate']) * value] for crsCode,insCode,value in tempHistVec]
					tempLastVec = tempTermVec
					if len(tempLastVec) == 1:
						continue
					if testSet[tempStd] == []:
						testSet[tempStd] = collections.defaultdict(list)
					testSet[tempStd][tempTerm].append(tempHistVec[:])
					testSet[tempStd][tempTerm].append(tempLastVec[:])
		courseCurr = list(set(courseCurr))
		coursePrev = list(set(coursePrev))
		userVec = list(set(userVec))
		instrVec = list(set(instrVec))
		#
		# remove cold start for courses
		#
		for tempStd in list(testSet.keys()):
			tempStdVec = testSet[tempStd]
			for tempTerm in list(tempStdVec.keys()):
				tempTrain = tempStdVec[tempTerm][0]
				tempTest = tempStdVec[tempTerm][1]
				tempTrain = [[crsCode,insCode, grd] for crsCode,insCode,grd in tempTrain if crsCode in coursePrev]
				tempTest = [[crsCode, insCode, grd] for crsCode,insCode,grd in tempTest if crsCode in courseCurr and insCode in instrVec]
				if tempTrain == [] or tempTest == [] or len(tempTest) == 1 or tempStd not in userVec:
					del testSet[tempStd][tempTerm]
				else:
					testSet[tempStd][tempTerm][0]=(tempTrain[:])
					testSet[tempStd][tempTerm][1]=(tempTest[:])
		itemVec = courseCurr + coursePrev
		itemVec = list(set(itemVec))
		# delete keys with []
		for tempKey in list(trainingSet.keys()):
			if trainingSet[tempKey] == []:
				del trainingSet[tempKey]
		for tempKey in list(testSet.keys()):
			if tempKey not in list(trainingSet.keys()):
				del testSet[tempKey]

		return userVec, itemVec, instrVec, trainingSet, testSet

	def write_generate_vector_set_ale(self,userFile,itemFile,instrFile, trainFile,testFile,userVec, itemVec, instrVec, trainingSet, testSet):
		np.savetxt(userFile, userVec, delimiter=",",fmt="%s")
		np.savetxt(itemFile, itemVec, delimiter=",",fmt="%s")
		np.savetxt(instrFile, instrVec, delimiter=",",fmt="%s")
		#
		fh = codecs.open(trainFile,"w")
		saveJson = json.dumps(trainingSet)
		fh.write(saveJson)
		fh.close()
		#
		fh = codecs.open(testFile,"w")
		saveJson = json.dumps(testSet)
		fh.write(saveJson)
		fh.close()

	def read_generate_vector_set_ale(self,userFile,itemFile,instrFile,trainFile,testFile):
		userVec = pd.read_csv(userFile,header=None)
		userVec = userVec.values.tolist()
		userVec = [a[0] for a in userVec]
		#
		itemVec = pd.read_csv(itemFile,header=None)
		itemVec = itemVec.values.tolist()
		itemVec = [a[0] for a in itemVec]
		#
		instrVec = pd.read_csv(instrFile,header=None)
		instrVec = instrVec.values.tolist()
		instrVec = [a[0] for a in instrVec]
		#
		with open(trainFile) as json_file:
			trainingSet = json.load(json_file)
		with open(testFile) as json_file:
			testSet = json.load(json_file)
		return userVec, itemVec, instrVec, trainingSet, testSet





	def generate_vector_set_ck(self):
		self.prepareDataDict()
		paraDict = self.paraDict
		userVec, itemVec = [],[]
		coursePrev, courseCurr = [],[]
		trainingSet = collections.defaultdict(list)
		testSet = collections.defaultdict(list)
		for tempStd in list(self.dataDict.keys()):
			tempStdVec = self.dataDict[tempStd]
			# cold start for student
			if len(tempStdVec) < 3:
				continue
			tempHistVec = []
			tempLastVec = []
			firstMark = 0
			for tempTerm in self.args.trainingTerms:
				if tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					if firstMark == 0:
						firstMark = 1
						tempLastVec = [[crsCode, grd] for crsCode,_,grd in tempTermVec]
					else:
						tempHistVec += tempLastVec
						tempHistVec = [[crsCode, math.exp(-paraDict['decayRate']) * value] for crsCode,value in tempHistVec]
						tempLastVec = [[crsCode, grd] for crsCode,_,grd in tempTermVec]
						if len(tempLastVec) == 1:
							continue
						courseCurr += [crsCode for crsCode,_,_ in tempTermVec]
						coursePrev += [crsCode for crsCode,_ in tempHistVec]
						if trainingSet[tempStd] == []:
							trainingSet[tempStd] = collections.defaultdict(list)
						trainingSet[tempStd][tempTerm].append(tempHistVec[:])
						trainingSet[tempStd][tempTerm].append(tempLastVec[:])
						userVec.append(tempStd)
				courseCurr = list(set(courseCurr))
				coursePrev = list(set(coursePrev))
			# no training samples for this student
			if trainingSet[tempStd] == []:
				continue
			for tempTerm in self.args.testTerms:
				if tempTerm in list(tempStdVec.keys()):
					tempTermVec = tempStdVec[tempTerm]
					tempHistVec += tempLastVec
					tempHistVec = [[crsCode, math.exp(-paraDict['decayRate']) * value] for crsCode,value in tempHistVec]
					tempLastVec = [[crsCode, grd] for crsCode,_,grd in tempTermVec]
					if len(tempLastVec) == 1:
						continue
					if testSet[tempStd] == []:
						testSet[tempStd] = collections.defaultdict(list)
					testSet[tempStd][tempTerm].append(tempHistVec[:])
					testSet[tempStd][tempTerm].append(tempLastVec[:])
		courseCurr = list(set(courseCurr))
		coursePrev = list(set(coursePrev))
		userVec = list(set(userVec))
		#
		# remove cold start for courses
		#
		for tempStd in list(testSet.keys()):
			tempStdVec = testSet[tempStd]
			for tempTerm in list(tempStdVec.keys()):
				tempTrain = tempStdVec[tempTerm][0]
				tempTest = tempStdVec[tempTerm][1]
				tempTrain = [[crsCode, grd] for crsCode,grd in tempTrain if crsCode in coursePrev]
				tempTest = [[crsCode, grd] for crsCode,grd in tempTest if crsCode in courseCurr]
				if tempTrain == [] or tempTest == [] or len(tempTest) == 1:
					del testSet[tempStd][tempTerm]
				else:
					testSet[tempStd][tempTerm][0]=(tempTrain[:])
					testSet[tempStd][tempTerm][1]=(tempTest[:])
		itemVec = courseCurr + coursePrev
		itemVec = list(set(itemVec))
		# delete keys with []
		for tempKey in list(trainingSet.keys()):
			if trainingSet[tempKey] == []:
				del trainingSet[tempKey]
		for tempKey in list(testSet.keys()):
			if tempKey not in list(trainingSet.keys()):
				del testSet[tempKey]

		return userVec, itemVec, trainingSet, testSet

	def write_generate_vector_set_ck(self,userFile,itemFile,trainFile,testFile,userVec, itemVec, trainingSet, testSet):
		np.savetxt(userFile, userVec, delimiter=",",fmt="%s")
		np.savetxt(itemFile, itemVec, delimiter=",",fmt="%s")
		#
		fh = codecs.open(trainFile,"w")
		saveJson = json.dumps(trainingSet)
		fh.write(saveJson)
		fh.close()
		#
		fh = codecs.open(testFile,"w")
		saveJson = json.dumps(testSet)
		fh.write(saveJson)
		fh.close()

	def read_generate_vector_set_ck(self,userFile,itemFile,trainFile,testFile):
		userVec = pd.read_csv(userFile,header=None)
		userVec = userVec.values.tolist()
		userVec = [a[0] for a in userVec]
		#
		itemVec = pd.read_csv(itemFile,header=None)
		itemVec = itemVec.values.tolist()
		itemVec = [a[0] for a in itemVec]
		#
		with open(trainFile) as json_file:
			trainingSet = json.load(json_file)
		with open(testFile) as json_file:
			testSet = json.load(json_file)
		return userVec, itemVec, trainingSet, testSet

	def apriori_temp(self):
		userVec, itemVec, trainingSet, testSet = self.generate_vector_set_temporal()
		countVec=[]
		countDict=collections.defaultdict(int)
		for tempStd in list(trainingSet.keys()):
			tempStdVec = trainingSet[tempStd]
			for tempTerm in list(tempStdVec.keys()):
				tempTermVec = tempStdVec[tempTerm]
				tempLen = len(tempTermVec)
				tempCrsVec = []
				for j in range(tempLen):
					crsCode = tempTermVec[j][0]
					tempCrsVec.append(crsCode)
				countVec.append(tempCrsVec)
		for tempList in countVec:
			for tempValue in tempList:
				countDict[tempValue] += 1
