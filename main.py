import numpy as np
import pandas as pd
import math
import torch
from torch import nn
from torch.autograd import Variable
import time
import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import codecs
import json

import sys
import argparse

import readData
import MF
import MFCC
import CK
import CKCC
import CKCC_lf
import MFCC_lf
import ALE
import NCF

import caseStudy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath',
        help='Filename of the input dataset.',
        required=True)
    parser.add_argument('--majorVec',
        help='The majors in consideration.',
        required=True)
    parser.add_argument('--trainingTerms',
        help='Terms in training sets.',
        required=True)
    parser.add_argument('--testTerms',
        help='Terms in training sets.',
        required=True)
    parser.add_argument('--model',
        help='Model to run.',
        choices=['MF', 'MFCC', 'CK', 'CKCC', 'CKCC_lf', 'MFCC_lf','caseStudy','ALE','NCF'],
        required=True)
    parser.add_argument('--paraDictPath',
        help='Filename of parameter dictionary.',
        required=True)
    parser.add_argument('--islinear',
        help='Indicator for co-taken courses influence.',
        default=0,
        type=int)
    parser.add_argument('--userFile',
        help='Filename for users.')
    parser.add_argument('--itemFile',
        help='Filename for items.')
    parser.add_argument('--instrFile',
        help='Filename for instructors.')
    parser.add_argument('--trainFile',
        help='Filename for training set.')
    parser.add_argument('--testFile',
        help='Filename for test set.')
    parser.add_argument('--ifRead',
        help='Indicator for read//write data.',
        type=int,
        required=True)
    parser.add_argument('--ifFTF',
        help='If the current student group is FTF.',
        default=1,
        type=int)
    parser.add_argument('--logFile',
        help='Filename for log data.',
        required=True)
    parser.add_argument('--lf',
        help='Weight for f(.) function.',
        default=1,
        type=float)
    parser.add_argument('--coCrsSum',
        help='Input for NN.',
        default=1,
        type=int)

    args = parser.parse_args()
    return args

def train_model(dataset, args):
    if args.model == 'MF':
        model = MF.RMSE_MF(dataset, args)
    elif args.model == 'MFCC':
        model = MFCC.RMSE_MFCC(dataset, args)
    elif args.model == 'CK':
        model = CK.RMSE_CK(dataset, args)
    elif args.model == 'CKCC':
        model = CKCC.RMSE_CKCC(dataset, args)
    elif args.model == 'CKCC_lf':
        model = CKCC_lf.RMSE_CKCC_lf(dataset, args)
    elif args.model == 'MFCC_lf':
        model = MFCC_lf.RMSE_MFCC_lf(dataset, args)
    elif args.model == 'ALE':
        model = ALE.RMSE_ALE(dataset, args)
    elif args.model == 'NCF':
        model = NCF.RMSE_NCF(dataset, args)
    elif args.model == 'caseStudy':
        model = caseStudy.caseStudy(dataset, args, 0, 1)

    model.train()

def printOutArgs(args):
    #logFile  = args.logFile
    #ff = open(logFile, 'a+')
    print('\n\n')
    print('dataPath- ',args.dataPath)
    print('majorVec- ',args.majorVec)
    print('trainingTerms- ',args.trainingTerms)
    print('testTerms- ',args.testTerms)
    print('model- ',args.model)
    print('paraDictPath- ',args.paraDictPath)
    print('islinear- ',args.islinear)
    print('userFile- ',args.userFile)
    print('itemFile- ',args.itemFile)
    print('instrFile- ',args.instrFile)
    print('trainFile- ',args.trainFile)
    print('testFile- ',args.testFile)
    print('ifRead- ',args.ifRead)
    print('ifFTF- ',args.ifFTF)
    print('logFile- ',args.logFile)
    print('lf- ',args.lf)
    print('coCrsSum- ',args.coCrsSum)
    print('\n\n')
    print('\n\n', file=open(args.logFile, "a"))
    print('dataPath- ',args.dataPath, file=open(args.logFile, "a"))
    print('majorVec- ',args.majorVec, file=open(args.logFile, "a"))
    print('trainingTerms- ',args.trainingTerms, file=open(args.logFile, "a"))
    print('testTerms- ',args.testTerms, file=open(args.logFile, "a"))
    print('model- ',args.model, file=open(args.logFile, "a"))
    print('paraDictPath- ',args.paraDictPath, file=open(args.logFile, "a"))
    print('islinear- ',args.islinear, file=open(args.logFile, "a"))
    print('userFile- ',args.userFile, file=open(args.logFile, "a"))
    print('itemFile- ',args.itemFile, file=open(args.logFile, "a"))
    print('instrFile- ',args.instrFile, file=open(args.logFile, "a"))
    print('trainFile- ',args.trainFile, file=open(args.logFile, "a"))
    print('testFile- ',args.testFile, file=open(args.logFile, "a"))
    print('ifRead- ',args.ifRead, file=open(args.logFile, "a"))
    print('ifFTF- ',args.ifFTF, file=open(args.logFile, "a"))
    print('logFile- ',args.logFile, file=open(args.logFile, "a"))
    print('lf- ',args.lf, file=open(args.logFile, "a"))
    print('coCrsSum- ',args.coCrsSum, file=open(args.logFile, "a"))
    print('\n\n', file=open(args.logFile, "a"))


if __name__ == '__main__':
    # args
    args = parse_args()
    print("\n\n", file=open(args.logFile, "a"))
    print(datetime.datetime.now(), file=open(args.logFile, "a"))
    print("\n\n", file=open(args.logFile, "a"))
    args.majorVec = args.majorVec[1:-1].split(',')
    args.trainingTerms = args.trainingTerms[1:-1].split(',')
    args.testTerms = args.testTerms[1:-1].split(',')
    printOutArgs(args)

    # dataset
    d = readData.Dataset(args)

    # train model
    train_model(d, args)
