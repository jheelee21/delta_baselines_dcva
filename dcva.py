# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Sudipan Saha
"""
import os
import sys
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import math
import matplotlib.gridspec as gridspec
import pickle as pickle
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer23
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer8
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer10
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer11
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer5
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer2
from skimage.transform import resize
from skimage import filters
from skimage import morphology
import cv2 as cv
from kmodes.kmodes import KModes
import PIL
import cv2
from scipy.spatial.distance import cdist
import scipy.stats as sistats
from saturateSomePercentile import saturateImage
from options import optionsDCVA

from data_loader import *
from validation_metrics import *
import random

from tqdm import tqdm


##Parsing options
opt = optionsDCVA().parseOptions()
dataPath = opt.dataPath
inputChannels = opt.inputChannels
outputLayerNumbers = np.array(opt.layersToProcess.split(','),dtype=int)
thresholdingStrategy = opt.thresholding
otsuScalingFactor = opt.otsuScalingFactor
objectMinSize = opt.objectMinSize
topPercentSaturationOfImageOk=opt.topPercentSaturationOfImageOk
topPercentToSaturate=opt.topPercentToSaturate
multipleCDBool=opt.multipleCDBool
changeVectorBinarizationStrategy=opt.changeVectorBinarizationStrategy
clusteringStrategy=opt.clusteringStrategy
clusterNumber=opt.clusterNumber
hierarchicalDistanceStrategy=opt.hierarchicalDistanceStrategy


nanVar=float('nan')

#Defining parameters related to the CNN
sizeReductionTable=[nanVar,nanVar,1,nanVar,nanVar,2,nanVar,nanVar,4,nanVar,\
                            4,4,4,4,4,4,4,4,4,
                            nanVar,2,nanVar,nanVar,1,nanVar,nanVar,1,1] 
featurePercentileToDiscardTable=[nanVar,nanVar,90,nanVar,nanVar,90,nanVar,nanVar,95,nanVar,\
                            95,95,95,95,95,95,95,95,95,nanVar,95,nanVar,nanVar,95
                            ,nanVar,nanVar,0,0]
filterNumberTable=[nanVar,nanVar,64,nanVar,nanVar,128,nanVar,nanVar,256,nanVar,\
                            256,256,256,256,256,256,256,256,256,nanVar,128,nanVar,nanVar,64,nanVar,nanVar,1,1]


GPU_NUM = 5
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print ('Current cuda device ', torch.cuda.current_device())

seed = 777

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


fp = "../../../data/delta/xBD_ori/geotiffs/test/"
img_path = fp + "images_256/"
label_path = fp + "targets_256/"
disaster_lst = ["santa-rosa-wildfire", "hurricane-harvey", "palu-tsunami"]
preChangeImage, postChangeImage, damage_label = load_xBD(img_path, label_path, disaster_lst)
cd_label = np.zeros(shape=damage_label.shape)
num_img = preChangeImage.shape[0]


#Pre-change and post-change image normalization
# if topPercentSaturationOfImageOk:
#     preChangeImageNormalized=saturateImage().saturateSomePercentileMultispectral(preChangeImage, topPercentToSaturate)
#     postChangeImageNormalized=saturateImage().saturateSomePercentileMultispectral(postChangeImage, topPercentToSaturate)

imageSizeRow=256
imageSizeCol=256
imageNumberOfChannel=3


#Initilizing net / model (G_B: acts as feature extractor here)
input_nc=imageNumberOfChannel #input number of channels
output_nc=6 #from Potsdam dataset number of classes
ngf=64 # number of gen filters in first conv layer
norm_layer = nn.BatchNorm2d
use_dropout=False


netForFeatureExtractionLayer23=ResnetFeatureExtractor9FeatureFromLayer23(input_nc, output_nc, ngf, norm_layer, use_dropout, 9).to(device)
netForFeatureExtractionLayer11=ResnetFeatureExtractor9FeatureFromLayer11(input_nc, output_nc, ngf, norm_layer, use_dropout, 9).to(device)
netForFeatureExtractionLayer10=ResnetFeatureExtractor9FeatureFromLayer10(input_nc, output_nc, ngf, norm_layer, use_dropout, 9).to(device)
netForFeatureExtractionLayer8=ResnetFeatureExtractor9FeatureFromLayer8(input_nc, output_nc, ngf, norm_layer, use_dropout, 9).to(device)
netForFeatureExtractionLayer5=ResnetFeatureExtractor9FeatureFromLayer5(input_nc, output_nc, ngf, norm_layer, use_dropout, 9).to(device)
netForFeatureExtractionLayer2=ResnetFeatureExtractor9FeatureFromLayer2(input_nc, output_nc, ngf, norm_layer, use_dropout, 9).to(device)


if inputChannels=='RGB':
    state_dict=torch.load('./trainedNet/RGB/trainedModelFinal')
    if imageNumberOfChannel!=3:
        sys.exit('Input images do not have 3 channels while loaded model is for R-G-B input')
elif inputChannels=='RGBNIR':
    state_dict=torch.load('./trainedNet/RGBIR/trainedModelFinal')
    if imageNumberOfChannel!=4:
        sys.exit('Input images do not have 4 channels while loaded model is for R-G-B-NIR input')
else:
    sys.exit('Image channels not valid - valid arguments RGB or RGBNIR')


netForFeatureExtractionLayer23Dict=netForFeatureExtractionLayer23.state_dict()
state_dictForLayer23=state_dict
state_dictForLayer23={k: v for k, v in state_dict.items() if k in netForFeatureExtractionLayer23Dict}

netForFeatureExtractionLayer11Dict=netForFeatureExtractionLayer11.state_dict()
state_dictForLayer11=state_dict
state_dictForLayer11={k: v for k, v in state_dict.items() if k in netForFeatureExtractionLayer11Dict}

netForFeatureExtractionLayer10Dict=netForFeatureExtractionLayer10.state_dict()
state_dictForLayer10=state_dict
state_dictForLayer10={k: v for k, v in state_dict.items() if k in netForFeatureExtractionLayer10Dict}

netForFeatureExtractionLayer8Dict=netForFeatureExtractionLayer8.state_dict()
state_dictForLayer8=state_dict
state_dictForLayer8={k: v for k, v in state_dict.items() if k in netForFeatureExtractionLayer8Dict}

netForFeatureExtractionLayer5Dict=netForFeatureExtractionLayer5.state_dict()
state_dictForLayer5=state_dict
state_dictForLayer5={k: v for k, v in state_dict.items() if k in netForFeatureExtractionLayer5Dict}

netForFeatureExtractionLayer2Dict=netForFeatureExtractionLayer2.state_dict()
state_dictForLayer2=state_dict
state_dictForLayer2={k: v for k, v in state_dict.items() if k in netForFeatureExtractionLayer2Dict}

netForFeatureExtractionLayer23.load_state_dict(state_dictForLayer23)
netForFeatureExtractionLayer11.load_state_dict(state_dictForLayer11)
netForFeatureExtractionLayer10.load_state_dict(state_dictForLayer10)
netForFeatureExtractionLayer8.load_state_dict(state_dictForLayer8)
netForFeatureExtractionLayer5.load_state_dict(state_dictForLayer5)
netForFeatureExtractionLayer2.load_state_dict(state_dictForLayer2)



input_nc=imageNumberOfChannel #input number of channels
output_nc=imageNumberOfChannel #output number of channels
ngf=64 # number of gen filters in first conv layer
norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
use_dropout=False


#changing all nets to eval mode
netForFeatureExtractionLayer23.eval()
netForFeatureExtractionLayer23.requires_grad=False

netForFeatureExtractionLayer11.eval()
netForFeatureExtractionLayer11.requires_grad=False

netForFeatureExtractionLayer10.eval()
netForFeatureExtractionLayer10.requires_grad=False

netForFeatureExtractionLayer8.eval()
netForFeatureExtractionLayer8.requires_grad=False

netForFeatureExtractionLayer5.eval()
netForFeatureExtractionLayer5.requires_grad=False

netForFeatureExtractionLayer2.eval()
netForFeatureExtractionLayer2.requires_grad=False


torch.no_grad()


eachPatch=imageSizeRow
numImageSplitRow=imageSizeRow/eachPatch
numImageSplitCol=imageSizeCol/eachPatch
cutY=list(range(0,imageSizeRow,eachPatch))
cutX=list(range(0,imageSizeCol,eachPatch))
additionalPatchPixel=64


layerWiseFeatureExtractorFunction=[nanVar,nanVar,netForFeatureExtractionLayer2,nanVar,nanVar,netForFeatureExtractionLayer5,nanVar,nanVar,netForFeatureExtractionLayer8,nanVar,\
                            netForFeatureExtractionLayer10,netForFeatureExtractionLayer11,nanVar,nanVar,nanVar,nanVar,nanVar,nanVar,nanVar,nanVar,\
                            nanVar,nanVar,nanVar,netForFeatureExtractionLayer23,nanVar,nanVar,nanVar,nanVar]


##Checking validity of feature extraction layers
validFeatureExtractionLayers=[2,5,8,10,11,23] ##Feature extraction from only these layers have been defined here
for outputLayer in outputLayerNumbers:
    if outputLayer not in validFeatureExtractionLayers:
        sys.exit('Feature extraction layer is not valid, valid values are 2,5,8,10,11,23')

##Extracting bi-temporal features
modelInputMean=0.406
last_layer = np.max(outputLayer)

for outputLayerIter in range(0,len(outputLayerNumbers)):
    outputLayerNumber=outputLayerNumbers[outputLayerIter]
    filterNumberForOutputLayer=filterNumberTable[outputLayerNumber]
    featurePercentileToDiscard=featurePercentileToDiscardTable[outputLayerNumber]
    featureNumberToRetain=int(np.floor(filterNumberForOutputLayer*((100-featurePercentileToDiscard)/100)))
    sizeReductionForOutputLayer=sizeReductionTable[outputLayerNumber]
    patchOffsetFactor=int(additionalPatchPixel/sizeReductionForOutputLayer)
    print('Processing layer number:'+str(outputLayerNumber))

    pbar = tqdm(range(len(preChangeImage)))
    i = 0
    for img1, img2 in zip(preChangeImage, postChangeImage):
        if np.sum(img1 == 0) == 0 and np.sum(img2 == 0) == 0:
            patchToProcessDate1 = img1
            patchToProcessDate2 = img2

            timeVector1Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])
            timeVector2Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])

            #converting to pytorch varibales and changing dimension for input to net
            patchToProcessDate1=patchToProcessDate1-modelInputMean

            inputToNetDate1=torch.from_numpy(patchToProcessDate1)
            inputToNetDate1=inputToNetDate1.float()
            inputToNetDate1=np.swapaxes(inputToNetDate1,0,2)
            inputToNetDate1=np.swapaxes(inputToNetDate1,1,2)
            inputToNetDate1=inputToNetDate1.unsqueeze(0).to(device)


            patchToProcessDate2=patchToProcessDate2-modelInputMean

            inputToNetDate2=torch.from_numpy(patchToProcessDate2)
            inputToNetDate2=inputToNetDate2.float()
            inputToNetDate2=np.swapaxes(inputToNetDate2,0,2)
            inputToNetDate2=np.swapaxes(inputToNetDate2,1,2)
            inputToNetDate2=inputToNetDate2.unsqueeze(0).to(device)

            
            #running model on image 1 and converting features to numpy format
            with torch.no_grad():
                obtainedFeatureVals1=layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate1)
            obtainedFeatureVals1=obtainedFeatureVals1.squeeze()
            obtainedFeatureVals1=obtainedFeatureVals1.cpu().data.numpy()

            #running model on image 2 and converting features to numpy format
            with torch.no_grad():
                obtainedFeatureVals2=layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate2)
            obtainedFeatureVals2=obtainedFeatureVals2.squeeze()
            obtainedFeatureVals2=obtainedFeatureVals2.cpu().data.numpy()
            #this features are in format (filterNumber, sizeRow, sizeCol)


            ##clipping values to +1 to -1 range, be careful, if network is changed, maybe we need to modify this
            obtainedFeatureVals1=np.clip(obtainedFeatureVals1,-1,+1)
            obtainedFeatureVals2=np.clip(obtainedFeatureVals2,-1,+1)


            for processingFeatureIter in range(0,filterNumberForOutputLayer):
                timeVector1Feature[:,:,processingFeatureIter]=resize(obtainedFeatureVals1[processingFeatureIter,:,:],(256,256))
                timeVector2Feature[:,:,processingFeatureIter]=resize(obtainedFeatureVals2[processingFeatureIter,:,:],(256,256))

            timeVectorDifferenceMatrix=timeVector2Feature - timeVector1Feature
            stdVectorDifference=np.std(timeVectorDifferenceMatrix,axis=(0,1))

            featuresOrderedPerStd=np.argsort(-stdVectorDifference)   #negated array to get argsort result in descending order
            nonZeroVector=featuresOrderedPerStd[0:featureNumberToRetain]
            modifiedTimeVector1=timeVector1Feature[:,:,nonZeroVector.astype(int)]
            modifiedTimeVector2=timeVector2Feature[:,:,nonZeroVector.astype(int)]


            ##Normalize the features (separate for both images)
            meanVectorsTime1Image=np.mean(modifiedTimeVector1,axis=(0,1))
            stdVectorsTime1Image=np.std(modifiedTimeVector1,axis=(0,1))
            stdVectorsTime1Image = stdVectorsTime1Image != 0
            normalizedModifiedTimeVector1=(modifiedTimeVector1-meanVectorsTime1Image)/stdVectorsTime1Image

            meanVectorsTime2Image=np.mean(modifiedTimeVector2,axis=(0,1))      
            stdVectorsTime2Image=np.std(modifiedTimeVector2,axis=(0,1))
            stdVectorsTime2Image = stdVectorsTime2Image != 0
            normalizedModifiedTimeVector2=(modifiedTimeVector2-meanVectorsTime2Image)/stdVectorsTime2Image


            ##feature aggregation across channels
            if outputLayerIter==0:
                timeVector1FeatureAggregated=np.copy(normalizedModifiedTimeVector1)
                timeVector2FeatureAggregated=np.copy(normalizedModifiedTimeVector2)
            else:
                timeVector1FeatureAggregated=np.concatenate((timeVector1FeatureAggregated,normalizedModifiedTimeVector1),axis=2)
                timeVector2FeatureAggregated=np.concatenate((timeVector2FeatureAggregated,normalizedModifiedTimeVector2),axis=2)

            del obtainedFeatureVals1, obtainedFeatureVals2, timeVector1Feature, timeVector2Feature, inputToNetDate1, inputToNetDate2 

            if outputLayerNumber == last_layer:                
                absoluteModifiedTimeVectorDifference=\
                            np.absolute(saturateImage().saturateSomePercentileMultispectral(timeVector1FeatureAggregated,5)-\
                            saturateImage().saturateSomePercentileMultispectral(timeVector2FeatureAggregated,5))

                #take absolute value for binary CD
                detectedChangeMap=np.linalg.norm(absoluteModifiedTimeVectorDifference,axis=(2))
                detectedChangeMapNormalized=(detectedChangeMap-np.amin(detectedChangeMap))/(np.amax(detectedChangeMap)-np.amin(detectedChangeMap))
                #plt.figure()
                #plt.imshow(detectedChangeMapNormalized)


                #detectedChangeMapNormalized=filters.gaussian(detectedChangeMapNormalized,3) #this one is with constant sigma
                cdMap=np.zeros(detectedChangeMapNormalized.shape, dtype=bool)

                if thresholdingStrategy == 'adaptive':
                    for sigma in range(101,202,50):
                        adaptiveThreshold=2*filters.gaussian(detectedChangeMapNormalized,sigma)
                        cdMapTemp=(detectedChangeMapNormalized>adaptiveThreshold) 
                        cdMapTemp=morphology.remove_small_objects(cdMapTemp,min_size=objectMinSize)
                        cdMap=cdMap | cdMapTemp
                elif thresholdingStrategy == 'otsu':
                    otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
                    cdMap = (detectedChangeMapNormalized>otsuThreshold) 
                    cdMap=morphology.remove_small_objects(cdMap,min_size=objectMinSize)
                elif thresholdingStrategy == 'scaledOtsu':
                    otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
                    cdMap = (detectedChangeMapNormalized>otsuScalingFactor*otsuThreshold) 
                    cdMap=morphology.remove_small_objects(cdMap,min_size=objectMinSize)
                else: 
                    sys.exit('Unknown thresholding strategy')
                cdMap=morphology.binary_closing(cdMap,morphology.disk(3))

    #             #Creating directory to save result
    #                 resultDirectory = './result/'
    #                 if not os.path.exists(resultDirectory):
    #                     os.makedirs(resultDirectory)

    #             #Saving the Binary CD result (a .mat file and a .png file)
    #             sio.savemat(resultDirectory+'binaryCdResult.mat', mdict={'cdMap': cdMap})
    #             plt.imsave(resultDirectory+'binaryCdResult_' + str(i) + '.png',np.repeat(np.expand_dims(cdMap,2),3,2).astype(float))
                cd_label[i,:,:] = cdMap
        i+=1
        pbar.update(1)

print("validation")
acc_mean, P, R, F1 = validation(damage_label, cd_label)

print("acc_mean:" + str(acc_mean))
print("P:" + str(P))
print("R:" + str(R))
print("F1:" + str(F1))
