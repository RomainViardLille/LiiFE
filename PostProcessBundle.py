#~ from tractseg.data import dataset_specific_utils
#~ from tractseg.libs.AFQ_MultiCompCorrection import AFQ_MultiCompCorrection
#~ from tractseg.libs.AFQ_MultiCompCorrection import get_significant_areas
#~ from tractseg.libs import metric_utils
#~ from tractseg.libs import tracking
#~ from tractseg.libs import tractometry
from dipy.tracking.utils import length
# Compute lookup table
from dipy.denoise.enhancement_kernel import EnhancementKernel
# Apply FBC measures
from dipy.tracking.fbcmeasures import FBCMeasures

import os, sys, re
import argparse
import math
from decimal import Decimal
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import scipy.stats
import pandas as pd
from tqdm import tqdm

import glob as glob
import nibabel as nib
from scipy.stats import t as t_dist
from tractseg.libs import plot_utils
from sklearn.linear_model import LinearRegression
from dipy.tracking.streamline import Streamlines
import dipy.stats.analysis as dsa

#TRACTO_TCK_IN=sys.argv[1]
#TRACTO_TCK_OUT=sys.argv[2]

TMP=sys.argv[1]
SUBJ=sys.argv[2]
#print(TRACTO_TCK_IN)
TRK=sys.argv[3]
TRACTO_TCK_IN=os.path.join("/NAS","dumbo","protocoles","Strokdem_discon","data",TMP,SUBJ,"FOD_iFOD2_trackings",TRK+".tck")
TRACTO_TCK_OUT=os.path.join("/NAS","dumbo","protocoles","Strokdem_discon","data",TMP,SUBJ,"FOD_iFOD2_trackings",TRK+"filtered.tck")

print(TRACTO_TCK_IN,TRACTO_TCK_OUT)

if  (os.path.isfile(TRACTO_TCK_IN)):
	#LIFE
	print('LIFE proceeding')
	D33 = 1.0
	D44 = 0.02
	t = 1
	k = EnhancementKernel(D33, D44, t)
	print("LUT computed")

	#for TRK in ["CC_1","CC_2","CC_3","CC_4","CC_5","CC_6","CC_7","AF_left","STR_left","FPT_left","AF_right","STR_right","FPT_right"] :

	if (not os.path.isfile(TRACTO_TCK_OUT)):

		sl_file = nib.streamlines.load(TRACTO_TCK_IN)
		streamlines = sl_file.streamlines

		fbc = FBCMeasures(streamlines,k,num_threads=10,verbose=True)

		# Apply a threshold on the RFBC to remove spurious fibers
		fbc_sl_thres, clrs_thres, rfbc_thres = fbc.get_points_rfbc_thresholded(0.125, emphasis=0.01,verbose=True)

		print('Nb fibers before LIFE : ',len(streamlines))
		print('Nb fibers after LIFE : ',len(fbc_sl_thres))

		# ~ print("Cleaning 1 : Length")

		# ~ val=list(length(streamlines))
		# ~ long_streamlines = Streamlines()
		# ~ thres = np.mean(val)+2*np.std(val) 
		# ~ for i, sl in enumerate(streamlines):
			# ~ if val[i] < thres:
				# ~ long_streamlines.append(sl)

		# ~ print('Nb fibers after cleaning 1 (keep streamline shorter than (mean + 2 std) ) : ',len(long_streamlines))        
		# ~ print('mean length and its std before cleaning : ',np.mean(val),np.std(val))
		# ~ val=list(length(long_streamlines))
		# ~ print('mean length and its std after cleaning : ',np.mean(val),np.std(val))

		# ~ print("Cleaning 1 : Mahalanobis ")
		# ~ weights = dsa.gaussian_weights(long_streamlines)
		# ~ mal_streamlines = Streamlines()
		# ~ thres = np.mean(weights)+2*np.std(weights) 
		# ~ for i, sl in enumerate(long_streamlines):
			# ~ if np.mean(weights,axis=1)[i] < thres:
				# ~ mal_streamlines.append(sl)

		# ~ print('Nb fibers after cleaning 2 (keep streamline where mahalanobis distance is less than (mean + 2 std) ) : ',len(mal_streamlines))       
		# ~ print('mean mahalanobis distance and its std before cleaning : ',np.mean(weights),np.std(weights))
		# ~ weights = dsa.gaussian_weights(mal_streamlines)
		# ~ print('mean mahalanobis distance and its std after cleaning : ',np.mean(weights),np.std(weights))

		tracto = nib.streamlines.tractogram.Tractogram(fbc_sl_thres,affine_to_rasmm=sl_file.header['voxel_to_rasmm'])
		nib.streamlines.save(tracto,TRACTO_TCK_OUT)  
