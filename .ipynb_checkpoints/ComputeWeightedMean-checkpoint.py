import os, sys
import numpy as np
import nibabel as nb

Quant=sys.argv[1] 
img = nb.load(Quant)
dataQuant= img.get_fdata()

TrackDensity=sys.argv[2] 
img = nb.load(TrackDensity)
dataDensity = img.get_fdata()

nbSeg=len(sys.argv)-3
for seg in range(3,3+nbSeg):
	img = nb.load(sys.argv[seg])
	dataDensity = dataDensity + img.get_fdata()

val_ret=np.sum(np.multiply(dataDensity,dataQuant))/np.sum(dataDensity)
print(np.round(val_ret,4))
