import os, sys
import numpy as np
import nibabel as nb

Quant=sys.argv[1] 
img = nb.load(Quant)
dataQuant= img.get_fdata()

TrackDensity=sys.argv[2] 
img = nb.load(TrackDensity)
dataDensity = img.get_fdata()
LABEL_VAL=int(sys.argv[3])
dataLabel=np.where(dataDensity==LABEL_VAL,1,0)
val_ret=np.sum(np.multiply(dataLabel,dataQuant))/np.sum(dataLabel)
print(np.round(val_ret,6))
