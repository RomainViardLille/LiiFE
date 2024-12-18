import os, sys
import numpy as np
import nibabel as nb

fileName=sys.argv[1] 
imgNifti = nb.load(fileName)
data= imgNifti.get_fdata()

for cpt in range(0,data.shape[3]):
    data[:,:,:,cpt] = np.roll(data[:,:,:,cpt],-1,axis=0)
    data[:,:,:,cpt] = np.roll(data[:,:,:,cpt],1,axis=1)

nb.save(nb.Nifti1Image(data, imgNifti.affine ),fileName)
