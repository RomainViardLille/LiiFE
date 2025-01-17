import os, sys
import numpy as np
import nibabel as nb

fileName=sys.argv[1] 
imgNifti = nb.load(fileName)
data= imgNifti.get_fdata()

#gestion d'un volume 4D vs un volume 3D
if len(data.shape)>3 :
	for cpt in range(0,data.shape[3]):
		data[:,:,:,cpt] = np.roll(data[:,:,:,cpt],-1,axis=0)
		data[:,:,:,cpt] = np.roll(data[:,:,:,cpt],1,axis=1)
else :
	data= np.roll(data,-1,axis=0)
	data= np.roll(data,1,axis=1)

nb.save(nb.Nifti1Image(data, imgNifti.affine ),fileName)
