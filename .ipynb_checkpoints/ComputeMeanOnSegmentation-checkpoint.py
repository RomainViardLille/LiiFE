import nibabel as nib
import os,sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

QUANT=sys.argv[1] 
LABEL=sys.argv[2] 
LABEL_VAL=int(sys.argv[3])

img_val=nib.load(QUANT)
val=img_val.get_fdata()

img_seg=nib.load(LABEL)
seg=img_seg.get_fdata()

interp_func=RegularGridInterpolator((np.linspace(0,255,256),np.linspace(0,255,256),np.linspace(0,79,80)),img_val.get_fdata())
vectRet=[]
for index, cluster_bool in np.ndenumerate(seg==LABEL_VAL) :
    if cluster_bool :
        x1,y1,z1=index
        x_coord,y_coord,z_coord=img_seg.affine[:3,:3]@[x1,y1,z1]+img_seg.affine[:3,3]
        affine_inv=np.linalg.inv(img_val.affine)
        vox_indices=np.dot(affine_inv,[x_coord,y_coord,z_coord,1])[:3]
        vectRet.append(vox_indices)    

print(np.round(np.mean(interp_func(vectRet)),0),end="")
print(np.round(np.mean(val(vectRet)),0),end="")
