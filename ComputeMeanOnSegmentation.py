import nibabel as nib
import os,sys
import numpy as np
#from scipy.interpolate import RegularGridInterpolator

img_val=nib.load(sys.argv[1] )
val=img_val.get_fdata()

img_seg=nib.load(sys.argv[2] )
seg=img_seg.get_fdata()

# ~ interp_func=RegularGridInterpolator((np.linspace(0,255,256),np.linspace(0,255,256),np.linspace(0,79,80)),img_val.get_fdata())
vectRet=[]
vectWeight=[]
matLabel=seg


if len(sys.argv)>3 : #Moyenne non pondéré par les valeurs de la ROI (on utilise la/les valeur(s) en argument
	if len(sys.argv[3]) < 6 :
		matLabel=(seg==int(sys.argv[3]))
		for cpt in range(4,len(sys.argv)) :
			matLabel=(matLabel | (seg==int(sys.argv[cpt])))
else :  #Moyenne pondérée
	print(len(sys.argv))
	if len(sys.argv)>3 :
		for cpt in range(3,len(sys.argv)):
			print(sys.argv[cpt])
			seg = nib.load(sys.argv[cpt])
			matLabel = matLabel + seg.get_fdata()

if np.sum(matLabel) < 1:
	sys.exit()

for index, cluster_bool in np.ndenumerate(matLabel) :
	if cluster_bool :
		x1,y1,z1=index
		x_coord,y_coord,z_coord=img_seg.affine[:3,:3]@[x1,y1,z1]+img_seg.affine[:3,3]
		affine_inv=np.linalg.inv(img_val.affine)
		vox_indices=np.dot(affine_inv,[x_coord,y_coord,z_coord,1])[:3]
		vectRet.append(vox_indices)
		vectWeight.append(cluster_bool)

# ~ print(np.round(np.mean(interp_func(vectRet)),0),end="")
vectRet_int = [list(map(round,vecteur)) for vecteur in vectRet ]
vectRet_int_array=np.array(vectRet_int)

if len(sys.argv)>3 :
	# ~ if isinstance(int(sys.argv[3]),int) :
	if len(sys.argv[3]) < 6 :
		print(np.round(np.mean(val[vectRet_int_array[:,0],vectRet_int_array[:,1],vectRet_int_array[:,2]]),4),end="")
	else :
		print(np.round((val[vectRet_int_array[:,0],vectRet_int_array[:,1],vectRet_int_array[:,2]]*vectWeight)/np.sum(vectWeight),4),end="")
		# ~ print(np.round(np.sum(np.multiply(matLabel,val))/np.sum(matLabel),4),end="")
else :
		print(np.round((val[vectRet_int_array[:,0],vectRet_int_array[:,1],vectRet_int_array[:,2]]*vectWeight)/np.sum(vectWeight),4),end="")
	# ~ print(np.round(np.sum(np.multiply(matLabel,val))/np.sum(matLabel),4))
