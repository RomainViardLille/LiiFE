#!/home/global/anaconda37/bin/python
# -*-coding:Latin-1 -*

import nibabel as nib
import numpy as np
import os,sys
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
from scipy import stats
import plotnine as p9
import csv
import statsmodels.api as sm
from radiomics import featureextractor, getTestCase
import SimpleITK as sitk
import import_ipynb
#sys.path.append('/home/romain/SVN/python/romain')
#from Predistim_Library import *
import radiomics as rm
from scipy import ndimage
from scipy.ndimage import gaussian_filter

def getValueAndSetInDataFrame4CoocurrenceMatrix(imgFile,maskFile,sub,struct,parametre):
	params = os.path.join("/","home","romain","SVN","python","romain","Params2.yaml")
	listPara=(['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Entropy','original_firstorder_InterquartileRange',
	'original_firstorder_Kurtosis','original_firstorder_Median','original_firstorder_Minimum','original_firstorder_Range',
	'original_firstorder_RootMeanSquared','original_firstorder_Skewness','original_firstorder_Uniformity','original_firstorder_Variance','original_glcm_Autocorrelation',
	'original_glcm_ClusterProminence','original_glcm_Contrast','original_glcm_Correlation','original_glcm_InverseVariance','original_glcm_JointAverage',
	'original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_MCC','original_glcm_MaximumProbability','original_glcm_SumSquares','original_glcm_Idn','original_glcm_DifferenceEntropy','original_glcm_DifferenceVariance','original_glcm_SumAverage','original_glcm_Idmn','original_glcm_Id'])
	sitk_img = sitk.GetImageFromArray(imgFile)
	sitk_mask = sitk.GetImageFromArray(maskFile)
	extractor = featureextractor.RadiomicsFeatureExtractor(params)
	result = extractor.execute(sitk_img,sitk_mask)
	for para in listPara:
		df.loc[sub,parametre+"_"+para.split('_')[-1]+"_"+struct]=result[para]

SUBJ=sys.argv[1]
CQT1=sys.argv[2]
CQTE=sys.argv[3]

STUDY_PATH=os.path.join("/NAS","deathrow","protocoles","predistim")
CQ_FILE=os.path.join(STUDY_PATH,'Predistim_MRIdata_v20200406_update20201204.xlsx')
CLI_FILE=os.path.join(STUDY_PATH,'DataCli_20210104_09122020.xlsx')
SNP_FILE=os.path.join(STUDY_PATH,'SNPs_dosages_20201106.xlsx')
IMA_FILE=os.path.join(STUDY_PATH,'20210105_T1_R2_QSM_Values.xlsx')

df = pd.DataFrame()

if int(CQT1)>1 :
	listePara=[]
	listePara.append('3DT1')
	if int(CQTE)>1 :
		listePara.append('R2')
		listePara.append('QSM')
for PARA in listePara:
	if PARA=='3DT1' :
		#file=os.path.join(STUDY_PATH,PARA,SUBJ,'3DT1_'+SUBJ+'.nii.gz')
		file=os.path.join(STUDY_PATH,'HCP',SUBJ,'T1w',SUBJ,'mri','T1FS_resliced.nii.gz')
	else :
		file=os.path.join(STUDY_PATH,PARA,SUBJ,PARA+'_lin_3DT1.nii.gz')

	#if (((PARA=='3DT1') & (int(CQT1)>1)) or ((int(CQT1)>1) & (int(CQTE)>1))):
	#print(ind,SUBJ)
	if not(PARA=='QSM' and  not(SUBJ[0:2] in ['01','06','07','08','17','19'])):
		if (os.path.isfile(file)) :
			#print(file)
			img_data=np.asanyarray(nib.load(file).dataobj)
			#VOLBRAIN caudé 3/4, Putamen 5/6, thalamus 7/8, Globus pallidus 9/10, hipocampus 11/12, amigdala 13/14
			struct_file=os.path.join(STUDY_PATH,'3DT1',SUBJ,'native_lab_'+SUBJ+'_resliced.nii.gz')
			#print(struct_file)
			if os.path.exists(struct_file) :
				list_oar=["Ventricules_L","Ventricules_R","Caudate_L","Caudate_R","Putamen_L","Putamen_R","Thalamus_L","Thalamus_R","Globus_pallidus_L","Globus_pallidus_R","Hippocampus_L","Hippocampus_R","Amigdala_L","Amigdala_R"]
				struct = nib.load(struct_file)
				struct_data=np.asanyarray(struct.dataobj,dtype=int)
				voxel_dims = (struct.header["pixdim"])[1]*(struct.header["pixdim"])[2]*(struct.header["pixdim"])[3]
				for cpt,name_struct in zip(np.arange(1,17,1),list_oar):
					tmp = (struct_data==cpt)
					if ( len(img_data[tmp]) > 1 ):
						getValueAndSetInDataFrame4CoocurrenceMatrix(img_data,tmp.astype(int),SUBJ,name_struct,PARA)
						df.loc[SUBJ,PARA+"_volume_"+name_struct]=len(img_data[tmp])*voxel_dims
						df.loc[SUBJ,PARA+"_Maximum_"+name_struct]=np.nanmax(img_data[tmp])
			else :
				print("Pas de volbrain pour ",SUBJ)

			#FREESURFER cingulaireAnterieure 1002&1026/2002&2026 superieurFrontal 1028/2028
			#struct_file=os.path.join(STUDY_PATH,'3DT1',SUBJ,'aparc+aseg_resliced.nii.gz')
			struct_file=os.path.join(STUDY_PATH,'HCP',SUBJ,'T1w',SUBJ,'mri','aparc+aseg_resliced.nii.gz')
			if os.path.exists(struct_file) :
				struct = nib.load(struct_file)
				struct_data=np.asanyarray(struct.dataobj)
				voxel_dims = (struct.header["pixdim"])[1]*(struct.header["pixdim"])[2]*(struct.header["pixdim"])[3]
				tmp = ((struct_data==1002) | (struct_data==1026))
				if ( len(img_data[tmp]) > 1 ):
					getValueAndSetInDataFrame4CoocurrenceMatrix(img_data,tmp.astype(int),SUBJ,"CingulaireAnt_L",PARA)
					df.loc[SUBJ,PARA+"_volume_CingulaireAnt_L"]=len(img_data[tmp])*voxel_dims
					df.loc[SUBJ,PARA+"_Maximum_CingulaireAnt_L"]=np.nanmax(img_data[tmp])
				tmp = ((struct_data==2002) | (struct_data==2026))
				if ( len(img_data[tmp]) > 1 ):
					getValueAndSetInDataFrame4CoocurrenceMatrix(img_data,tmp.astype(int),SUBJ,"CingulaireAnt_R",PARA)
					df.loc[SUBJ,PARA+"_volume_CingulaireAnt_R"]=len(img_data[tmp])*voxel_dims
					df.loc[SUBJ,PARA+"_Maximum_CingulaireAnt_R"]=np.nanmax(img_data[tmp])
				tmp = (struct_data==1028)
				if ( len(img_data[tmp]) > 1 ):
					getValueAndSetInDataFrame4CoocurrenceMatrix(img_data,tmp.astype(int),SUBJ,"FrontalSup_L",PARA)
					df.loc[SUBJ,PARA+"_volume_FrontalSup_L"]=len(img_data[tmp])*voxel_dims
					df.loc[SUBJ,PARA+"_Maximum_FrontalSup_L"]=np.nanmax(img_data[tmp])
				tmp = (struct_data==2028)
				if ( len(img_data[tmp]) > 1 ):
					getValueAndSetInDataFrame4CoocurrenceMatrix(img_data,tmp.astype(int),SUBJ,"FrontalSup_R",PARA)
					df.loc[SUBJ,PARA+"_volume_FrontalSup_R"]=len(img_data[tmp])*voxel_dims
					df.loc[SUBJ,PARA+"_Maximum_FrontalSup_R"]=np.nanmax(img_data[tmp])
				tmp = (struct_data==43) #ajout  Ventricule
				erod=ndimage.binary_erosion(tmp, structure=np.ones((5,5,5))).astype(float)
				idx = np.where(erod)
				if ( len(img_data[tmp]) > 1 ) and ( len(idx[0]) > 5) :
					getValueAndSetInDataFrame4CoocurrenceMatrix(img_data,erod.astype(int),SUBJ,"Ventricule_R",PARA)
					df.loc[SUBJ,PARA+"_volume_Ventricule_R"]=len(img_data[idx])*voxel_dims
					df.loc[SUBJ,PARA+"_Maximum_Ventricule_R"]=np.nanmax(img_data[idx])
				tmp = (struct_data==4) #ajout  Ventricule
				erod=ndimage.binary_erosion(tmp, structure=np.ones((5,5,5))).astype(float)
				idx = np.where(erod)
				if ( len(img_data[tmp]) > 1 ) and ( len(idx[0]) > 5) :
					getValueAndSetInDataFrame4CoocurrenceMatrix(img_data,erod.astype(int),SUBJ,"Ventricule_L",PARA)
					df.loc[SUBJ,PARA+"_volume_Ventricule_L"]=len(img_data[idx])*voxel_dims
					df.loc[SUBJ,PARA+"_Maximum_Ventricule_L"]=np.nanmax(img_data[idx])
				tmp = (struct_data==41) #ajout White Matter
				erod=ndimage.binary_erosion(tmp, structure=np.ones((5,5,5))).astype(float)
				erod[:,:int(struct_data.shape[1]/2),:]=0
				idx = np.where(erod)
				if ( len(img_data[tmp]) > 1 ) and ( len(idx[0]) > 5) :
					getValueAndSetInDataFrame4CoocurrenceMatrix(img_data,erod.astype(int),SUBJ,"WM_R",PARA)
					df.loc[SUBJ,PARA+"_volume_WM_R"]=len(img_data[idx])*voxel_dims
					df.loc[SUBJ,PARA+"_Maximum_WM_R"]=np.nanmax(img_data[idx])
				tmp = (struct_data==2) #ajout White Matter
				erod=ndimage.binary_erosion(tmp, structure=np.ones((5,5,5))).astype(float)
				erod[:,:int(struct_data.shape[1]/2),:]=0
				idx = np.where(erod)
				if ( len(img_data[tmp]) > 1 ) and ( len(idx[0]) > 5) :
					getValueAndSetInDataFrame4CoocurrenceMatrix(img_data,erod.astype(int),SUBJ,"WM_L",PARA)
					df.loc[SUBJ,PARA+"_volume_WM_L"]=len(img_data[idx])*voxel_dims
					df.loc[SUBJ,PARA+"_Maximum_WM_L"]=np.nanmax(img_data[idx])
			else :
				print("Pas de freesurfer pour",SUBJ)

			#print("ATLAS_GPe_GPi_STh_STR_RN_SN")
			for cpt2,name_struct in enumerate(['FLASH_RN_L_','FLASH_SN_L_','QSM_GPe_L_','QSM_GPi_L_','FLASH_STh_L_','MP2RAGE_STR_L_','FLASH_RN_R_','FLASH_SN_R_','QSM_GPe_R_','QSM_GPi_R_','FLASH_STh_R_','MP2RAGE_STR_R_']):
				struct_file=os.path.join(STUDY_PATH,'3DT1',SUBJ,name_struct+SUBJ+'_on3DT1.nii.gz')
				if os.path.exists(struct_file):
					struct_L_ = nib.load(struct_file)
					struct_data=np.asanyarray(struct_L_.dataobj)
					voxel_dims = (struct_L_.header["pixdim"])[1]*(struct_L_.header["pixdim"])[2]*(struct_L_.header["pixdim"])[3]
					tmp = (struct_data>0.5)
					if ( len(img_data[tmp]) > 1):
						getValueAndSetInDataFrame4CoocurrenceMatrix(img_data,tmp.astype(int),SUBJ,name_struct,PARA)
						df.loc[SUBJ,PARA+"_volume_"+name_struct]=len(img_data[tmp])*voxel_dims
						df.loc[SUBJ,PARA+"_Maximum_"+name_struct]=np.nanmax(img_data[tmp])
				else :
					print("Pas d'atlas pour",SUBJ,struct_file)
		else :
			print("Pas de fichier para",file)

df.loc[SUBJ,'CQT1']=CQT1
df.loc[SUBJ,'CQTE']=CQTE
df.to_csv(os.path.join(STUDY_PATH,SUBJ+'_V1_20211022_T1_R2_QSM_Values_Lat.csv'))
