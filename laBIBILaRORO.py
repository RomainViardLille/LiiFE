from scipy.spatial.distance import cdist
import os
import numpy as np
#import igraph as ig
from sklearn.linear_model import LinearRegression
import pingouin as pg
from scipy.stats import f_oneway
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#%matplotlib inline
from brainstat.datasets import fetch_mask, fetch_template_surface,fetch_parcellation
from brainspace.mesh.mesh_io import read_surface
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect
from nilearn import surface, datasets, plotting
from brainspace.plotting import plot_hemispheres
from IPython.display import display,Image
import statsmodels.api as sm
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect,MixedEffect

glasser = fetch_parcellation("fsaverage5", "glasser", 360)
pial_left, pial_right = fetch_template_surface("fsaverage5",layer="pial",join=False)
inflated_left, inflated_right = fetch_template_surface("fsaverage5",layer="inflated",join=False)
inflated_combined = fetch_template_surface("fsaverage5",layer="inflated",join=True)
pial_combined = fetch_template_surface("fsaverage5", join=True)
pial_combined_fslr32k = fetch_template_surface("fslr32k", join=True)
pial_left_fslr32k, pial_right_fslr32k = fetch_template_surface("fslr32k", join=False)
inflated_left_fslr32k,inflated_right_fslr32k = fetch_template_surface("fslr32k",layer="inflated", join=False)
mask = fetch_mask("fsaverage5")

pd.option_context('mode.use_inf_as_na', True)

# Obtenir les couleurs de la colormap 'tab10'
tab10_colors = plt.get_cmap('tab10').colors
tab20_colors = plt.get_cmap('tab20').colors

# Fonction pour convertir une couleur RGB en code d'échappement ANSI
def rgb_to_ansi(r, g, b):
    return f'\033[38;2;{int(r*255)};{int(g*255)};{int(b*255)}m'

# Convertir les couleurs en codes d'échappement ANSI
ansi_colors10 = [rgb_to_ansi(*color) for color in tab10_colors]
ansi_colors20 = [rgb_to_ansi(*color) for color in tab20_colors]

# Définir les séquences d'échappement ANSI pour la couleur bleue
BLUE = "\033[94m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"  # Utilisation du magenta pour le rose
YELLOW = "\033[93m"  # Utilisation du jaune pour l'orange
RED = "\033[91m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"


print("La bibil à RORO loaded")

def compare_matrix(matrice1,matrice2,name1,name2,nb_perm,columns,community_pathname):
    """ 
        matrice1=matriceG1
        matrice2=matriceG2
        name1=nameG1
        name2=nameG2
        nb_perm=10
        columns=matriceG1.columns
        community_pathname=os.path.join(DIR,"screenshots")
    """
    # Calculer la distance entre les deux matrices à l'aide de la fonction cdist()
    distance_ref = cdist(matrice1.corr(), matrice2.corr(), metric='euclidean')
    #np.fill_diagonal(distance_ref, 0)

    # Afficher la matrice de distances
    print(f"Euclidean distance between {name1} and {name2} is {np.mean(distance_ref):.3f}")
    dist_list4nodes=[]
    dist_list4edges=[]

    for cpt in range(nb_perm):
        df_perm=pd.concat([matrice1,matrice2],axis=0).sample(frac=1)
        #permutations = np.random.permutation(df_perm.shape[0])
        #tableau_2d_permuted = df_perm[permutations, :]
        #=tableau_2d_permuted[:matrice1.shape[0]]
        #matrice2_tmp=tableau_2d_permuted[matrice1.shape[0]:]
        distance = cdist(df_perm.iloc[:matrice1.shape[0],:].corr(), df_perm.iloc[matrice1.shape[0]:,:].corr(), metric='euclidean')

        # Afficher la matrice de distances
        dist_list4nodes.append(np.mean(distance))
        dist_list4edges.append(distance)


    nb_sup=np.sum(np.array(dist_list4nodes)>=np.mean(distance_ref))
    if (nb_sup/nb_perm < 0.05):
        print(f"Sur {nb_perm} permutations, seulement {nb_sup} ont une distance superieure,les deux matrices sont donc significativement différentes au seuil de p<0.05 (p={nb_sup/nb_perm})")
    else:
        print(f"Sur {nb_perm} permutations, {nb_sup} ont une distance superieure,les deux matrices ne sont donc significativement différentes (p={nb_sup/nb_perm}) au seuil de 0.05")

    #nodes
    n = int(len(np.mean(distance_ref,axis=0)) * 0.05)
    indices = np.argpartition(np.mean(distance_ref,axis=0), -n)[-n:]
    indices = indices[np.argsort(np.mean(distance_ref,axis=0)[indices])[::-1]]
    print(f"Indices des 5% de nodes ayant valeurs les plus hautes : {indices} et {columns[indices]}")
    
    #edges
    nb_nodes=len(np.mean(distance_ref,axis=0)) 
    nb_edges = int((nb_nodes * nb_nodes - nb_nodes) /2)
    dist_sum=np.sum((np.array(dist_list4edges)<distance),axis=0)
    dist_sum=dist_sum/nb_perm
    
    b_fdr,ps_fdr=sm.stats.fdrcorrection(1-dist_sum.ravel(),alpha=0.05,method='indep',is_sorted=False)  
    print(f"{nb_nodes} nodes, {nb_edges} connexions, {np.sum(b_fdr)} significantly different for p_fdr < 0.05")
    sqrt_b_fdr = int(np.sqrt(b_fdr.shape[0]))
    mask=((np.abs(matrice1.corr()) > 0.6) | (np.abs(matrice2.corr()) > 0.6)) & (b_fdr.reshape(sqrt_b_fdr,sqrt_b_fdr))
    highest_indices = np.argwhere(mask)
    nb_edges_signdiff=len(highest_indices)
    print(f"et {nb_edges_signdiff} avec des correlations >0.6 dans une des deux matrices")
    
    df_tmp = pd.DataFrame(data=dist_sum, columns=matrice1.columns, index=matrice1.columns)
    #np.fill_diagonal(matriceG1,)
    #displayed_mat = np.triu(displayed_mat,1)
    plt.figure()
    sns.heatmap(1-df_tmp.where(mask),xticklabels=columns,yticklabels=columns,vmin=-1, vmax=1,cbar_kws={'shrink': 0.5})
    plt.title(f"p_fdr < 0.05 and corr > 0.6 ")
    plt.savefig(os.path.join(community_pathname,f"{name1}_{name2}.png"), bbox_inches='tight', pad_inches=0.1)
    plt.show()
    for couple in highest_indices:
        print(f"la correlation entre {columns[couple[0]]} et {columns[couple[1]]} est de {matrice1.iloc[couple[0],couple[1]]:.3f} pour {name1} et {matrice2.iloc[couple[0],couple[1]]:.3f} pour {name2}  (p_value = {1-dist_sum[couple[0],couple[1]]:.3f})")
    return dist_sum

def get_communities(matrice,df_roi,fig_title,community_pathname):
    #matrice=np.where([(matrice<-0.6) | (matrice>0.6)],matrice,0)

    # Charger la matrice de corrélation à partir d'un fichier ou d'une variable
    matrice_correlation = np.absolute(matrice)
    # Convertir la matrice de corrélation en un graphique à l'aide de igraph
    graph = ig.Graph.Weighted_Adjacency(matrice_correlation, mode="upper")

    # Effectuer la détection de communautés à l'aide de l'algorithme de Louvain
    communities = graph.community_multilevel(weights=graph.es["weight"])
    
    # Extraire les étiquettes de communauté pour chaque nœud
    node_labels1 = communities.membership
    n_communities = len(set(node_labels1))
    print(f"{n_communities} found")
    new_order=[]
    for cpt in range(n_communities):
        tmp_array=np.where(np.array(node_labels1)==cpt)
        new_order=np.concatenate([new_order,np.array(tmp_array[0])])
        print(f"ROI for community {cpt} are : {matrice.columns[tmp_array]}")
        view = plotting.view_connectome(matrice.iloc[tmp_array][matrice.columns[tmp_array]], list(df_roi.loc[matrice.columns[tmp_array]].values), edge_threshold='95%')
        view.save_as_html(os.path.join(community_pathname,f"{fig_title}_{cpt}.html"))

    matrice_com = matrice.iloc[new_order,new_order]
    plt.figure()
    sns.heatmap(matrice_com,vmin=-1, vmax=1,cbar_kws={'shrink': 0.5},cmap="jet")
    list_com=list()
    list_com.append(0)
    for cpt in set(communities.membership):
        list_com.append(communities.membership.count(cpt))
    for cpt in np.arange(1,len(list_com)-1):
        a=np.sum(list_com[:cpt])
        b=np.sum(list_com[:cpt+1])
        c=np.sum(list_com[:cpt+2])
        plt.plot([a, c], [b, b],linestyle='-', color='green',alpha=1,linewidth=3)
        plt.plot([b, b], [a, c],linestyle='-', color='green',alpha=1,linewidth=3)
    plt.title(fig_title)
    plt.savefig(os.path.join(community_pathname,f"{fig_title}.png"), bbox_inches='tight', pad_inches=0.1)
    plt.show()

    return communities,new_order

def unconfound(y, confound, group_data=False):
    """
    This will remove the influence "confound" has on "y".

    If the data is made up of two groups, the group label (indicating the group) must be the first column of
    'confound'. The group label will be considered when fitting the linear model, but will not be considered when
    calculating the residuals.

    Args:
        y: [samples, targets]
        confound: [samples, confounds]
        group_data: if the data is made up of two groups (e.g. for t-test) or is just
                    one group (e.g. for correlation analysis)
    Returns:
        y_correct: [samples, targets]
    """
    # Demeaning beforehand or using intercept=True has similar effect
    #y = demean(y)
    #confound = demean(confound)

    lr = LinearRegression(fit_intercept=True).fit(confound, y)  # lr.coef_: [targets, confounds]
    if group_data:
        y_predicted_by_confound = lr.coef_[:, 1:] @ confound[:, 1:].T
    else:
        y_predicted_by_confound = lr.coef_ @ confound.T  # [targets, samples]
    y_corrected = y.T - y_predicted_by_confound
    return y_corrected.T  # [samples, targets]

# Normalisation Min-Max pour les variables explicatives et les covariables
def MinMax_func(thelistCol,theDF):
    for col in  thelistCol:
    # Calculer les valeurs min et max de la colonne
        min_value = theDF[col].min()
        max_value = theDF[col].max()
    # Appliquer la formule de normalisation min-max
        theDF[col+'NormMinMax'] = (theDF[col] - min_value) / (max_value - min_value)

### Normalisation Z-Score pour les variables explicatives et les covariables
def ZScore_func(thelistCol,theDF):
    for col in thelistCol :
    # Calculer les valeurs min et max de la colonne
        meanVal = theDF[col].mean()
        stdVal = theDF[col].std()
    # Appliquer la formule de normalisation min-max
        theDF[col+'NormZScore'] = (theDF[col] - meanVal) / stdVal

def filter_group(group_df,variable,nb_std=3):
    mean = group_df[variable].mean()
    std = group_df[variable].std()
    return group_df[(group_df[variable] >= mean - nb_std * std) & (group_df[variable] <= mean + nb_std * std)]
    
def remove_outliers_bygroup(df, variable, group, nb_std=3,verbose=False):
    """
    Supprime les lignes du DataFrame df qui dépassent de plus ou moins trois écarts-types
    de la moyenne de la variable spécifiée par sous-groupes.

    Parameters:
    df (pd.DataFrame): Le DataFrame à traiter.
    variable (str): Le nom de la variable pour laquelle supprimer les outliers.
    group (str): Le nom de la colonne contenant les sous-groupes.
    nb_std (int): Le nombre d'écarts-types pour définir les outliers.

    Returns:
    pd.DataFrame: Le DataFrame sans les outliers.
    """
    if verbose:
        print(f"Removing outliers for variable '{variable}' by group '{group}' with {nb_std} standard deviations.")
        print(f"Initial number of rows: {len(df)}")
        print(df.groupby(group)[variable].agg(['mean', 'std']))
    df = df.groupby(group).apply(lambda x: filter_group(x, variable, nb_std)).reset_index(drop=True)
    if verbose:
        print(f"Final number of rows: {len(df)}")
        print(df.groupby(group)[variable].agg(['mean', 'std']))

    return df

def test_covar_funcOnTwoGroups(theCovList,theDF,ong,group1,group2,p_thres=0):
    theDF = theDF.replace([np.inf, -np.inf], np.nan)
    #regression des covariables
    for covar in theCovList :
        g1=theDF[theDF[ong] == group1][covar]
        g2=theDF[theDF[ong] == group2][covar]    
        t_statistic, p_value = stats.ttest_ind(g1,g2)
        print(f"T-test for {covar} : p-value={np.round(p_value,5)}")
        if p_value < p_thres:
            plt.figure()
            sns.boxplot(x=ong, y=covar, data=theDF, palette="Set2")
            sns.stripplot(x=ong, y=covar, data=theDF, color=".3", jitter=True)
            plt.title(f"{covar} - {group1} vs {group2} (p={np.round(p_value,5)})")
            plt.show()


def test_covar_funcWithANOVA(theCovList,theDF,ong,p_thres=0):
    for covar in theCovList :
        print(f"\033[1;32m {covar} \033[0m")
        table = theDF.groupby(ong).agg({covar: ['size','mean',"min", "max", "std"]}) 
        aov = pg.anova(data=theDF, dv=covar, between=ong, detailed=True)
        print(f"{table.round(1)} \n ANOVA for {covar} : {np.round(aov[:]['p-unc'].values[0],5)}")    
        if aov[:]['p-unc'].values[0] < p_thres:
            plt.figure()
            sns.boxplot(x=ong, y=covar, data=theDF, palette="Set2")
            sns.stripplot(x=ong, y=covar, data=theDF, color=".3", jitter=True)
            plt.title(f"{covar} (p={np.round(aov[:]['p-unc'].values[0],5)})")
            plt.show()
        # FDR-corrected post hocs with Hedges'g effect size
        #posthoc = pg.pairwise_tests(data=df, dv=covar, within='group',parametric=True, padjust='fdr_bh', effsize='hedges')
        #pg.print_table(posthoc, floatfmt='.3f')
        
def regress_covar_func(theSulcusList,theCovList,theDFsulcus,theDFvar,display=False):
    theDFvar= pd.DataFrame(theDFvar[theCovList].astype(np.float32),columns=theCovList) 
    #regression des covariables
    for covar in theCovList :
        if display:
            plt.figure()
            plt.plot(theDFvar[covar], theDFsulcus[theSulcusList[0]], "bo", label="Données brutes") # les coordonnées (x, y) representés par des points
        theSulcusList=theDFsulcus[theSulcusList].dropna(axis=1).columns          
        theDFsulcus = theDFsulcus.copy()  # Ajouté pour éviter le warning
        theDFsulcus.loc[:,theSulcusList] = theDFsulcus[theSulcusList].apply(lambda x: unconfound(x, theDFvar[covar].values.reshape(-1,1), False))
        if display:
            plt.plot(theDFvar[covar], theDFsulcus[theSulcusList[0]], "g.", label="Données corr") # les coordonnées (x, y) representés par des points        
            plt.xlabel(covar)
            plt.ylabel(theSulcusList[0])

def grp_comp_surface_func(groupUsed,listeCovar,ssdf_covar,ssdf_CT,STUDY_PATH,threshold_p=0.01,threshold_size=50,MyPalette=plt.get_cmap("Pastel1")):
        GP1=ssdf_covar[groupUsed].unique()[0]
        GP2=ssdf_covar[groupUsed].unique()[1]
   
        contrast_group = (ssdf_covar[groupUsed] == GP1).astype(int) - (ssdf_covar[groupUsed] == GP2).astype(int)
        term_group = FixedEffect(ssdf_covar[groupUsed])

        #model_group = term_group
        model_group_age_educ=term_group
        
        for varc in listeCovar:
            model_group_age_educ = model_group_age_educ + FixedEffect(ssdf_covar[varc])

        #RV term_age = FixedEffect(ssdf_covar[listeCovar[0]])
        #RV term_educ = FixedEffect(ssdf_covar[listeCovar[1]])
        #term_subject = MixedEffect(ssdf_covar["IDENTIFIANT"])
    
        #Créer le modèle avec covariables
        #RV model_group_age_educ = term_group + term_age + term_educ + term_age * term_educ #+ term_subject

        print(f"Group comparison between {GP1} and {GP2}")
        #print(f"Contrast : {contrast_group}")
        #print(f"Group 1 : {GP1} ({np.sum(ssdf_covar[groupUsed] == GP1)} subjects)")
        #print(f"Group 2 : {GP2} ({np.sum(ssdf_covar[groupUsed] == GP2)} subjects)")
        #print(f"terms : {term_group}")
        #print(f"model : {model_group_age_educ}")    

        slm_group = SLM(
            model_group_age_educ,
            contrast_group,
            surf=pial_combined,
            mask=mask,
            correction=["fdr", "rft"],
            two_tailed=True,
            cluster_threshold=0.01,
        )
        print(f"fitting the model")
        slm_group.fit(ssdf_CT.values)

        cp = [np.copy(slm_group.P["pval"]["C"])]
        [np.place(x, np.logical_or(x > 0.05, ~mask), np.nan) for x in cp]
        
        for contrast in [0,1]:           
            filtered_df = slm_group.P['clus'][contrast][(slm_group.P['clus'][contrast]['nverts'] > threshold_size) & (slm_group.P['clus'][contrast]['P'] < threshold_p)]
            if len(filtered_df)>0 :
                figure_title=os.path.join(STUDY_PATH,f"CTSurface_groupscomparison_{GP1}_vs_{GP2}_contrast{contrast}.jpg")
                labText=f"{np.sum(ssdf_covar[groupUsed] == GP1)} {GP1}\n{np.sum(ssdf_covar[groupUsed] == GP2)} {GP2}"
                max_cluster_nb=np.max(filtered_df["clusid"].values)
                test = np.copy(slm_group.P["clusid"][contrast].reshape(-1))
                test = np.where(test > max_cluster_nb, 0, test)
                print(figure_title)
                plot_hemispheres(inflated_left, inflated_right,test,label_text=[labText],color_range=(0,9),color_bar=True,
                    cmap="tab10", embed_nb=True, size=(1400, 200),zoom=1.45, nan_color=(0.7, 0.7, 0.7, 1),
                    cb__labelTextProperty={"fontSize": 12},transparent_bg=False,screenshot=True, offscreen=False,filename=figure_title)
                display(Image(filename=figure_title))
                print(f"{BLUE}***** \n clusters infos : \n***** \n {RESET} {filtered_df}")
                df_tmp=slm_group.P['peak'][contrast]
                print(f"{BLUE}***** \n peaks infos : \n***** \n {RESET} {df_tmp[(df_tmp['clusid'] <= max_cluster_nb)]}")
                for clu in filtered_df["clusid"].values :
                    df_copy=ssdf_covar.loc[:,[groupUsed]+listeCovar]
                    inds=np.where(slm_group.P['clusid'][contrast]==clu)[1]
                    name=f"cluster_{str(clu)}"
                    print(f'{ansi_colors10[clu]} {BOLD} {name} {RESET}')
                    df_copy[name] = np.mean(ssdf_CT.values[:,inds],1)
                    plt.figure()
                    ax = plt.gca()
                    regress_covar_func([name],listeCovar,df_copy,df_copy,False)
                    # Faire une copie du dictionnaire palette
                    palette_copy = MyPalette.copy()

                    # Obtenir les groupes uniques présents dans df_copy
                    groups_present = df_copy[groupUsed].unique()

                    # Supprimer les groupes non présents dans df_copy de la copie du dictionnaire palette
                    palette_copy = {group: color for group, color in palette_copy.items() if group in groups_present}

                    # sns.stripplot(df_copy,y=name,x=groupUsed,size=3,color="black", order=palette_copy.keys())
                    # sns.violinplot(df_copy,y=name,x=groupUsed, palette=palette_copy,order=palette_copy.keys())
                    sns.stripplot(df_copy, y=name, x=groupUsed, hue=groupUsed,legend=False, size=3, color="black")
                    sns.violinplot(df_copy, y=name, x=groupUsed, hue=groupUsed, palette=palette_copy, order=palette_copy.keys(), legend=False)
                    figure_title=os.path.join(STUDY_PATH,f"groups_compare_{name}_contrast{contrast}.png")
                    print(figure_title)
                    plt.savefig(figure_title)
                    plt.close()
                    display(Image(filename=figure_title))

                    # Effectuer un test t avec Pingouin
                    ttest_results = pg.ttest(df_copy[df_copy[groupUsed] == GP1][name],df_copy[df_copy[groupUsed] == GP2][name])
                    tvalue=ttest_results["T"].values[0]
                    pvalue=ttest_results["p-val"].values[0]
                    print(f"T-Test (oneside) : {tvalue:.3f} and p_value = {pvalue:.3f}")

def corr_surface_var_func(listeDesVariabesATester,listeCovar,theDF1,theDF2,STUDY_PATH,threshold_p=0.001,threshold_size=50,MyPalette=plt.get_cmap("Pastel1")):
    pd.option_context('mode.use_inf_as_na', True)
    for varc in listeDesVariabesATester:
        theDF1=theDF1.loc[theDF1.index.dropna()]
        common_index = theDF1.index.intersection(theDF2.index)
        common_index_on_var = theDF1.loc[common_index][varc].dropna().index
        term_group = FixedEffect(theDF1.loc[common_index_on_var][[varc]+listeCovar])
        model_group_age_educ = term_group
        
        contrast_group = theDF1.loc[common_index_on_var][varc]

        slm_group = SLM(
            model_group_age_educ,
            contrast_group,
            surf=inflated_combined,
            mask=mask,
            correction=["fdr", "rft"],
            two_tailed=True,
            cluster_threshold=threshold_p,
        )

        slm_group.fit(theDF2.loc[common_index_on_var].values)
       
        for contrast in [0,1] :
            if(len(slm_group.P['clus'][contrast])>0):
                figure_title=os.path.join(STUDY_PATH,f"correlation_CT_{varc}.jpg")
                labText=f"{varc}\n{len(common_index_on_var)} values"

                filtered_df = slm_group.P['clus'][contrast][(slm_group.P['clus'][contrast]['nverts'] > threshold_size) & (slm_group.P['clus'][contrast]['P'] < threshold_p)]
                if len(filtered_df)>0 :
                    max_cluster_nb=np.max(filtered_df["clusid"].values)
                    test = np.copy(slm_group.P["clusid"][contrast].reshape(-1))
                    test = np.where(test > max_cluster_nb, 0, test)
                    plot_hemispheres(inflated_left, inflated_right,test,label_text=[labText],color_range=(0,9),color_bar=True,
                                    cmap="tab10", embed_nb=True, size=(1400, 200),zoom=1.45, nan_color=(0.7, 0.7, 0.7, 1),
                                    cb__labelTextProperty={"fontSize": 12},transparent_bg=False,screenshot=True, offscreen=False,filename=figure_title)
                    print(figure_title)
                    display(Image(filename=figure_title))
                    print(f"{BLUE}***** \n clusters infos : \n***** \n {RESET} {filtered_df}")
                    df_tmp=slm_group.P['peak'][contrast]
                    print(f"{BLUE}***** \n peaks infos : \n***** \n {RESET} {df_tmp[(df_tmp['clusid'] <= max_cluster_nb)]}")
            
                    for clu in filtered_df["clusid"].values :
                        df_copy = theDF1.loc[common_index_on_var][[varc,'GROUPE','SOUS_GROUPE']+listeCovar].copy()
                        inds=np.where(slm_group.P['clusid'][contrast]==clu)[1]
                        name=f"cluster_{str(clu)}"
                        
                        print(f'{ansi_colors10[clu]} {BOLD} {name} {RESET}')
                        df_copy[name]=np.mean(theDF2.loc[common_index_on_var].values[:,inds],1)
                        regress_covar_func([name],listeCovar,df_copy,df_copy,False)
                        
                        # Faire une copie du dictionnaire palette
                        palette_copy = MyPalette.copy()

                        # Obtenir les groupes uniques présents dans df_copy
                        groups_present = df_copy["GROUPE"].unique()

                        # Supprimer les groupes non présents dans df_copy de la copie du dictionnaire palette
                        palette_copy = {group: color for group, color in palette_copy.items() if group in groups_present}

                        fig=sns.jointplot(df_copy,y=name,x=varc,hue="GROUPE", palette=palette_copy)

                        for diag in df_copy["GROUPE"].unique():
                            sns.regplot(x=varc, y=name, data=df_copy[df_copy["GROUPE"] == diag],scatter_kws={'s': 10},color=palette_copy[diag])
                            plt.gca().set_ylabel(f"Mean cortical thickness in {name}")  
                            #figure_title=os.path.join(STUDY_PATH,"results",f"corr_all_corr_{name}.png")    

                        plt.close(fig.fig)
                        print(os.path.join(STUDY_PATH,f"corr_all_corr_{name}.png"))
                        fig.savefig(os.path.join(STUDY_PATH,"corr_all_corr_{name}.png"))
                        display(Image(filename=os.path.join(STUDY_PATH,"corr_all_corr_{name}.png")))
                        
                        # Fit linear regression model
                        model = LinearRegression().fit(df_copy[varc].values.reshape(-1,1),df_copy[name].values.reshape(-1,1))

                        # Compute explained variance
                        #explained_variance = explained_variance_score(df_copy[name].values.reshape(-1,1), model.predict(df_copy[varc].values.reshape(-1,1),))

                        # Calculer le coefficient de corrélation (Pearson)
                        correlation_coefficient, p_val = pearsonr(df_copy[varc],df_copy[name])

                        # Calculer le coefficient de détermination (R²)
                        r_squared = correlation_coefficient ** 2
                        
                        print(f"Coefficient de corrélation (Pearson) : {correlation_coefficient:.3f} and p_value = {p_val:.3f}")
                        print(f"Coefficient de détermination aka Explained variance (R²) : {r_squared:.3f}")
       
def corr_sillons_var_func(listeDesSillons,listeDesVariabes,theDF1,theDF2,thres_sign=0.05,corr_thres=0.4):
    #df_sill=theDF1[theDF1.index.isin(etude1_index)][listeDesSillons]
    df_sill=theDF1.loc[:][listeDesSillons]
    df_scores=theDF2.loc[df_sill.index][listeDesVariabes]

    df_corr=df_sill.join(df_scores).corr()
    df_corr.drop(df_sill.columns, inplace=True)
    df_corr.drop(df_scores.columns, axis=1, inplace=True)

    test=df_sill.join(df_scores).dropna()
    print(f"Nombre de sujets : {len(test)}")
    if (len(test)>29):
        p_values = test.apply(lambda x: test.apply(lambda y: pearsonr(x, y)[1]))
    else :
        p_values = test.apply(lambda x: test.apply(lambda y: spearmanr(x, y)[1]))
    
    p_values.drop(df_sill.columns, inplace=True)
    p_values.drop(df_scores.columns, axis=1, inplace=True)
    
    #b_fdr,p_fdr=sm.stats.fdrcorrection(p_values,alpha=0.05,method='indep',is_sorted=False)  
    #p_values=p_fdr
    s = df_corr*(p_values<thres_sign)
    result = s[np.abs(s) > corr_thres].stack()
    result.index.names = ['Imaging data', 'Clinical data']    
    print(result)
    
#test=p_values.dropna().values.flatten().T
#b_fdr,p_fdr=sm.stats.fdrcorrection(test,alpha=0.001,is_sorted=False)

# Créer une matrice d'annotation avec des chaînes vides pour les valeurs nulles
#annot_str = np.where(df_corr*mask != 0, df_corr.values, '').astype(str)

    sns.heatmap(np.round(df_corr,1)*(p_values<thres_sign),vmin=-0.4, vmax=0.4,cbar_kws={'shrink': 0.5},annot=True,cmap="coolwarm")
    #plt.legend(loc='upper left')
    plt.title(f"correlation for p_unc < {thres_sign}")
    plt.xlabel("Sulcus")
    plt.ylabel("Clinical data ")
    
    for ind in result.index:
        plt.figure()
        sns.jointplot(x=ind[0], y=ind[1], data=test,kind="reg")
        plt.show()
    
    plt.figure()