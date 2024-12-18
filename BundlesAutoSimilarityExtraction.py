from dipy.io.streamline import load_tractogram, save_tractogram
import numpy as np
from dipy.segment.bundles import bundle_shape_similarity
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import set_number_of_points
from dipy.viz import window, actor
from time import sleep
import os,sys

def show_both_bundles(bundles, colors=None, show=True, fname=None):
    """
    Display both bundles in a 3D scene.

    Parameters:
    - bundles: list of bundles to display.
    - colors: list of colors for each bundle. If None, default colors will be used.
    - show: boolean value indicating whether to show the scene. Default is True.
    - fname: file name to save the scene as an image. If None, the scene will not be saved.

    Returns:
    None
    """

    scene = window.Scene()
    scene.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines_actor = actor.streamtube(bundle, color, linewidth=0.3)
        lines_actor.RotateX(-90)
        lines_actor.RotateZ(90)
        scene.add(lines_actor)
    if show:
        window.show(scene)
    if fname is not None:
        sleep(1)
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))
        
print("START2")
DIR="/NAS/dumbo/protocoles/CQTracto/4630660/atlas/atlas/pop_average/"
threshold = 15
rng = np.random.RandomState()
clust_thr = [0]
my_trk_file_path2 = sys.argv[1]

parts = my_trk_file_path2.split("dwi/", 1)  # Le second argument '1' indique de diviser en deux parties au maximum
extracted_path = parts[0]

file_name_with_ext = os.path.basename(my_trk_file_path2)
file_name, file_ext = os.path.splitext(file_name_with_ext)
fx=f"{file_name.split('_')[-2]}_{file_name.split('_')[-1]}.trk"
ref_path1=os.path.join(DIR,fx)

print(my_trk_file_path2,ref_path1)

ref_trk=load_tractogram(ref_path1, 'same')


my_b0_file_path2 = my_trk_file_path2.replace("/auto_tract","").split("dwi_desc")[0]+"dwi_desc-b0.nii.gz"
my_trk2 = load_tractogram(my_trk_file_path2,my_b0_file_path2)

cb_subj1 = set_number_of_points(ref_trk.streamlines, 20)
cb_subj2 = set_number_of_points(my_trk2.streamlines, 20)

srr = StreamlineLinearRegistration()
print("début du recalage")
srm = srr.optimize(static=cb_subj1, moving=cb_subj2)
print("recalage terminé")
cb_subj2_aligned = srm.transform(cb_subj2)

ba_score = bundle_shape_similarity(cb_subj1, cb_subj2_aligned, rng, clust_thr, threshold)

show_both_bundles([cb_subj1, cb_subj2],
                  colors=[window.colors.orange, window.colors.red],
                  show=False,
                  fname=os.path.join(extracted_path,"QC","bundlesS",f"before_registration_{file_name}.png"))

show_both_bundles([cb_subj1, cb_subj2_aligned],
                  colors=[window.colors.orange, window.colors.red],
                  show=False,
                  fname=os.path.join(extracted_path,"QC","bundlesS",f"after_registration_{file_name}.png"))


# Ouverture du fichier en mode écriture
with open(os.path.join(extracted_path,"QC","bundlesS",f"{file_name}.txt"), "w") as file:
    # Écriture de la valeur de la variable dans le fichier
    file.write(str(ba_score))
