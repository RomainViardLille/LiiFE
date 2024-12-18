import matplotlib.pyplot as plt
import numpy as np

import trimesh
import os
import nibabel as nib
import sys
from trimesh.curvature import (
    discrete_gaussian_curvature_measure,
    discrete_mean_curvature_measure,
    sphere_ball_intersection,
)

def process_surface_file(file_path, radius=0.1):
    if os.path.exists(file_path):
        surf = nib.load(file_path)
        vertices = surf.darrays[0].data
        triangles = surf.darrays[1].data

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        gauss = np.array(
            discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius)
            / sphere_ball_intersection(1, radius)
        )

        for i in range(gauss.shape[0]):
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            output_file = os.path.join(os.path.dirname(file_path), f'{name}_gaussCurv_{radius}.npy')
            np.save(output_file, gauss)
        
        return np.mean(np.abs(gauss))

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python computeGausCurvature.py <file_path> ")
        sys.exit(1)

    file_path = sys.argv[1]
    radius = float(sys.argv[2])
    print(process_surface_file(file_path,radius))