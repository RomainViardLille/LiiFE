import numpy as np

def extract_translation_rotation(transform_matrix):
    """
    Extrait les translations et rotations à partir d'une matrice de transformation 4x4.

    Args:
        transform_matrix (numpy.ndarray): Matrice de transformation 4x4.

    Returns:
        tuple: (translations, rotations)
            translations (numpy.ndarray): Vecteur de translation [tx, ty, tz].
            rotations (numpy.ndarray): Angles de rotation [rx, ry, rz] en radians.
    """
    # Vérifier que la matrice est bien 4x4
    if transform_matrix.shape != (4, 4):
        raise ValueError("La matrice de transformation doit être de taille 4x4.")

    # Extraire la sous-matrice de rotation 3x3
    rotation_matrix = transform_matrix[0:3, 0:3]

    # Extraire le vecteur de translation
    translations = transform_matrix[0:3, 3]

    # Calculer les angles de rotation à partir de la matrice de rotation
    # Angle de rotation autour de l'axe X (rx)
    rx = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    # Angle de rotation autour de l'axe Y (ry)
    ry = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))

    # Angle de rotation autour de l'axe Z (rz)
    rz = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convertir les angles en degrés si nécessaire
    rotations = np.array([rx, ry, rz])

    return translations, rotations

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de matrice de transformation 4x4 obtenue après un recalage 9DoF avec FLIRT
    transform_matrix = np.array([
        [0.999, -0.010, 0.040, 1.5],
        [0.010, 0.999, -0.030, -2.0],
        [-0.040, 0.030, 0.999, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    translations, rotations = extract_translation_rotation(transform_matrix)

    print("Translations (tx, ty, tz):", translations)
    print("Rotations (rx, ry, rz) en radians:", rotations)
    print("Rotations (rx, ry, rz) en degrés:", np.degrees(rotations))
