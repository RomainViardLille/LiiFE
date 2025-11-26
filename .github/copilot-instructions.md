# Copilot Instructions for AI Agents

## Vue d'ensemble du projet
Ce dépôt contient de nombreux notebooks Jupyter et scripts Python pour l'analyse d'images médicales, la segmentation, la manipulation de fichiers NIfTI/MAT, et l'extraction de métriques. Les workflows sont orientés recherche, avec des traitements par lots sur des dossiers structurés par sujet et protocole.

## Structure et conventions principales
- Les notebooks sont utilisés pour l'exploration interactive, la conversion de formats (DICOM, NIfTI, MAT), et l'analyse statistique.
- Les scripts Python (ex : `BundlesAutoSimilarityExtraction.py`, `ExtractRadiomixV1.py`) réalisent des traitements batch ou des extractions de features.
- Les chemins d'accès sont souvent codés en dur et pointent vers des serveurs NAS ou des dossiers locaux structurés par projet/sujet.
- Les conventions de nommage des fichiers suivent généralement le schéma : `<projet>/<protocole>/<sujet>/<type_de_fichier>`.

## Dépendances et intégrations
- Utilisation intensive de `nibabel`, `numpy`, `scipy.io`, `glob`, `os` pour la manipulation de données médicales.
- Certains notebooks/scripts utilisent `dicom2nifti` pour la conversion DICOM → NIfTI.
- Les analyses de similarité ou de segmentation utilisent des modules comme `scipy.spatial.distance` (ex : Dice coefficient).
- Les dépendances doivent être installées dans l'environnement Python actif (voir les imports en tête de chaque notebook/script).

## Workflows critiques
- **Conversion MAT/NIfTI** : voir les cellules avec `scipy.io.matlab.loadmat` et `nib.save` pour transformer des matrices MATLAB en images NIfTI.
- **Traitement batch** : les boucles sur `glob.iglob` ou `glob.glob` permettent d'appliquer des traitements à tous les sujets d'un protocole.
- **Comparaison de segmentations** : extraction de labels spécifiques et calcul de Dice via `scipy.spatial.distance.dice`.
- **Modification d'affines** : création d'objets NIfTI avec des matrices d'affinité personnalisées pour corriger l'orientation ou la résolution.

## Exemples de patterns
- Conversion et sauvegarde NIfTI :
  ```python
  img = nib.Nifti1Image(data, affine)
  nib.save(img, output_path)
  ```
- Traitement batch sur fichiers :
  ```python
  for it in glob.iglob('/chemin/vers/dossier/*.mat'):
      # traitement
  ```
- Calcul de Dice entre deux segmentations :
  ```python
  dice_coef = 1 - dice(label1.flatten(), label2.flatten())
  ```

## Points d'attention
- Les chemins d'accès doivent être adaptés si le code est déplacé ou exécuté sur un autre système.
- Certains scripts supposent l'existence de fichiers ou dossiers spécifiques (pas de gestion d'erreur avancée).
- Les notebooks servent de documentation vivante : se référer aux cellules markdown pour le contexte scientifique ou méthodologique.

## Fichiers/répertoires clés
- Notebooks Jupyter : exploration, prototypage, documentation.
- Scripts Python : automatisation, traitements batch.
- Données d'entrée/sortie : images NIfTI, fichiers MAT, CSV, etc.

Pour toute modification, respecter les patterns existants et documenter les nouveaux workflows dans un notebook ou un README dédié.