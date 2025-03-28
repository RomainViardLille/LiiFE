# -*- coding: utf-8 -*-
"""
====================================================================
Full Example Pipeline for Statistical Shape Modeling with ShapeWorks
====================================================================
This example is set to serve as a test case for new ShapeWorks users, and each
step is explained in the shapeworks including the pre-processing, the 
optimization and, the post ShapeWorks visualization.

First import the necessary modules
"""
import os
from GroomUtils import *
from OptimizeUtils import *
from AnalyzeUtils import *
import CommonUtils

    """
    Les fichiers .nrrd (n=10 pour le moment) sont ici /home/neuroa/Downloads/ShapeWorks-v5.5.0-linux/Examples/Python/Data/Test_HomeData/Output/segmentations/
    """
    chdir('/NAS/deathrow/protocoles/predistim/DualSyndrome_QD/Shape/shapeworks/ShapeWorks-v5.5.0-linux/Examples/Python')
    outputDirectory = "Data/TestHome_Data/Output"

    fileList = sorted(glob.glob(outputDirectory + "/segmentations/*.nrrd"))
    
    if args.tiny_test:
        args.use_single_scale = 1
        fileList = fileList[0:10]

    """
    ## GROOM : Data Pre-processing 
    For the unprepped data the first few steps are 
    -- Isotropic resampling
    -- Center
    -- Padding
    -- Center of Mass Alignment
    -- Rigid Alignment
    -- Largest Bounding Box and Cropping 
    For a detailed explanation of grooming steps see: /docs/workflow/groom.md
    """

    print("\nStep 2. Groom - Data Pre-processing\n")
    if int(args.interactive) != 0:
        input("Press Enter to continue")

    groomDir = outputDirectory + 'groomed/'
    if not os.path.exists(groomDir):
        os.makedirs(groomDir)


    if args.start_with_image_and_segmentation_data:
        print("\n\n************************ WARNING ************************")
        print("'start_with_image_and_segmentation_data' tag was used \nbut Ellipsoid data set does not have images.")
        print("Continuing to run use case with segmentations only.")
        print("*********************************************************\n\n")

    if int(args.start_with_prepped_data) == 1:
        dtFiles = sorted(glob.glob(outputDirectory + '/groomed/distance_transforms/*.nrrd'))
    else:
        """Apply isotropic resampling"""
        resampledFiles = applyIsotropicResampling(groomDir + "resampled/segmentations", fileList)

        """Apply centering"""
        centeredFiles = center(groomDir + "centered/segmentations", resampledFiles)

        """Apply padding"""
        paddedFiles = applyPadding(groomDir + "padded/segmentations", centeredFiles, 10)

        """Apply center of mass alignment"""
        comFiles = applyCOMAlignment(groomDir + "com_aligned/segmentations", paddedFiles, None)

        """Apply rigid alignment"""
        rigidFiles = applyRigidAlignment(groomDir + "aligned/segmentations", comFiles, None, comFiles[0])

        """Compute largest bounding box and apply cropping"""
        croppedFiles = applyCropping(groomDir + "cropped/segmentations", rigidFiles, groomDir + "aligned/segmentations/*.aligned.nrrd")

        """
        We convert the scans to distance transforms, this step is common for both the 
        prepped as well as unprepped data, just provide correct filenames.
        """

        print("\nStep 3. Groom - Convert to distance transforms\n")
        if int(args.interactive) != 0:
            input("Press Enter to continue")

        dtFiles = applyDistanceTransforms(groomDir, croppedFiles)


    """
    ## OPTIMIZE : Particle Based Optimization

    Now that we have the distance transform representation of data we create 
    the parameter files for the shapeworks particle optimization routine.
    For more details on the plethora of parameters for shapeworks please refer 
    to docs/workflow/optimze.md
    First we need to create a dictionary for all the parameters required by this
    optimization routine
    """

    print("\nStep 4. Optimize - Particle Based Optimization\n")
    if int(args.interactive) != 0:
        input("Press Enter to continue")

    pointDir = outputDirectory + 'shape_models/'
    if not os.path.exists(pointDir):
        os.makedirs(pointDir)

    parameterDictionary = {
        "number_of_particles": 128,
        "use_normals": 0,
        "normal_weight": 0.0,
        "checkpointing_interval": 200,
        "keep_checkpoints": 0,
        "iterations_per_split": 1000,
        "optimization_iterations": 1000,
        "starting_regularization": 100,
        "ending_regularization": 0.05,
        "recompute_regularization_interval": 2,
        "domains_per_shape": 1,
        "domain_type": 'mesh',
        "relative_weighting": 10,
        "initial_relative_weighting": 0.05,
        "procrustes_interval": 1,
        "procrustes_scaling": 1,
        "save_init_splits": 0,
        "verbosity": 2
    }

    if args.tiny_test:
        parameterDictionary["number_of_particles"] = 32
        parameterDictionary["optimization_iterations"] = 25

    if not args.use_single_scale:
        parameterDictionary["use_shape_statistics_after"] = 32

    """
    Now we execute a single scale particle optimization function.
    """
    [localPointFiles, worldPointFiles] = runShapeWorksOptimize(pointDir, dtFiles, parameterDictionary)

    if args.tiny_test:
        print("Done with tiny test")
        exit()
          
    """
    ## ANALYZE : Shape Analysis and Visualization

    Shapeworks yields relatively sparse correspondence models that may be inadequate to reconstruct 
    thin structures and high curvature regions of the underlying anatomical surfaces. 
    However, for many applications, we require a denser correspondence model, for example, 
    to construct better surface meshes, make more detailed measurements, or conduct biomechanical 
    or other simulations on mesh surfaces. One option for denser modeling is 
    to increase the number of particles per shape sample. However, this approach necessarily 
    increases the computational overhead, especially when modeling large clinical cohorts.

    Here we adopt a template-deformation approach to establish an inter-sample dense surface correspondence, 
    given a sparse set of optimized particles. To avoid introducing bias due to the template choice, we developed
    an unbiased framework for template mesh construction. The dense template mesh is then constructed 
    by triangulating the isosurface of the mean distance transform. This unbiased strategy will preserve 
    the topology of the desired anatomy  by taking into account the shape population of interest. 
    In order to recover a sample-specific surface mesh, a warping function is constructed using the 
    sample-level particle system and the mean/template particle system as control points. 
    This warping function is then used to deform the template dense mesh to the sample space.
    """
    
    print("\nStep 5. Analysis - Launch ShapeWorksStudio - sparse correspondence model.\n")
    if args.interactive != 0:
        input("Press Enter to continue")

    launchShapeWorksStudio(pointDir, dtFiles, localPointFiles, worldPointFiles)
