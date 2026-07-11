import SimpleITK as sitk
import numpy as np
from pathlib import Path


def compute_bounding_box(mask):
    coords = np.argwhere(mask)

    if coords.shape[0] == 0:
        return None

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    return {
        "min": min_coords.tolist(),
        "max": max_coords.tolist()
    }


def compute_centroid(mask):
    coords = np.argwhere(mask)

    if coords.shape[0] == 0:
        return None

    centroid = coords.mean(axis=0)

    return centroid.tolist()


# =====================================
# Load segmentation file
# =====================================

file_path = r"C:\Users\2362519\Downloads\ImageCAS-STACOM2025-02-10-2025\ImageCAS-STACOM2025-02-10-2025\segmentations\190.img.nii.gz"

img = sitk.ReadImage(file_path)

# Full segmentation
mask = sitk.GetArrayFromImage(img)

# =====================================
# LABEL DEFINITIONS
# =====================================
#
# 0  = Background
# 1  = Myocardium
# 2  = LA
# 3  = LV
# 4  = RA
# 5  = RV
# 6  = Aorta
# 7  = PA
# 8  = LAA
# 9  = Coronary
# 10 = PV
#
# =====================================

LA_LABEL = 2

# Create true LA mask only
la_mask = (mask == LA_LABEL)

# =====================================
# Basic Information
# =====================================

file_name = Path(file_path).name

mask_shape = mask.shape

spacing = img.GetSpacing()  # (x, y, z)

spacing_x, spacing_y, spacing_z = spacing

# =====================================
# Label Statistics
# =====================================

# print("\n===== LABEL COUNTS =====")

# unique_labels, counts = np.unique(mask, return_counts=True)

# for label, count in zip(unique_labels, counts):
#     print(f"Label {label}: {count}")

# print("========================\n")

# =====================================
# LA Voxels
# =====================================

la_voxels = np.count_nonzero(la_mask)

# =====================================
# LA Volume
# =====================================

voxel_volume_mm3 = spacing_x * spacing_y * spacing_z

la_volume_mm3 = la_voxels * voxel_volume_mm3

la_volume_ml = la_volume_mm3 / 1000

# =====================================
# LA Bounding Box
# =====================================

bbox = compute_bounding_box(la_mask)

# =====================================
# AP / ML / SI Dimensions
# =====================================

if bbox is not None:

    min_z, min_y, min_x = bbox["min"]
    max_z, max_y, max_x = bbox["max"]

    # NumPy order = (z, y, x)

    si_diameter_mm = (max_z - min_z + 1) * spacing_z
    ap_diameter_mm = (max_y - min_y + 1) * spacing_y
    ml_diameter_mm = (max_x - min_x + 1) * spacing_x

else:

    si_diameter_mm = None
    ap_diameter_mm = None
    ml_diameter_mm = None

# =====================================
# LA Centroid
# =====================================

centroid = compute_centroid(la_mask)

# =====================================
# Results
# =====================================

print("===== LEFT ATRIUM (LABEL = 2) =====")

print(f"1. File Name           : {file_name}")

print(f"2. Mask Shape          : {mask_shape}")

print(f"3. Spacing (x,y,z)     : {spacing}")

print(f"4. LA Voxels           : {la_voxels}")

print(f"5. LA Volume (mL)      : {la_volume_ml:.2f}")

if bbox is not None:
    print(f"6. Bounding Box Min    : {bbox['min']}")
    print(f"   Bounding Box Max    : {bbox['max']}")

if ap_diameter_mm is not None:
    print(f"7. AP Diameter (mm)    : {ap_diameter_mm:.2f}")
    print(f"   ML Diameter (mm)    : {ml_diameter_mm:.2f}")
    print(f"   SI Diameter (mm)    : {si_diameter_mm:.2f}")

if centroid is not None:
    print(f"8. Centroid (z,y,x)    : {[round(v, 2) for v in centroid]}")

print("===================================")

#PCA Analysis:
from sklearn.decomposition import PCA
import numpy as np


def compute_shape_features(la_mask, spacing):

    spacing_x, spacing_y, spacing_z = spacing

    coords = np.argwhere(la_mask)

    coords_mm = np.column_stack([
        coords[:, 2] * spacing_x,
        coords[:, 1] * spacing_y,
        coords[:, 0] * spacing_z
    ])

    pca = PCA(n_components=3)
    projected = pca.fit_transform(coords_mm)

    major_axis_length_mm = (
        projected[:, 0].max() -
        projected[:, 0].min()
    )

    minor_axis_length_mm = (
        projected[:, 1].max() -
        projected[:, 1].min()
    )

    least_axis_length_mm = (
        projected[:, 2].max() -
        projected[:, 2].min()
    )

    elongation = (
        minor_axis_length_mm /
        major_axis_length_mm
    )

    flatness = (
        least_axis_length_mm /
        major_axis_length_mm
    )

    return {
        "major_axis_length_mm": major_axis_length_mm,
        "minor_axis_length_mm": minor_axis_length_mm,
        "least_axis_length_mm": least_axis_length_mm,
        "elongation": elongation,
        "flatness": flatness
    }

shape_features = compute_shape_features(
    la_mask,
    spacing
)

# =====================================
# Categories
# =====================================

elongation = shape_features["elongation"]
flatness = shape_features["flatness"]


# LA Volume Category

if la_volume_ml < 75:
    la_volume_category = "normal"

elif la_volume_ml < 100:
    la_volume_category = "mildly_enlarged"

elif la_volume_ml < 130:
    la_volume_category = "moderately_enlarged"

else:
    la_volume_category = "severely_enlarged"


# Shape Category

if elongation >= 0.85 and flatness >= 0.85:
    shape_category = "spherical"

elif elongation >= 0.75 and flatness < 0.60:
    shape_category = "wide_and_flat"

elif elongation < 0.65:
    shape_category = "elongated"

else:
    shape_category = "intermediate"

print("\n===== SHAPE FEATURES =====")

print(
    f"Major Axis Length (mm) : "
    f"{shape_features['major_axis_length_mm']:.2f}"
)

print(
    f"Minor Axis Length (mm) : "
    f"{shape_features['minor_axis_length_mm']:.2f}"
)

print(
    f"Least Axis Length (mm) : "
    f"{shape_features['least_axis_length_mm']:.2f}"
)

print(
    f"Elongation             : "
    f"{shape_features['elongation']:.3f}"
)

print(
    f"Flatness               : "
    f"{shape_features['flatness']:.3f}"
)

print(
    f"LA Volume Category     : "
    f"{la_volume_category}"
)

print(
    f"Shape Category         : "
    f"{shape_category}"
)

print("==========================")

from skimage.measure import marching_cubes, mesh_surface_area
import numpy as np


def compute_sphericity(la_mask, spacing):
    """
    Computes sphericity of a binary LA mask.

    Parameters
    ----------
    la_mask : numpy.ndarray
        Binary mask (True/False)

    spacing : tuple
        (spacing_x, spacing_y, spacing_z)

    Returns
    -------
    dict
    """

    spacing_x, spacing_y, spacing_z = spacing

    # Volume
    voxel_volume_mm3 = spacing_x * spacing_y * spacing_z

    volume_mm3 = (
        np.count_nonzero(la_mask)
        * voxel_volume_mm3
    )

    # marching_cubes expects spacing in array order
    # la_mask.shape = (z, y, x)

    verts, faces, _, _ = marching_cubes(
        la_mask.astype(np.uint8),
        level=0.5,
        spacing=(spacing_z, spacing_y, spacing_x)
    )

    surface_area_mm2 = mesh_surface_area(
        verts,
        faces
    )

    sphericity = (
        (np.pi ** (1 / 3))
        * ((6 * volume_mm3) ** (2 / 3))
        / surface_area_mm2
    )

    return {
        "volume_mm3": volume_mm3,
        "surface_area_mm2": surface_area_mm2,
        "sphericity": sphericity
    }

sphericity_info = compute_sphericity(
    la_mask,
    spacing
)
# =====================================
# Sphericity Category
# =====================================

sphericity = sphericity_info["sphericity"]

if sphericity >= 0.90:
    sphericity_category = "very_high"

elif sphericity >= 0.80:
    sphericity_category = "high"

elif sphericity >= 0.70:
    sphericity_category = "moderate"

else:
    sphericity_category = "low"

print("\n===== SPHERICITY =====")

print(
    f"Surface Area (mm²) : "
    f"{sphericity_info['surface_area_mm2']:.2f}"
)

print(
    f"Sphericity          : "
    f"{sphericity_info['sphericity']:.3f}"
)

print(
    f"Sphericity Category : "
    f"{sphericity_category}"
)


print("======================")

from scipy import ndimage
import numpy as np


def analyze_pv(mask, min_size = 1000):

    PV_LABEL = 10

    pv_mask = (mask == PV_LABEL)

    pv_voxels = np.count_nonzero(pv_mask)

    labeled_pv, num_components = ndimage.label(
        pv_mask
    )

    component_sizes = []
    valid_components = []

    for component_id in range(1, num_components + 1):

        size = np.sum(
            labeled_pv == component_id
        )

        component_sizes.append(size)
        
        if size >= min_size:
            valid_components.append(size)

    
    valid_components = sorted(
        valid_components,
        reverse=True
    )


    component_sizes = sorted(
        component_sizes,
        reverse=True
    )

    return {
        # "pv_voxels": pv_voxels,
        # "pv_number": num_components,
        # "component_sizes": component_sizes

        
        "raw_components": num_components,
        "raw_component_sizes": component_sizes,
        "pv_number": len(valid_components),
        "valid_component_sizes": valid_components

    }

pv_info = analyze_pv(mask)
print(pv_info)

# print("\n===== PV ANALYSIS =====")

# print(
#     f"PV Voxels : "
#     f"{pv_info['pv_voxels']}"
# )

# print(
#     f"PV Components : "
#     f"{pv_info['pv_number']}"
# )

# print(
#     f"Component Sizes : "
#     f"{pv_info['component_sizes']}"
# )

# print("=======================")