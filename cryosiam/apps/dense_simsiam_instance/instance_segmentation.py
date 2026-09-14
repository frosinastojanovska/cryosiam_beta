import itk
import numpy as np
import scipy.ndimage as ndi
from skimage.measure import label
from skimage.filters import gaussian
from skimage.segmentation import relabel_sequential
from skimage.segmentation import watershed, join_segmentations

try:
    import elf.segmentation.multicut as mc
    import elf.segmentation.features as feats
except ImportError:
    mc = None
    feats = None


def apply_bilateral_filter(image, domain_sigma=1, range_sigma=0.2):
    img = itk.image_from_array(image)
    filtered = itk.bilateral_image_filter(img, DomainSigma=domain_sigma, RangeSigma=range_sigma)
    return itk.array_from_image(filtered)


def find_markers(distances, threshold_min=1, threshold_max=3):
    if threshold_min == threshold_max:
        return label(distances >= threshold_min)
    markers = np.zeros(distances.shape, dtype=np.int64)
    initial_separation = label(distances >= threshold_min)
    for thresh in np.arange(threshold_min + 0.5, threshold_max + 0.5, 0.5):
        current_separation = label(distances >= thresh)
        mask = initial_separation[(current_separation == 0) & (initial_separation != 0)]
        second_mask = initial_separation[(current_separation != 0) & (initial_separation != 0)]
        ids = np.setdiff1d(mask, second_mask)
        if ids.shape[0] >= 1:
            selection = np.isin(initial_separation, ids)
            markers[selection] = initial_separation[selection].copy()
        markers, _, _ = relabel_sequential(markers, offset=1)
        initial_separation, _, _ = relabel_sequential(current_separation, offset=np.max(markers) + 1)
    markers[initial_separation > 0] = initial_separation[initial_separation > 0]
    markers, _, _ = relabel_sequential(markers, offset=1)
    return markers


def instance_segmentation(foreground, distances, boundaries, threshold_min=0, threshold_max=5,
                          boundary_bias=.45, assignment_threshold=.4, distance_type=0, postprocessing=True):
    distances = gaussian(distances, sigma=0.5)
    boundaries = apply_bilateral_filter(boundaries, domain_sigma=1, range_sigma=0.1)
    if assignment_threshold < 1:
        foreground = gaussian(foreground, sigma=0.5)
    distances[distances < 0] = 0
    distances[foreground < assignment_threshold] = 0
    if distance_type == 1:
        distances2 = np.clip(ndi.distance_transform_edt(foreground >= assignment_threshold), a_min=0, a_max=5)
        distances2 = gaussian(distances2, sigma=0.5)
        distances = (distances + distances2) / 2
    elif distance_type == 2:
        distances2 = np.clip(ndi.distance_transform_edt(foreground >= assignment_threshold), a_min=0, a_max=5)
        distances2 = gaussian(distances2, sigma=0.5)
        distances = distances2
    markers = find_markers(distances, threshold_min, threshold_max).astype(np.uint32)
    # watershed_seg = ws.watershed(-distances.astype(np.float32), seeds=markers)[0]
    watershed_seg = watershed(-distances, markers=markers, mask=foreground >= assignment_threshold).astype(np.uint32)
    if not postprocessing:
        return watershed_seg
    if boundary_bias == 1:
        return watershed_seg
    # compute the region adjacency graph
    rag = feats.compute_rag(watershed_seg)
    # compute the edge costs
    # features = feats.compute_boundary_mean_and_length(rag, boundaries)
    # costs, sizes = features[:, 0], features[:, 1]
    # # we choose a boundary bias smaller than 0.5 in order to decrease the degree of over segmentation
    # costs = mc.transform_probabilities_to_costs(costs, edge_sizes=sizes, beta=boundary_bias)
    #----probs = feats.compute_boundary_features(rag, boundaries)[:, 0]
    #---- python-elf api > 0.9
    probs = feats.compute_boundary_features(rag, watershed_seg, boundaries)[:, 0]
    costs = mc.transform_probabilities_to_costs(probs, beta=boundary_bias)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    #---- segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    #---- python-elf api > 0.9
    segmentation = feats.project_node_labels_to_pixels(rag, watershed_seg, node_labels)
    watershed_seg[segmentation > 0] = 0
    segmentation = join_segmentations(segmentation, watershed_seg)
    return segmentation
