import os
import argparse

import torch
import imageio
import numpy as np

from tqdm import tqdm
from PIL import Image

import modules.midas.utils as utils

import pipeline
import metrics
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import matplotlib.patches as patches_plt
from utils_eval import compute_ls_solution
import cv2

def get_superpixel(rgb_img, depth_img_inv, sparse_depth):
    # Perform superpixel segmentation on RGB image
    segments_slic = slic(rgb_img, n_segments=5, compactness=40, convert2lab=True, sigma=2)
    
    # Get number of superpixels
    num_superpixels = np.max(segments_slic) + 1
    
    # Initialize array to store depth values for each superpixel patch
    depth_patches_inv = []; depth_sparse_patches = []
    mask_array = np.zeros_like(depth_img_inv)
    
    # Iterate over each pixel in the depth image and store depth values in corresponding superpixel patch
    for segment in range(num_superpixels):
        mask = segments_slic == segment
        
        mask_array += mask.astype(np.uint8)
        
        depth_patch = depth_img_inv * mask
        depth_gt_patch = sparse_depth * mask
        
        depth_patches_inv.append(depth_patch)
        depth_sparse_patches.append(depth_gt_patch)
    
    return mark_boundaries(rgb_img, segments_slic), depth_patches_inv, depth_sparse_patches, mask_array

def get_ls_solution(depth_infer, input_sparse_depth, validity_map, min_pred, max_pred, max_depth, min_depth, mask, target_depth):

    input_sparse_depth_valid = (input_sparse_depth < max_depth) * (input_sparse_depth > min_depth)
    
    if validity_map is not None:
        input_sparse_depth_valid *= validity_map.astype(bool)

    input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
    input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
    input_sparse_depth = 1.0 / input_sparse_depth

    scaled_depth, scale_ls, shift_ls = compute_ls_solution(depth_infer, input_sparse_depth, input_sparse_depth_valid, min_pred, max_pred)
    
    error_w_int_depth_ls = metrics.ErrorMetrics()
    error_w_int_depth_ls.compute(scaled_depth, target_depth, mask.astype(bool))
    rmse_ls = error_w_int_depth_ls.rmse
    return rmse_ls, scale_ls, shift_ls

def visualize_patches(image, patches, sparse_points):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot sparse points
    sparse_x = [point[0] for point in sparse_points]
    sparse_y = [point[1] for point in sparse_points]
    ax.plot(sparse_x, sparse_y, 'ro', markersize=3)

    # Plot patches
    for patch in patches:
        patch_rect = patches_plt.Rectangle((patch[0], patch[1]), patch[2] - patch[0], patch[3] - patch[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(patch_rect)

    plt.show()
    
def divide_image_into_patches(image, patch_size, sparse_points):
    patches = []

    # Define the grid size
    grid_rows = image.shape[0] // patch_size
    grid_cols = image.shape[1] // patch_size

    u, v = np.nonzero(sparse_points)
    
    # Iterate over each grid cell
    for row in range(grid_rows):
        for col in range(grid_cols):
            patch_start_x = col * patch_size
            patch_start_y = row * patch_size
            patch_end_x = patch_start_x + patch_size
            patch_end_y = patch_start_y + patch_size

            # Check if any sparse point lies within the patch
            point_inside_patch = False
            for i in range(len(u)):
                point_x = u[i]
                point_y = v[i]
                if (patch_start_x <= point_x < patch_end_x) and (patch_start_y <= point_y < patch_end_y):
                    point_inside_patch = True
                    break

            # If a point is inside the patch, mark it as valid
            if point_inside_patch:
                patches.append((patch_start_x, patch_start_y, patch_end_x, patch_end_y))

    return patches

def reconstruct_depth_image(depth_patches, mask_array):
    # Sum up all depth patches while avoiding overlapping regions
    reconstructed_depth = sum(depth_patches)
    # Divide by the mask array to average the values where regions overlap
    reconstructed_depth /= np.where(mask_array > 0, mask_array, 1)
    
    return reconstructed_depth

def visualize_sparse_depth(input_sparse_depth, input_image):
    # if input sparse depth u,v non zero values plot them on input image
    if np.any(input_sparse_depth):
        # get non zero values
        u, v = np.nonzero(input_sparse_depth)
        # plot them on input image
        for i in range(len(u)):
            cv2.circle(input_image, (v[i], u[i]), 1, (0, 255, 0), -1)
    cv2.imshow("sparse depth", input_image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def visualize_depth_img(gt_img_cpu, viz_bound, infer_depth_patches, sparse_depth_patches, mask_array):
    reconstructed_depth = reconstruct_depth_image(infer_depth_patches, mask_array)
    reconstructed_gt_depth = reconstruct_depth_image(sparse_depth_patches, mask_array)
        
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.imshow(gt_img_cpu)
    ax0.set_title('Original Image')

    ax1.imshow(viz_bound)
    ax1.set_title('Superpixel Segmentation')
        
    ax2.imshow(reconstructed_depth)
    ax2.set_title('Reconstructed Depth')
        
    ax3.imshow(reconstructed_gt_depth)
    ax3.set_title('Reconstructed GT Depth')
    
    plt.show()
    return reconstructed_depth, reconstructed_gt_depth

def scale_patches(infer_depth_patches_inv, sparse_depth_patches, min_pred, max_pred, max_depth, min_depth):
    ## for each patch, compute the scale and shift values
    scaled_inv_patches = []
    #gt_depth_patches_inv_copy = gt_depth_patches_inv.copy()
    for idx in range(len(infer_depth_patches_inv)):
        input_sparse_depth_valid = (sparse_depth_patches[idx] < max_depth) * (sparse_depth_patches[idx] > min_depth)
        scaled_depth, scale_ls, shift_ls = compute_ls_solution(infer_depth_patches_inv[idx], 
                                                               sparse_depth_patches[idx], 
                                                               input_sparse_depth_valid, 
                                                               min_pred, max_pred)
        
        # scaled_inv_patch, scale, shift = scale_up(infer_depth_patches_inv[idx], 
        #                                             sparse_depth_patches[idx],
        #                                             min_pred, max_pred)
        scaled_inv_patches.append(scaled_depth)
        #print(scale, shift)
    return scaled_inv_patches

def evaluate(dataset_path, depth_predictor, nsamples, sml_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # ranges for VOID
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 8.0

    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    # get inputs
    with open(f"{dataset_path}/test_image.txt") as f: 
        test_image_list = [line.rstrip() for line in f]
        
    # initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

    # iterate through inputs list
    #for i in tqdm(range(len(test_image_list))):
    for i in tqdm(range(0,1,1)):   
        # Image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)

        # Sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0
        
        #patches = divide_image_into_patches(input_image, 43, input_sparse_depth)
        #visualize_patches(input_image, patches, input_sparse_depth)
        ## sparse depth validity map
        # validity_map_fp = input_image_fp.replace("image", "validity_map")
        # validity_map = np.array(Image.open(validity_map_fp), dtype=np.float32)
        # assert(np.all(np.unique(validity_map) == [0, 256]))
        # validity_map[validity_map > 0] = 1
        validity_map = None

        # target (ground truth) depth
        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0

        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth
        target_depth = 1.0 / target_depth

        depth_infer_inv = method.infer_depth(input_image)
        viz_bound, infer_depth_patches_inv, sparse_depth_patches, mask_array = get_superpixel(input_image, depth_infer_inv, input_sparse_depth)
        
        
        viz_image = input_image.copy()
        visualize_sparse_depth(input_sparse_depth, viz_image)
        
        scaled_patches = scale_patches(infer_depth_patches_inv, sparse_depth_patches, min_pred, max_pred, max_depth, min_depth)
        visualize_depth_img(viz_image, viz_bound, scaled_patches, sparse_depth_patches, mask_array)
        
        # run pipeline
        output = method.run(input_image, input_sparse_depth, validity_map, device)

        # Compute error metrics using intermediate (globally aligned) depth
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate = output["ga_depth"], 
            target = target_depth, 
            valid = mask.astype(bool),
        )

        # Compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = output["sml_depth"], 
            target = target_depth, 
            valid = mask.astype(bool),
        )

        # Accumulate error metrics
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)


    # Compute average error metrics
    print("Averaging metrics for globally-aligned depth over {} samples".format(
        avg_error_w_int_depth.total_count
    ))
    avg_error_w_int_depth.average()

    print("Averaging metrics for SML-aligned depth over {} samples".format(
        avg_error_w_pred.total_count
    ))
    avg_error_w_pred.average()

    from prettytable import PrettyTable
    summary_tb = PrettyTable()
    summary_tb.field_names = ["metric", "GA Only", "GA+SML", "Patches"]

    summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}"])
    summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}"])
    summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}"])
    summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}"])
    summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}"])
    summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}"])
    
    print(summary_tb)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset-path', type=str, default='input',
                        help='Path to VOID release dataset.')
    parser.add_argument('-dp', '--depth-predictor', type=str, default='dpt_hybrid', 
                        help='Name of depth predictor to use in pipeline.')
    parser.add_argument('-ns', '--nsamples', type=int, default=150, 
                        help='Number of sparse metric depth samples available.')
    parser.add_argument('-sm', '--sml-model-path', type=str, default='/home/rpng/Documents/sai_ws/splat_vins_repos_test/VI-Depth/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt', 
                        help='Enter path of weigths of SML model.')

    args = parser.parse_args()
    print(args)
    
    evaluate(
        args.dataset_path,
        args.depth_predictor, 
        args.nsamples, 
        args.sml_model_path,
    )