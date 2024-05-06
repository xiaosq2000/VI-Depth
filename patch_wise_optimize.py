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
    counter = 0
    if np.any(input_sparse_depth):
        # get non zero values
        u, v = np.nonzero(input_sparse_depth)
        # plot them on input image
        for i in range(len(u)):
            cv2.circle(input_image, (v[i], u[i]), 1, (0, 255, 0), -1)
            counter += 1
    print("Number of sparse points: ", counter)
    plt.imshow(input_image)
    plt.show()

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

#reduce sparse depth points
def reduce_sparse_points(input_sparse_points, num_points):
    u, v = np.nonzero(input_sparse_points)
    sparse_points = np.column_stack((u, v))
    
    if len(sparse_points) <= num_points:
        return sparse_points
    
    indices = np.random.choice(len(sparse_points), num_points, replace=False)
    reduced_sparse_points = sparse_points[indices]
    
    reduced_sparse_points_img = np.zeros_like(input_sparse_points, dtype=input_sparse_points.dtype)
    for i in range(len(reduced_sparse_points)):
        reduced_sparse_points_img[reduced_sparse_points[i][0], reduced_sparse_points[i][1]] = input_sparse_points[reduced_sparse_points[i][0], reduced_sparse_points[i][1]]

    return reduced_sparse_points_img

def output_model_sparse(input_image, input_sparse_depth, device, method, mask, target_depth_inv, num_points=None):
    reduce_sparse_depth = input_sparse_depth
    if num_points is not None:
        reduce_sparse_depth = reduce_sparse_points(input_sparse_depth, num_points)
    
    print("Number of sparse points: ", np.count_nonzero(reduce_sparse_depth))
    output = method.run(input_image, reduce_sparse_depth, None, device)

    error_w_int_depth = metrics.ErrorMetrics()
    error_w_int_depth.compute(
        estimate = output["ga_depth"], 
        target = target_depth_inv, 
        valid = mask.astype(bool),
    )
    error_w_pred = metrics.ErrorMetrics()
    error_w_pred.compute(
        estimate = output["sml_depth"],
        target = target_depth_inv,
        valid = mask.astype(bool),
    )
    return error_w_int_depth, error_w_pred

def uniform_sample_depth(num_features, target_depth_inv, input_sparse_depth, input_image, display=False):
    # Extract non-zero indices from sparse depth image
    viz_img = input_image.copy()

    height, width = target_depth_inv.shape

    # Flatten the depth image to a 1D array
    flattened_depth = target_depth_inv.flatten()

    # Find the non-zero depth indices
    non_zero_indices = np.nonzero(flattened_depth)[0]

    # Calculate the number of non-zero depth values
    total_non_zero_depth = len(non_zero_indices)

    # Compute the sampling ratio based on the number of features
    sampling_ratio = num_features / total_non_zero_depth

    # Randomly select depth indices based on the sampling ratio
    sampled_indices = np.random.choice(non_zero_indices, size=num_features, replace=False)

    # Initialize the sampled depth array
    sampled_depth_inv = np.zeros_like(flattened_depth)

    # Set sampled depth values to 1.0 / original depth values
    sampled_depth_inv[sampled_indices] = 1.0 / flattened_depth[sampled_indices]

    # Reshape the sampled depth array to the original image shape
    sampled_depth_inv = sampled_depth_inv.reshape((height, width))

    # Display the resulting image if specified
    if display:
        non_zero_locations = np.nonzero(sampled_depth_inv)
        for u, v in zip(non_zero_locations[0], non_zero_locations[1]):
            cv2.circle(viz_img, (v, u), 1, (0, 255, 0), -1)

        plt.title("Sampled Image")
        plt.imshow(viz_img)
        plt.show()
        #cv2.imshow("Sampled Image", sampled_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    return sampled_depth_inv

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
    avg_error_w_sampled_depth = metrics.ErrorMetricsAverager()

    #avg_error_w_int_depth_20 = metrics.ErrorMetricsAverager()
    #avg_error_w_int_depth_50 = metrics.ErrorMetricsAverager()
    #avg_error_w_int_depth_70 = metrics.ErrorMetricsAverager()

    avg_error_w_pred = metrics.ErrorMetricsAverager()
    avg_error_w_pred_sampled = metrics.ErrorMetricsAverager()
    #avg_error_w_pred_20 = metrics.ErrorMetricsAverager()
    #avg_error_w_pred_50 = metrics.ErrorMetricsAverager()
    #avg_error_w_pred_70 = metrics.ErrorMetricsAverager()

    # iterate through inputs list
    #for i in tqdm(range(len(test_image_list))):
    for i in tqdm(range(1,2,1)):   
        # Image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)

        # Sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 1000.0
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
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 1000.0
        target_depth[target_depth <= 0] = 0.0

        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth
        target_depth = 1.0 / target_depth

        viz_image = input_image.copy()
        visualize_sparse_depth(input_sparse_depth, viz_image)
        
        sampled_input_sparse = uniform_sample_depth(800, target_depth, input_sparse_depth, input_image, display=True)
        #depth_infer_inv = method.infer_depth(input_image)
        #viz_bound, infer_depth_patches_inv, sparse_depth_patches, mask_array = get_superpixel(input_image, depth_infer_inv, input_sparse_depth)
        
        
        #viz_image = input_image.copy()
        #visualize_sparse_depth(input_sparse_depth, viz_image)
        #viz_image_reduced = input_image.copy()
        #reduce_sparse_depth_20 = reduce_sparse_points(input_sparse_depth, 20)
        #reduce_sparse_depth_50 = reduce_sparse_points(input_sparse_depth, 50)
        #reduce_sparse_depth_70 = reduce_sparse_points(input_sparse_depth, 70)
        #visualize_sparse_depth(reduce_sparse_depth, viz_image_reduced)

        # scaled_patches = scale_patches(infer_depth_patches_inv, sparse_depth_patches, min_pred, max_pred, max_depth, min_depth)
        # visualize_depth_img(viz_image, viz_bound, scaled_patches, sparse_depth_patches, mask_array)
        
        # run pipeline
        #output = method.run(input_image, input_sparse_depth, validity_map, device)
        #output_20 = method.run(input_image, reduce_sparse_depth_20, validity_map, device)
        #output_50 = method.run(input_image, reduce_sparse_depth_50, validity_map, device)
        #output_70 = method.run(input_image, reduce_sparse_depth_70, validity_map, device)

        ## Compute error metrics using intermediate (globally aligned) depth
        # error_w_int_depth = metrics.ErrorMetrics()
        # error_w_init_depth_20 = metrics.ErrorMetrics()
        # error_w_init_depth_50 = metrics.ErrorMetrics()
        # error_w_init_depth_70 = metrics.ErrorMetrics()

        # error_w_int_depth.compute(
        #     estimate = output["ga_depth"], 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )

        # error_w_init_depth_20.compute(
        #     estimate = output_20["ga_depth"], 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )

        # error_w_init_depth_50.compute(
        #     estimate = output_50["ga_depth"], 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )

        # error_w_init_depth_70.compute(
        #     estimate = output_70["ga_depth"], 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )

        ## Compute error metrics using SML output depth
        # error_w_pred = metrics.ErrorMetrics()
        # error_w_pred_20 = metrics.ErrorMetrics()
        # error_w_pred_50 = metrics.ErrorMetrics()
        # error_w_pred_70 = metrics.ErrorMetrics()

        # error_w_pred.compute(
        #     estimate = output["sml_depth"], 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )

        # error_w_pred_20.compute(
        #     estimate = output_20["sml_depth"], 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )

        # error_w_pred_50.compute(
        #     estimate = output_50["sml_depth"], 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )

        # error_w_pred_70.compute(
        #     estimate = output_70["sml_depth"], 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )

        #sampled_input_sparse , input_sparse_depth
        error_w_int_depth, error_w_pred = output_model_sparse(input_image, input_sparse_depth, device, method, mask, target_depth, num_points=None)
        error_w_sampled_depth, error_w_pred_sampled = output_model_sparse(input_image, sampled_input_sparse, device, method, mask, target_depth, num_points=None)
        # Accumulate error metrics
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_sampled_depth.accumulate(error_w_sampled_depth)
        #avg_error_w_int_depth_20.accumulate(error_w_init_depth_20)
        #avg_error_w_int_depth_50.accumulate(error_w_init_depth_50)
        #avg_error_w_int_depth_70.accumulate(error_w_init_depth_70)
        avg_error_w_pred.accumulate(error_w_pred)
        avg_error_w_pred_sampled.accumulate(error_w_pred_sampled)
        #avg_error_w_pred_20.accumulate(error_w_pred_20)
        #avg_error_w_pred_50.accumulate(error_w_pred_50)
        #avg_error_w_pred_70.accumulate(error_w_pred_70)


    # Compute average error metrics
    print("Averaging metrics for globally-aligned depth over {} samples".format(
        avg_error_w_int_depth.total_count
    ))
    avg_error_w_int_depth.average()
    avg_error_w_sampled_depth.average()
    #avg_error_w_int_depth_20.average()
    #avg_error_w_int_depth_50.average()
    #avg_error_w_int_depth_70.average()

    print("Averaging metrics for SML-aligned depth over {} samples".format(
        avg_error_w_pred.total_count
    ))
    avg_error_w_pred.average()
    avg_error_w_pred_sampled.average()
    #avg_error_w_pred_20.average()
    #avg_error_w_pred_50.average()
    #avg_error_w_pred_70.average()

    from prettytable import PrettyTable
    summary_tb = PrettyTable()
    summary_tb.field_names = ["metric", "GA Only", "GA sampled", "GA+SML", "GA+SML sampled"]

    summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_sampled_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}", f"{avg_error_w_pred_sampled.rmse_avg:7.2f}"])
    summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_sampled_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}", f"{avg_error_w_pred_sampled.mae_avg:7.2f}"])
    summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_sampled_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}", f"{avg_error_w_pred_sampled.absrel_avg:8.3f}"])
    summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_sampled_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}", f"{avg_error_w_pred_sampled.inv_rmse_avg:7.2f}"])
    summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_sampled_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}", f"{avg_error_w_pred_sampled.inv_mae_avg:7.2f}"])
    summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_sampled_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}", f"{avg_error_w_pred_sampled.inv_absrel_avg:8.3f}"])
    
    print(summary_tb)

    #summary_tb.field_names = ["metric", "GA Only", "GA Only_20", "GA Only_50", "GA Only_70", "GA+SML", "GA+SML_20", "GA+SML_50", "GA+SML_70"]

    #summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_int_depth_20.rmse_avg:7.2f}",f"{avg_error_w_int_depth_50.rmse_avg:7.2f}", f"{avg_error_w_int_depth_70.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}", f"{avg_error_w_pred_20.rmse_avg:7.2f}", f"{avg_error_w_pred_50.rmse_avg:7.2f}", f"{avg_error_w_pred_70.rmse_avg:7.2f}"])
    #summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_int_depth_20.mae_avg:7.2f}", f"{avg_error_w_int_depth_50.mae_avg:7.2f}", f"{avg_error_w_int_depth_70.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}", f"{avg_error_w_pred_20.mae_avg:7.2f}", f"{avg_error_w_pred_50.mae_avg:7.2f}", f"{avg_error_w_pred_70.mae_avg:7.2f}"])
    #summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_int_depth_20.absrel_avg:8.3f}", f"{avg_error_w_int_depth_50.absrel_avg:8.3f}", f"{avg_error_w_int_depth_70.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}", f"{avg_error_w_pred_20.absrel_avg:8.3f}", f"{avg_error_w_pred_50.absrel_avg:8.3f}", f"{avg_error_w_pred_70.absrel_avg:8.3f}"])
    #summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_int_depth_20.inv_rmse_avg:7.2f}", f"{avg_error_w_int_depth_50.inv_rmse_avg:7.2f}", f"{avg_error_w_int_depth_70.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}", f"{avg_error_w_pred_20.inv_rmse_avg:7.2f}", f"{avg_error_w_pred_50.inv_rmse_avg:7.2f}", f"{avg_error_w_pred_70.inv_rmse_avg:7.2f}"])
    #summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_int_depth_20.inv_mae_avg:7.2f}", f"{avg_error_w_int_depth_50.inv_mae_avg:7.2f}", f"{avg_error_w_int_depth_70.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}", f"{avg_error_w_pred_20.inv_mae_avg:7.2f}", f"{avg_error_w_pred_50.inv_mae_avg:7.2f}", f"{avg_error_w_pred_70.inv_mae_avg:7.2f}"])
    #summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_int_depth_20.inv_absrel_avg:8.3f}", f"{avg_error_w_int_depth_50.inv_absrel_avg:8.3f}", f"{avg_error_w_int_depth_70.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}", f"{avg_error_w_pred_20.inv_absrel_avg:8.3f}", f"{avg_error_w_pred_50.inv_absrel_avg:8.3f}", f"{avg_error_w_pred_70.inv_absrel_avg:8.3f}"])
    
    #print(summary_tb)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
#/media/saimouli/RPNG_FLASH_4/datasets/VOID_150
    parser.add_argument('-ds', '--dataset-path', type=str, default='/home/rpng/datasets/splat_vins/table1',
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