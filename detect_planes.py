import os
import argparse
import glob

import torch
import numpy as np

from PIL import Image

import modules.midas.utils as utils

import pipeline
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

import metrics
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import KMeans

def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)


def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth


def detect_normals(depth_image, input_sparse_depth, input_image, display=False):
    rows, cols = depth_image.shape
    depth_denoise_img = cv2.GaussianBlur(depth_image, (3, 3), 0)
    viz_img = input_image.copy()
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Calculate the partial derivatives of depth with respect to x and y
    dx = cv2.Sobel(depth_denoise_img, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(depth_denoise_img, cv2.CV_32F, 0, 1)
    
    #compute normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)
    
    # Map the normal vectors to the [0, 255] range and convert to uint8
    # normal = (normal + 1) * 127.5
    # normal = normal.clip(0, 255).astype(np.uint8)
    # normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    if display:
        normal_viz = (normal + 1) * 127.5
        normal_viz = normal_viz.clip(0, 255).astype(np.uint8)
        normal_bgr = cv2.cvtColor(normal_viz, cv2.COLOR_RGB2BGR)
        if np.any(input_sparse_depth):
            u, v = np.nonzero(input_sparse_depth)
            for i in range(len(u)):
                cv2.circle(viz_img, (v[i], u[i]), 7, (255, 0, 0), -1)

        # make these into same plot
        plt.figure()
        plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st subplot
        plt.imshow(viz_img)
        plt.subplot(1, 3, 2)  # 1 row, 2 columns, 2nd subplot
        plt.imshow(normal_bgr)
        plt.subplot(1, 3, 3)
        plt.imshow(depth_image) 
        plt.show()
    return normal

def find_planes(depth_points, normals):
    planes = []
    num_planes = 3  # Adjust the number of planes as needed

    for i in range(num_planes):
        # Use RANSAC to fit a plane
        ransac = RANSACRegressor()
        ransac.fit(depth_points, normals)

        # Extract plane parameters
        plane_normal = ransac.estimator_.coef_
        plane_distance = ransac.estimator_.intercept_

        # Store the plane parameters
        planes.append((plane_normal, plane_distance))

        # Remove inliers from the data for the next iteration
        inliers_mask = ransac.inlier_mask_
        depth_points = depth_points[~inliers_mask]
        normals = normals[~inliers_mask]

    return planes


def run(dataset_path, depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, 
        input_path, output_path, save_output):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    # get inputs
    with open(f"{dataset_path}/test_image.txt") as f: 
        test_image_list = [line.rstrip() for line in f]

    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

    for i in tqdm(range(0,4,1)):
    #for i in tqdm(range(len(test_image_list))):
        # Image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)

        # Sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0

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
        target_depth_inv = 1.0 / target_depth
        
        depth_infer_inv = method.infer_depth(input_image)

        normals = detect_normals(depth_infer_inv, input_sparse_depth, input_image, True)
        
        normals = detect_normals(target_depth, input_sparse_depth, input_image, True)
        
        output = method.run(input_image, input_sparse_depth, validity_map, device)
        
        
        # compute error metrics using intermediate (globally aligned) depth
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate = output["ga_depth"], 
            target = target_depth, 
            valid = mask.astype(bool),
        )

        # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = output["sml_depth"], 
            target = target_depth, 
            valid = mask.astype(bool),
        )
        
        # accumulate error metrics
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
        
        # cv2.imshow('depth_infer_inv', depth_infer_inv)
        # cv2.imshow('target_depth', target_depth)
        # cv2.waitKey(0)
        
        #planes = detect_planes_from_normals(normals, input_image, True)

        
        # plt.imshow(depth_infer_inv)
        # plt.show()

    avg_error_w_int_depth.average()
    avg_error_w_pred.average()

    from prettytable import PrettyTable
    summary_tb = PrettyTable()
    summary_tb.field_names = ["metric", "GA Only", "GA+SML"]

    summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}"])
    summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}"])
    summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}"])
    summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}"])
    summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}"])
    summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}"])
    
    print(summary_tb)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
#/media/saimouli/RPNG_FLASH_4/datasets/TABLE/table01_all_frames_sparse
    parser.add_argument('-ds', '--dataset-path', type=str, default='./input',
                        help='Path to VOID release dataset.')
    
    # model parameters
    parser.add_argument('-dp', '--depth-predictor', type=str, default='dpt_hybrid', 
                            help='Name of depth predictor to use in pipeline.')
    parser.add_argument('-ns', '--nsamples', type=int, default=150, 
                            help='Number of sparse metric depth samples available.')
    parser.add_argument('-sm', '--sml-model-path', type=str, default='/home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt', 
                            help='Path to trained SML model weights.')

    # depth parameters
    parser.add_argument('--min-pred', type=float, default=0.1, 
                            help='Min bound for predicted depth values.')
    parser.add_argument('--max-pred', type=float, default=8.0, 
                            help='Max bound for predicted depth values.')
    parser.add_argument('--min-depth', type=float, default=0.2, 
                            help='Min valid depth when evaluating.')
    parser.add_argument('--max-depth', type=float, default=5.0, 
                            help='Max valid depth when evaluating.')

    # I/O paths
    parser.add_argument('-i', '--input-path', type=str, default='./input', 
                            help='Path to inputs.')
    parser.add_argument('-o', '--output-path', type=str, default='./output', 
                            help='Path to outputs.')
    parser.add_argument('--save-output', dest='save_output', action='store_true', 
                            help='Save output depth map.')
    parser.set_defaults(save_output=False)

    args = parser.parse_args()
    print(args)
    
    run(
        args.dataset_path,
        args.depth_predictor, 
        args.nsamples, 
        args.sml_model_path, 
        args.min_pred,
        args.max_pred, 
        args.min_depth, 
        args.max_depth,
        args.input_path,
        args.output_path,
        args.save_output
    )