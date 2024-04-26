import numpy as np
import matplotlib.pyplot as plt
import metrics
from modules.estimator import LeastSquaresEstimator

def param_sweep(depth_infer, scale, shift):
    inv_metric =  depth_infer * scale + shift
    #inv_metric,_,_ = scale_up(depth_infer, sparse_depth, min_depth, max_depth)
    return inv_metric.astype(np.float32)

def param_sweep_shift(shift_ls, scale, depth_infer, gt_depth_inv, mask, rmse_ls, frame_idx, save_img = False):
    best_shift = shift_ls; best_shift_rmse = rmse_ls; shift_rmse= []
    min_counter = 0.03; max_counter = 0.03
    for shift in np.linspace(shift_ls-min_counter, shift_ls+max_counter, num=500):
        infer_metric_depth_inv = param_sweep(depth_infer, scale, shift)
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(infer_metric_depth_inv, gt_depth_inv, mask)
        shift_rmse.append(error_w_int_depth.rmse)
        if error_w_int_depth.rmse < best_shift_rmse:
            best_shift_rmse = error_w_int_depth.rmse
            best_shift = shift

    #if save_img == True:
    plt.figure(1)
    plt.plot(np.linspace(shift_ls-min_counter, shift_ls+max_counter, num=500), shift_rmse)
    plt.plot(shift_ls, rmse_ls, 'o')
    plt.plot(best_shift, best_shift_rmse, 'go')
    plt.legend(["RMSE", "LS"])
    plt.xlabel("Shift parameters")
    plt.ylabel("RMSE (mm)")
    #plt.show()
    return best_shift, best_shift_rmse

def param_sweep_scale(scale_ls, shift, depth_infer, gt_depth_inv, mask, rmse_ls, frame_idx, optim_shift=False, save_img = False):
    best_scale_rmse = rmse_ls; best_scale = scale_ls
    scale_rmse = []
    min_counter = 0.00001; max_counter = 0.00003
    #min_counter = 0; max_counter = 0
    for scale_iter in np.linspace(scale_ls-min_counter, scale_ls+max_counter, num=500):
        #print(f"Scale: {scale_iter}, Shift: {shift}")
        
        infer_metric_depth_inv = param_sweep(depth_infer, scale_iter, shift)
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(infer_metric_depth_inv, gt_depth_inv, mask)
        #print("RMSE: ", error_w_int_depth.rmse)
        scale_rmse.append(error_w_int_depth.rmse)
        if error_w_int_depth.rmse < best_scale_rmse:
            best_scale_rmse = error_w_int_depth.rmse
            best_scale = scale_iter
            
    plt.figure(2)
    plt.plot(np.linspace(scale_ls-min_counter, scale_ls+max_counter, num=500), scale_rmse)
    plt.plot(scale_ls, rmse_ls, 'o')
    plt.plot(best_scale, best_scale_rmse, 'go')
    plt.legend(["RMSE", "LS"])
    plt.xlabel("Scale parameters")
    plt.ylabel("RMSE (mm)")
    #plt.show()
    return best_scale, best_scale_rmse

def compute_ls_solution(depth_pred, input_sparse_depth, input_sparse_depth_valid, min_pred, max_pred):
    # global scale and shift alignment
    GlobalAlignment = LeastSquaresEstimator(
        estimate=depth_pred,
        target=input_sparse_depth,
        valid=input_sparse_depth_valid
    )
    GlobalAlignment.compute_scale_and_shift()
    GlobalAlignment.apply_scale_and_shift()
    GlobalAlignment.clamp_min_max(clamp_min=min_pred, clamp_max=max_pred)
    int_depth = GlobalAlignment.output.astype(np.float32)
    return int_depth, GlobalAlignment.scale, GlobalAlignment.shift