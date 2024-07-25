import torch
import numpy as np
import torch.nn.functional as F
import cv2

def resize_and_pad(tensor, target_size, ensure_multiple_of=32, interpolation=cv2.INTER_CUBIC):
    n, h, w, c = tensor.shape
    target_h, target_w = target_size

    # Determine the new size preserving the aspect ratio
    aspect_ratio = w / h
    if target_w / target_h > aspect_ratio:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)
    else:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    
    # Resize the tensor using OpenCV
    tensor_np = tensor.cpu().numpy()
    resized_tensor_np = np.zeros((n, new_h, new_w, c))
    for i in range(n):
        resized_tensor_np[i] = cv2.resize(tensor_np[i], (new_w, new_h), interpolation=interpolation)
    
    resized_tensor = torch.from_numpy(resized_tensor_np)

    # Calculate padding to ensure dimensions are multiples of 32
    pad_h = (ensure_multiple_of - new_h % ensure_multiple_of) % ensure_multiple_of
    pad_w = (ensure_multiple_of - new_w % ensure_multiple_of) % ensure_multiple_of

    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left

    # Pad the resized tensor
    padded_tensor = F.pad(resized_tensor.permute(0, 3, 1, 2), (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode='constant', value=0)

    # Change back to (n, h, w, c)
    padded_tensor = padded_tensor.permute(0, 2, 3, 1)
    
    return padded_tensor.cuda().float()