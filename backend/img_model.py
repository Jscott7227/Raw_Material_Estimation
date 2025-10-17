"""Helpers for optional image-based mass estimation."""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import cm
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

if not hasattr(torchvision, "_is_tracing"):
    torchvision._is_tracing = lambda: False


def _load_sam() -> Optional[SamAutomaticMaskGenerator]:
    """Return a SAM mask generator if the checkpoint is available."""
    checkpoint = os.environ.get("SAM_CHECKPOINT_PATH", "sam_vit_b_01ec64.pth")
    if not os.path.exists(checkpoint):
        return None
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    sam.eval()
    return SamAutomaticMaskGenerator(sam)


mask_generator = _load_sam()


def color_filter_custom(image, mask):
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255

    # Optional cleanup (remove noise, smooth edges)
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    return clean_mask

import cv2
import numpy as np

def segment_grain_pile(image: np.ndarray) -> Tuple[np.ndarray, list, list]:
    if mask_generator is None:
        raise FileNotFoundError(
            "SAM checkpoint not found. Set SAM_CHECKPOINT_PATH or place 'sam_vit_b_01ec64.pth' in the backend directory."
        )

    sam_masks = mask_generator.generate(image)
    if not sam_masks:
        return image, [], []

    filtered_masks = []
    original_masks = []

    for mask_dict in sam_masks:
        sam_mask = mask_dict['segmentation']
        original_masks.append(sam_mask)

        # Apply your custom color filter
        filtered_mask = color_filter_custom(image, sam_mask)
        filtered_masks.append(filtered_mask)

    # Compute area of each filtered mask
    masks_with_area = [(f_mask, o_mask, np.sum(f_mask > 0))
                       for f_mask, o_mask in zip(filtered_masks, original_masks)]

    # Keep top 5 by area
    masks_sorted = sorted(masks_with_area, key=lambda x: x[2], reverse=True)[:5]

    # Compute image center
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Among top 5, find the one most centered
    masks_with_distance = []
    for f_mask, o_mask, area in masks_sorted:
        ys, xs = np.nonzero(f_mask)
        if len(xs) == 0:
            continue
        cx, cy = np.mean(xs), np.mean(ys)
        # No need for sqrt — same ordering
        distance = (cx - center_x)**2 + (cy - center_y)**2
        masks_with_distance.append((f_mask, o_mask, area, distance))

    if not masks_with_distance:
        return image, [], []

    # Pick most centered among the top 5 largest
    best_mask_tuple = min(masks_with_distance, key=lambda x: x[3])
    best_filtered_mask, best_original_mask, _, _ = best_mask_tuple

    return image, [best_filtered_mask], [best_original_mask]

def get_image_mask(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image, filtered_masks, _ = segment_grain_pile(image)
    if not filtered_masks:
        raise ValueError("Unable to locate material pile in supplied image.")
    heatmap = (cm.jet(filtered_masks[0])[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(image[:, :, ::-1], 0.5, heatmap, 0.5, 0)
    return overlay, filtered_masks[0]

def calc_volume(image):
    overlay, mask = get_image_mask(image=image)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    input_batch = transform(image)
    img_h, img_w = image.shape[:2]
    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    with torch.no_grad():
        depth = midas(input_batch)
        # Resize depth to original image size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(img_h, img_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().numpy()
        
    masked_depth = depth * mask
    masked_values = masked_depth[mask > 0]
    depth_norm = np.zeros_like(masked_depth)

    if masked_values.size > 0:
        depth_norm[mask > 0] = (masked_values - masked_values.min()) / (masked_values.max() - masked_values.min())
        
    H, W = depth.shape
    combined_mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Apply mask
    masked_depth = depth * combined_mask_resized
    masked_values = masked_depth[combined_mask_resized > 0]

    if masked_values.size > 0:
        # Fill areas outside mask with the minimum depth within the mask
        floor_value = masked_values.min()
        filled_depth = masked_depth.copy()
        filled_depth[combined_mask_resized == 0] = floor_value
    else:
        filled_depth = masked_depth

    # ---- VOLUME CALCULATION ----
    pixel_size_m = 0.003  # size of one pixel in meters
    pile_volume_m3 = np.sum(filled_depth[combined_mask_resized > 0]) * (pixel_size_m ** 2)
    
    return overlay, pile_volume_m3

def calc_weight(image, density):
    overlay, vol = calc_volume(image=image)
    short_ton = 907.1847
    mass_kg = vol * density
    mass_short_ton = mass_kg / short_ton
    return overlay, mass_short_ton



__all__ = ["calc_weight"]
