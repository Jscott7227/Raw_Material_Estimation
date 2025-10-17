import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torchvision
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
if not hasattr(torchvision, "_is_tracing"):
    torchvision._is_tracing = lambda: False
    

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.eval()
mask_generator = SamAutomaticMaskGenerator(sam)


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

def segment_grain_pile(image):
    print(image)
    sam_masks = mask_generator.generate(image)  # Generate SAM masks
    print(len(sam_masks))

    if not sam_masks:
        print("Fuck")
        return image, [], []  # No masks found

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

def get_image_mask(image):
        image, filtered_masks, sam_masks = segment_grain_pile(image)
        heatmap = (cm.jet(filtered_masks[0])[:, :, :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(image[:, :, ::-1], 0.5, heatmap, 0.5, 0)
        cv2.imwrite("overlay_mask.png", overlay)
        return overlay, filtered_masks[0]

image = cv2.cvtColor(cv2.imread("image (6).png"), cv2.COLOR_BGR2RGB)
get_image_mask(image)

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
        
image = cv2.cvtColor(cv2.imread("image (6).png"), cv2.COLOR_BGR2RGB)
overlay, mass = calc_weight(image=image, density=95)
print(mass)
    
# image_path = "image (6).png"
# output_folder = "test"
# os.makedirs(output_folder, exist_ok=True)
# image, filtered_masks, sam_masks = segment_grain_pile(image_path)
# base_name = os.path.splitext(os.path.basename(image_path))[0]
# print(len(filtered_masks))
# for i, mask in enumerate(filtered_masks):
#     print("WTF")
#     mask_path = os.path.join(output_folder, f"{base_name}_sammask_{i}.png")
#     cv2.imwrite(mask_path, mask)
    
#     # Optional display
#     plt.imshow(image)
#     plt.imshow(mask, alpha=0.5)
#     plt.axis("off")
#     plt.show()
