import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import random
from matplotlib import pyplot as plt
import time

from PIL import Image
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def test_cal_dice_coefficient(pred, target, epsilon=1e-10):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection) / (union + epsilon)
    return dice


def get_block_mask(gt_image, block_size):

    B, C, H, W = gt_image.shape
    assert B == 1 and C == 1, "Only support batch size = 1 and single channel."

    patches = F.unfold(gt_image, kernel_size=block_size, stride=block_size)
    # print(patches.shape)
    patch_mask = (patches == 1).any(dim=1).float()
    num_patches_h = H // block_size
    num_patches_w = W // block_size
    patch_mask = patch_mask.view(1, num_patches_h, num_patches_w)
    mask = patch_mask.repeat_interleave(block_size, dim=1).repeat_interleave(block_size, dim=2)
    return mask[0]  # (H, W)


def zero_occlusion_sensitivity(model, test_image, gt_image, x_start, y_start, threshold, patch_size=32, block_size=8):
    device = test_image.device
    model.eval()

    with torch.no_grad():
        base_output = (model(test_image) > 0.5).float()
        base_mask_roi = base_output[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
        base_dice = test_cal_dice_coefficient(base_mask_roi, gt_image[:, :, x_start:x_start + patch_size,
                                                             y_start:y_start + patch_size])
        base_dice_value = base_dice[0].item() * 100
        print(f"Base Dice: {base_dice_value:.2f}")

    _, c, h, w = test_image.shape
    heatmap = torch.zeros((h, w)).to(device)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):

            if (x_start + patch_size) > i > x_start and (y_start + patch_size) > j > y_start:
                continue

            occluded_image = test_image.clone()
            occluded_image[:, :, i:i + block_size, j:j + block_size] = 0

            with torch.no_grad():
                output = (model(occluded_image) > 0.5).float()
                mask_roi = output[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
                dice = test_cal_dice_coefficient(mask_roi, gt_image[:, :, x_start:x_start + patch_size,
                                                           y_start:y_start + patch_size])
                dice_value = dice[0].item() * 100

            dice_diff = dice_value - base_dice_value
            if abs(dice_diff) < threshold:
                dice_diff = 0
            heatmap[i: i + block_size, j: j + block_size] = dice_diff

    heatmap[x_start:x_start + patch_size, y_start:y_start + patch_size] = 0

    return heatmap, base_dice_value


def occlusion_sensitivity(model, test_image, gt_image, x_start, y_start, intv_patch_dir, image_transforms,
                          patch_size=32, block_size=8, init_patch_num=3, patch_num=50, threshold=0.01):
    device = test_image.device
    model.eval()

    intv_patches_path_list = list(Path(intv_patch_dir).glob("*.png"))

    with torch.no_grad():
        base_output = (model(test_image) > 0.5).float()
        base_mask_roi = base_output[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
        base_dice = test_cal_dice_coefficient(base_mask_roi, gt_image[:, :, x_start:x_start + patch_size,
                                                             y_start:y_start + patch_size])
        base_dice_value = base_dice[0].item() * 100
        print(f"Base Dice: {base_dice_value:.2f}")

    _, c, h, w = test_image.shape
    heatmap = torch.zeros((h, w)).to(device)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):

            if (x_start + patch_size) > i > x_start and (y_start + patch_size) > j > y_start:
                continue

            preliminary_diff = []
            for _ in range(init_patch_num):
                invt_patch_path = random.choice(intv_patches_path_list)
                invt_patch = image_transforms(Image.open(Path(invt_patch_path)).convert('RGB')).unsqueeze(0)
                occluded_image = test_image.clone()
                occluded_image[:, :, i:i + block_size, j:j + block_size] = invt_patch

                with torch.no_grad():
                    output = (model(occluded_image) > 0.5).float()
                    mask_roi = output[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
                    dice = test_cal_dice_coefficient(mask_roi, gt_image[:, :, x_start:x_start + patch_size,
                                                               y_start:y_start + patch_size])
                    dice_value = dice[0].item() * 100
                    diff = abs(dice_value - base_dice_value)
                    preliminary_diff.append(diff)

            if all(d < threshold for d in preliminary_diff):
                continue

            intv_patches_path = random.sample(intv_patches_path_list, patch_num)
            dice_diff = 0
            for invt_patch_path in intv_patches_path:
                invt_patch = image_transforms(Image.open(Path(invt_patch_path)).convert('RGB')).unsqueeze(0)
                occluded_image = test_image.clone()
                occluded_image[:, :, i:i + block_size, j:j + block_size] = invt_patch

                with torch.no_grad():
                    output = (model(occluded_image) > 0.5).float()
                    mask_roi = output[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
                    dice = test_cal_dice_coefficient(mask_roi, gt_image[:, :, x_start:x_start + patch_size,
                                                               y_start:y_start + patch_size])
                    dice_value = dice[0].item() * 100

                dice_diff += (dice_value - base_dice_value)

            avg_dice_diff = dice_diff / patch_num
            if abs(avg_dice_diff) < threshold:
                avg_dice_diff = 0
            heatmap[i: i + block_size, j: j + block_size] = avg_dice_diff

    heatmap[x_start:x_start + patch_size, y_start:y_start + patch_size] = 0

    return heatmap, base_dice_value


def occlusion_sensitivity_optimized(model, test_image, gt_image, x_start, y_start, intv_patches_tensor,
                          patch_size=32, block_size=8, init_patch_num=3, patch_num=50, threshold=0.01):
    device = test_image.device
    model.eval()
    total_patches = intv_patches_tensor.size(0)

    with torch.no_grad():
        base_output = (model(test_image) > 0.5).float()
        base_mask_roi = base_output[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
        gt_roi = gt_image[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
        base_dice = test_cal_dice_coefficient(base_mask_roi, gt_roi)
        base_dice_value = base_dice.item() * 100
        print(f"Base Dice: {base_dice_value:.2f}")

    _, c, h, w = test_image.shape
    heatmap = torch.zeros((h, w)).to(device)

    roi_mask = torch.zeros((h, w), dtype=torch.bool).to(device)
    roi_mask[x_start:x_start + patch_size, y_start:y_start + patch_size] = True

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if roi_mask[i:i + block_size, j:j + block_size].any():
                continue

            if total_patches < init_patch_num:
                continue

            selected_indices = torch.randint(0, total_patches, (init_patch_num,))
            selected_patches = intv_patches_tensor[selected_indices]

            occluded_images = test_image.repeat(init_patch_num, 1, 1, 1)
            occluded_images[:, :, i:i + block_size, j:j + block_size] = selected_patches

            with torch.no_grad():
                outputs = (model(occluded_images) > 0.5).float()
                mask_rois = outputs[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
                dice_scores = test_cal_dice_coefficient(mask_rois, gt_roi.repeat(init_patch_num, 1, 1, 1)) * 100

            diffs = (dice_scores - base_dice_value).abs()
            if (diffs < threshold).all():
                continue

            selected_indices = torch.randint(0, total_patches, (patch_num,))
            selected_patches = intv_patches_tensor[selected_indices]

            occluded_images = test_image.repeat(patch_num, 1, 1, 1)
            occluded_images[:, :, i:i + block_size, j:j + block_size] = selected_patches

            with torch.no_grad():
                outputs = (model(occluded_images) > 0.5).float()
                mask_rois = outputs[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
                dice_scores = test_cal_dice_coefficient(mask_rois, gt_roi.repeat(patch_num, 1, 1, 1)) * 100

            avg_diff = (dice_scores - base_dice_value).mean()
            if abs(avg_diff) < threshold:
                continue

            heatmap[i:i + block_size, j:j + block_size] = avg_diff

    heatmap[x_start:x_start + patch_size, y_start:y_start + patch_size] = 0
    return heatmap, base_dice_value


def plot_occlusion_heatmap(heatmap, base_dice_value, test_image, x_start, y_start, patch_size, block_size, threshold, save_path: Path, save_name):
    device = test_image.device

    visualize_heatmap_pie(heatmap=heatmap, patch_size=patch_size, block_size=block_size, threshold=threshold,
                          save_path=save_path, save_name=save_name, base_dice_value=base_dice_value)
    visualize_heatmap_bar(heatmap=heatmap, patch_size=patch_size, save_path=save_path, save_name=save_name)
    np.save(save_path / 'heatmap_np' / save_name, heatmap.cpu().numpy())

    max_abs = torch.max(torch.abs(heatmap))
    heatmap_norm = heatmap / (max_abs + 1e-8)

    '''
        Calculate the values for each channel.
        If the value is red, it means that after the patch is replaced, the Dice score becomes negative, 
        indicating that the original pixels in that region had a positive contribution. 
        Otherwise, if the Dice score is positive, 
        it indicates that the original pixels in that region had a negative contribution.
    '''

    mask_negative = heatmap_norm <= 0
    red = torch.where(mask_negative, torch.ones_like(heatmap_norm), 1.0 - heatmap_norm)
    green = torch.where(mask_negative, 1.0 + heatmap_norm, 1.0 - heatmap_norm)
    blue = torch.where(mask_negative, 1.0 + heatmap_norm, torch.ones_like(heatmap_norm))
    heatmap_rgb = torch.stack([red, green, blue], dim=0).clamp(0, 1)

    heatmap_pil = TF.to_pil_image(heatmap_rgb.cpu())
    heatmap_tensor = TF.to_tensor(heatmap_pil).to(device)

    test_img_vis = test_image.squeeze(0).clone()  # (3, H, W)
    test_img_vis = test_img_vis.clamp(0, 1)

    heatmap_tensor[:, x_start:x_start + patch_size, y_start:y_start + patch_size] = torch.tensor(
        [[[0.0]], [[1.0]], [[0.0]]], device=device
    ).expand(3, patch_size, patch_size)

    overlay = (0.7 * heatmap_tensor + 0.3 * test_img_vis).clamp(0, 1)
    no_overlay = heatmap_tensor.clamp(0, 1)

    overlay_pil = TF.to_pil_image(overlay.cpu())
    no_overlay_pil = TF.to_pil_image(no_overlay.cpu())

    overlay_pil.save(save_path / 'overlay' / save_name)
    no_overlay_pil.save(save_path / 'heatmap' / save_name)


def visualize_heatmap_pie(heatmap, base_dice_value, block_size, threshold, save_path: Path, save_name, patch_size=0, roi_mask=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.cpu().numpy()
    else:
        heatmap_np = heatmap

    if patch_size > 0 and roi_mask is None:
        roi_area = patch_size * patch_size
    elif patch_size == 0 and roi_mask is not None:
        roi_area = torch.sum(roi_mask == 1).item()
    flat = heatmap_np.flatten()
    block_area = block_size * block_size
    positive_count = np.sum(flat > 0)
    negative_count = np.sum(flat < 0)
    zero_count = np.sum(flat == 0) - roi_area

    pos_block = positive_count // block_area
    neg_block = negative_count // block_area
    zero_block = zero_count // block_area

    total_block = pos_block + neg_block + zero_block
    pos_perc, neg_perc = round((pos_block / total_block) * 100, 2), round((neg_block / total_block) * 100, 2)
    zero_perc = 100 - pos_perc - neg_perc

    bins = [-np.inf, -3, -1, -0.3, -threshold, threshold, 0.3, 1, 3, np.inf]

    labels = ["to -3", "-3 to -1", "-1 to -0.3", f"-0.3 to -{threshold}",
              f"-{threshold} to {threshold}",
              f"{threshold} to 0.3", "0.3 to 1", "1 to 3", "3 to"]
    hist, _ = np.histogram(heatmap_np, bins=bins)

    with open(f"{save_path / 'statistics' / save_name[:-4]}.txt", "w") as f:
        f.write(f"base dice: {base_dice_value}\n")
        f.write(f"pos: {pos_block}, neg: {neg_block}, zero: {zero_block}\n")
        f.write(f"pos: {pos_perc}, neg: {neg_perc}, zero: {zero_perc}\n")
        for label, count in zip(labels, hist):
            f.write(f"{label}_num: {count // block_area}\n")

    sizes = [positive_count, negative_count, zero_count]
    colors = ['blue', 'red', 'white']
    labels = ['Positive', 'Negative', 'Zero']

    fig, ax = plt.subplots(figsize=(3, 3))

    fig.patch.set_alpha(0)
    ax.axis('equal')
    ax.set_axis_off()

    wedges, texts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor='black', linewidth=4)
    )
    plt.tight_layout(pad=0)

    plt.savefig(
        f"{save_path / 'pie' / save_name}",
        dpi=300,
        bbox_inches='tight',
        pad_inches=0,
        transparent=True
    )
    plt.close()


def visualize_heatmap_bar(heatmap, save_path, save_name, zero_visual_scale=0.2, patch_size=0, roi_mask=None):
    max_abs = torch.max(torch.abs(heatmap))
    heatmap_norm = heatmap / (max_abs + 1e-8)
    mask_negative = heatmap_norm <= 0
    red = torch.where(mask_negative, torch.ones_like(heatmap_norm), 1.0 - heatmap_norm)
    green = torch.where(mask_negative, 1.0 + heatmap_norm, 1.0 - heatmap_norm)
    blue = torch.where(mask_negative, 1.0 + heatmap_norm, torch.ones_like(heatmap_norm))
    heatmap_rgb = torch.stack([red, green, blue], dim=0).clamp(0, 1).permute(1, 2, 0)

    heatmap_np = heatmap.cpu().numpy()
    heatmap_rgb_np = heatmap_rgb.cpu().numpy()

    if patch_size > 0 and roi_mask is None:
        roi_area = patch_size * patch_size
    elif patch_size == 0 and roi_mask is not None:
        roi_area = torch.sum(roi_mask == 1).item()

    num_pixels = heatmap_np.size - roi_area
    num_neg = np.sum(heatmap_np < 0)
    num_pos = np.sum(heatmap_np > 0)
    num_zero = num_pixels - num_neg - num_pos

    ratio_zero = num_zero / num_pixels
    ratio_zero_draw = zero_visual_scale * ratio_zero
    ratio_neg = num_neg / num_pixels + (1 - zero_visual_scale) * ratio_zero / 2
    ratio_pos = num_pos / num_pixels + (1 - zero_visual_scale) * ratio_zero / 2

    max_index = np.argmax(heatmap_np)
    min_index = np.argmin(heatmap_np)
    min_val = np.min(heatmap_np)
    max_val = np.max(heatmap_np)

    h, w = heatmap_np.shape
    row_max, col_max = np.unravel_index(max_index, (h, w))
    row_min, col_min = np.unravel_index(min_index, (h, w))

    color_max = heatmap_rgb_np[row_max, col_max]
    color_min = heatmap_rgb_np[row_min, col_min]

    # Generate a range of values from min to max for the color bar
    value_range = np.linspace(min_val, max_val, 100).reshape(1, -1)

    colors = [
        (0.0, color_min),
        (ratio_neg, "white"),
        (ratio_neg + ratio_zero_draw, "white"),
        (1.0, color_max)
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_ratio_cmap", colors)

    fig, ax = plt.subplots(figsize=(4, 0.8))

    ax.imshow(value_range, cmap=custom_cmap, aspect='auto', extent=[0, 1, 0, 0.1])

    ax.text(0, 1.4, f"{min_val:.2f}", va='center', ha='left', fontsize=25, fontweight='bold', color='black',
            transform=ax.transAxes)
    ax.text(1, 1.4, f"{max_val:.2f}", va='center', ha='right', fontsize=25, fontweight='bold', color='black',
            transform=ax.transAxes)

    ax.set_yticks([])
    ax.set_xticks([])

    plt.tight_layout(pad=0)

    plt.savefig(
        f"{save_path / 'bar' / save_name}",
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1,
        transparent=True
    )
    plt.close()


def read_coord_list(coord_file):
    block_top_left = []
    with open(coord_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_name = parts[0]
            i = int(parts[1])
            j = int(parts[2])
            block_top_left.append((image_name, i, j))
    print(f"Total image number: {len(block_top_left)}")
    return block_top_left


def make_save_dir(save_path):
    (save_path / 'bar').mkdir(exist_ok=True, parents=True)
    (save_path / 'pie').mkdir(exist_ok=True, parents=True)
    (save_path / 'statistics').mkdir(exist_ok=True, parents=True)
    (save_path / 'overlay').mkdir(exist_ok=True, parents=True)
    (save_path / 'heatmap').mkdir(exist_ok=True, parents=True)
    (save_path / 'heatmap_np').mkdir(exist_ok=True, parents=True)


def sensitivity_and_plot(
        sensitivity_type,
        model,
        test_image,
        gt_image,
        block_size,
        threshold,
        save_path,
        save_name,
        init_patch_num=None,
        patch_num=None,
        intv_patches_tensor=None,
        x_start=None,
        y_start=None,
        patch_size=None,
):
    if sensitivity_type == 'local':
        assert x_start is not None and y_start is not None and patch_size is not None, "parameters should not be None"
        heatmap, base_dice_value = occlusion_sensitivity_optimized(
            model, test_image, gt_image,
            x_start=x_start, y_start=y_start, patch_size=patch_size, block_size=block_size,
            intv_patches_tensor=intv_patches_tensor,
            init_patch_num=init_patch_num, patch_num=patch_num, threshold=threshold
        )
        plot_occlusion_heatmap(
            heatmap, base_dice_value, test_image,
            x_start=x_start, y_start=y_start, patch_size=patch_size, block_size=block_size,
            threshold=threshold, save_path=save_path, save_name=save_name
        )

    elif sensitivity_type == 'zero_local':
        heatmap, base_dice_value = zero_occlusion_sensitivity(
            model, test_image, gt_image,
            x_start=x_start, y_start=y_start, patch_size=patch_size, block_size=block_size, threshold=threshold
        )
        plot_occlusion_heatmap(
            heatmap, base_dice_value, test_image,
            x_start=x_start, y_start=y_start, patch_size=patch_size, block_size=block_size,
            threshold=threshold, save_path=save_path, save_name=save_name
        )



















