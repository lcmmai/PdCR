import torch
import numpy as np
import argparse

from PIL import Image
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils_ import set_seed, make_save_dir, visualize_heatmap_pie, visualize_heatmap_bar

from seg_model.model_zoo.unext.unext import UNext


def main(args):
    device = 'cuda:0'
    set_seed(42)

    model = UNext(num_classes=1)

    model_ckpt_path = Path(args.model_ckpt_path)
    checkpoint = torch.load(model_ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    model = model.to(device)

    sensitivity_type = 'local'

    intv_patches_path_list = list(Path('intervention_patches').glob("*.png"))
    intv_patches = [
        transforms.ToTensor()(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
        for p in intv_patches_path_list
    ]
    intv_patches_tensors = torch.cat(intv_patches, dim=0)  # (N_patches, C, H, W)

    save_path = Path(args.save_path)
    make_save_dir(save_path)

    image_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    top_left_i = args.top_left_i
    top_left_j = args.top_left_j
    patch_size = 32
    block_size = 8
    init_patch_num = 3
    patch_num = 50
    threshold = 0.02

    # Taking HAM10000 as an example
    test_image_path = Image.open(Path(f'demo/{args.image_stem}.jpg')).convert('RGB')
    gt_image_path = Image.open(Path(f'demo/{args.image_stem}_segmentation.png')).convert('L').resize((256, 256), Image.NEAREST)

    test_image = image_transforms(test_image_path).unsqueeze(0).to(device)
    gt_image = image_transforms(gt_image_path).unsqueeze(0).to(device)

    save_name = 'result.png'

    # duplicated and changed from utils_.py function: occlusion_sensitivity_optimized/plot_occlusion_heatmap
    def test_cal_dice_coefficient(pred, target, epsilon=1e-10):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection) / (union + epsilon)
        return dice

    with torch.no_grad():
        base_output = (model(test_image) > 0.5).float()
        base_mask_roi = base_output[:, :, top_left_i:top_left_i + patch_size, top_left_j:top_left_j + patch_size]
        gt_roi = gt_image[:, :, top_left_i:top_left_i + patch_size, top_left_j:top_left_j + patch_size]
        base_dice = test_cal_dice_coefficient(base_mask_roi, gt_roi)
        base_dice_value = base_dice.item() * 100
        print(f"Base Dice: {base_dice_value:.2f}")

    _, c, h, w = test_image.shape
    heatmap = torch.zeros((h, w)).to(device)

    roi_mask = torch.zeros((h, w), dtype=torch.bool).to(device)
    roi_mask[top_left_i:top_left_i + patch_size, top_left_j:top_left_j + patch_size] = True

    total_patches = intv_patches_tensors.size(0)

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if roi_mask[i:i + block_size, j:j + block_size].any():
                continue

            selected_indices = torch.randint(0, total_patches, (init_patch_num,))
            selected_patches = intv_patches_tensors[selected_indices]

            occluded_images = test_image.repeat(init_patch_num, 1, 1, 1)
            occluded_images[:, :, i:i + block_size, j:j + block_size] = selected_patches

            with torch.no_grad():
                outputs = (model(occluded_images) > 0.5).float()
                mask_rois = outputs[:, :, top_left_i:top_left_i + patch_size, top_left_j:top_left_j + patch_size]
                dice_scores = test_cal_dice_coefficient(mask_rois, gt_roi.repeat(init_patch_num, 1, 1, 1)) * 100

            diffs = (dice_scores - base_dice_value).abs()
            if (diffs < threshold).all():
                continue

            selected_indices = torch.randint(0, total_patches, (patch_num,))
            selected_patches = intv_patches_tensors[selected_indices]

            occluded_images = test_image.repeat(patch_num, 1, 1, 1)
            occluded_images[:, :, i:i + block_size, j:j + block_size] = selected_patches

            with torch.no_grad():
                outputs = (model(occluded_images) > 0.5).float()
                mask_rois = outputs[:, :, top_left_i:top_left_i + patch_size, top_left_j:top_left_j + patch_size]
                dice_scores = test_cal_dice_coefficient(mask_rois, gt_roi.repeat(patch_num, 1, 1, 1)) * 100

            avg_diff = (dice_scores - base_dice_value).mean()
            if abs(avg_diff) < threshold:
                continue

            heatmap[i:i + block_size, j:j + block_size] = avg_diff

    heatmap[top_left_i:top_left_i + patch_size, top_left_j:top_left_j + patch_size] = 0

    visualize_heatmap_pie(heatmap=heatmap, patch_size=patch_size, block_size=block_size, threshold=threshold,
                          save_path=save_path, save_name=save_name, base_dice_value=base_dice_value)
    visualize_heatmap_bar(heatmap=heatmap, patch_size=patch_size, save_path=save_path, save_name=save_name)
    np.save(save_path / 'heatmap_np' / save_name, heatmap.cpu().numpy())

    max_abs = torch.max(torch.abs(heatmap))
    heatmap_norm = heatmap / (max_abs + 1e-8)

    mask_negative = heatmap_norm <= 0
    red = torch.where(mask_negative, torch.ones_like(heatmap_norm), 1.0 - heatmap_norm)
    green = torch.where(mask_negative, 1.0 + heatmap_norm, 1.0 - heatmap_norm)
    blue = torch.where(mask_negative, 1.0 + heatmap_norm, torch.ones_like(heatmap_norm))
    heatmap_rgb = torch.stack([red, green, blue], dim=0).clamp(0, 1)

    heatmap_pil = TF.to_pil_image(heatmap_rgb.cpu())
    heatmap_tensor = TF.to_tensor(heatmap_pil).to(device)

    test_img_vis = test_image.squeeze(0).clone()  # (3, H, W)
    test_img_vis = test_img_vis.clamp(0, 1)

    heatmap_tensor[:, top_left_i:top_left_i + patch_size, top_left_j:top_left_j + patch_size] = torch.tensor(
        [[[0.0]], [[1.0]], [[0.0]]], device=device
    ).expand(3, patch_size, patch_size)

    overlay = (0.7 * heatmap_tensor + 0.3 * test_img_vis).clamp(0, 1)
    no_overlay = heatmap_tensor.clamp(0, 1)

    overlay_pil = TF.to_pil_image(overlay.cpu())
    no_overlay_pil = TF.to_pil_image(no_overlay.cpu())

    overlay_pil.save(save_path / 'overlay' / save_name)
    no_overlay_pil.save(save_path / 'heatmap' / save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        required=True,
        default="demo/unext_HAM10000_best.pth",
        help="Path to the model under test, UNeXt is default."
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="demo/result_folder",
        help="Directory where results will be saved"
    )

    parser.add_argument("--image_stem", type=str, default="ISIC_0033556")

    parser.add_argument("--top_left_i", type=int, default="72",
                        help="Coordinates of RoI the upper left corner")

    parser.add_argument("--top_left_j", type=int, default="88",
                        help="Coordinates of RoI the upper left corner")

    args = parser.parse_args()

    main(args)
