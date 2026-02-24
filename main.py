import importlib
import torch

from PIL import Image
from pathlib import Path
from torchvision import transforms

from seg_model.model_utils import build_model
from utils_ import set_seed, read_coord_list, make_save_dir, sensitivity_and_plot


def main(config, device, intv_patches_tensors, block_size, sensitivity_type):

    model = build_model(config.network, config.model_config)

    best_model = Path(config.work_dir) / f'{config.data_name}_best.pth'
    if best_model.exists():
        checkpoint = torch.load(best_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise ValueError("Pre-trained model not find in the resume model zoo, please check it.")

    model.eval()
    model = model.to(device)

    save_path = Path(f'exp_map/{config.data_name}_{sensitivity_type}_ts{config.threshold}_b{block_size}')
    make_save_dir(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    block_top_left = read_coord_list(config.coord_file)
    image_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    for image_name, top_left_i, top_left_j in block_top_left:
        if config.data_name == 'HAM10000':
            test_image_path = Image.open(Path(f'path to your HAM10000 dataset')).convert('RGB')
            gt_image_path = Image.open(Path(f'path to your FIVES dataset')).convert('L').resize((256, 256), Image.NEAREST)
        elif config.data_name == 'FIVES':
            test_image_path = Image.open(Path(f'path to your HAM10000 dataset groundtruth folder')).convert('RGB')
            gt_image_path = Image.open(Path(f'path to your FIVES dataset groundtruth folder')).convert('L').resize((256, 256), Image.NEAREST)

        test_image = image_transforms(test_image_path).unsqueeze(0).to(device)
        gt_image = image_transforms(gt_image_path).unsqueeze(0).to(device)

        if sensitivity_type == 'local':
            init_patch_num, patch_num = config.init_patch_num, config.patch_num
            intv_patches_tensor = intv_patches_tensors
            x_start, y_start, patch_size = top_left_i, top_left_j, 32
            save_name = f'{config.network}_{image_name}_x{x_start}_y{y_start}_p{patch_size}_b{block_size}.png'
        elif sensitivity_type == 'zero_local':
            init_patch_num, patch_num = None, None
            intv_patches_tensor = None
            x_start, y_start, patch_size = top_left_i, top_left_j, 32
            save_name = f'{config.network}_{image_name}_x{x_start}_y{y_start}_p{patch_size}_b{block_size}.png'

        sensitivity_and_plot(
                sensitivity_type,
                model,
                test_image,
                gt_image,
                block_size,
                threshold=config.threshold,
                save_path=save_path,
                save_name=save_name,
                init_patch_num=init_patch_num,
                patch_num=patch_num,
                intv_patches_tensor=intv_patches_tensor,
                x_start=x_start,
                y_start=y_start,
                patch_size=patch_size,
        )


if __name__ == '__main__':
    # model_name_list = ["unet", "segnet", "lightmunet", "hrnet", "unext", "vmunet", "unetpp", "ukan", "mcure", "segformer", "swinunet", "transunet"]
    model_name_list = ["unet"]
    device = "cuda:0"

    sensitivity_type = 'local'
    # sensitivity_type = 'zero_local'
    block_size = 8

    intv_patches_path_list = list(Path(f'your intv patches folder').glob("*.png"))[:10000]
    intv_patches = [
        transforms.ToTensor()(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
        for p in intv_patches_path_list
    ]
    intv_patches_tensors = torch.cat(intv_patches, dim=0)  # (N_patches, C, H, W)

    for seed, model_name in enumerate(model_name_list):
        print(model_name)
        model_module = importlib.import_module(f"seg_model.model_zoo.{model_name}.HAM10000_model_config")   # HAM10000
        # model_module = importlib.import_module(f"seg_model.model_zoo.{model_name}.FIVES_model_config")    # FIVES
        config = getattr(model_module, "setting_config")

        set_seed(seed)
        main(config, device, intv_patches_tensors, block_size, sensitivity_type)






