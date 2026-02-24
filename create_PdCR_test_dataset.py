from pathlib import Path
import random
from PIL import Image
import shutil


# Select 28 images from the FIVES test set for interpretability testing of the segmentation model.
# The remaining images are resized to 256×256, then divided into 8×8 patches.
# For each image, 100 patches are selected and saved to be used as masking replacement patches.


dataset_name = "FIVES"

patch_size = 8  # 256 / 8 = 32
one_side = 256 // patch_size

dataset_path = Path(f"path to your dataset")
output_folder = Path(f"path to your output folder")
output_folder.mkdir(parents=True, exist_ok=True)

patch_folder_8 = Path(f"path to your intervention patch folder {patch_size}")
patch_folder_8.mkdir(parents=True, exist_ok=True)

image_paths = list(dataset_path.glob("*.png"))
selected_images = random.sample(image_paths, 28)
remaining_images = set(image_paths) - set(selected_images)

for img_path in selected_images:
    shutil.copy(img_path, output_folder / img_path.name)

for img_path in remaining_images:
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    image_name = img_path.stem

    patches = []
    for i in range(one_side):     # 256 / 8 = 32
        for j in range(one_side):     # 256 / 8 = 32
            left = j * patch_size
            upper = i * patch_size
            patch = img.crop((left, upper, left + patch_size, upper + patch_size))
            patches.append((i, j, patch))

    selected_patches = random.sample(patches, 100)

    for i, j, patch in selected_patches:
        patch_filename = f"{image_name}_{i}_{j}.png"
        patch.save(patch_folder_8 / patch_filename)




