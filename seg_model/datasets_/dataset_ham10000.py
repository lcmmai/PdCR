from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path


class dataset_ham10000(Dataset):
    def __init__(self, data_path, image_size, mode='train'):
        assert mode in ['train', 'test']
        self.image_list = []
        self.mask_list = []
        self.mode = mode
        self.image_size = image_size

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

        self.img_path = Path(data_path) / mode
        self.gt_path = Path(data_path) / 'HAM10000_segmentations_lesion_tschandl'

        for image_path in self.img_path.iterdir():
            self.image_list.append(image_path)
            self.mask_list.append(self.gt_path / (image_path.name[:-4] + '_segmentation.png'))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        example = {}
        # print(self.image_list[index], self.mask_list[index])
        image = Image.open(self.image_list[index]).convert('RGB')
        mask = Image.open(self.mask_list[index]).convert('L').resize(self.image_size, Image.NEAREST)

        example["image"] = self.image_transforms(image)
        example["mask"] = self.image_transforms(mask)
        example["name"] = self.image_list[index].name

        return example



