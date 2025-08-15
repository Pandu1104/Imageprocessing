import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImagePairDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.image_filenames = sorted(os.listdir(lr_dir))
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')

        return self.transform(lr_image), self.transform(hr_image), filename
