# %%
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_list[idx])
        img = Image.open(img_path)

        w, h = img.size
        assert w % 4 == 0
        assert h % 4 == 0

        # Resize the image to the desired output size
        downsized_img = img.resize((int(w/4), int(h/4)), resample=Image.BICUBIC)

        if self.transform:
            img = self.transform(img)

        downsized_img = transforms.ToTensor()(downsized_img)
        img = transforms.ToTensor()(img)
        return downsized_img, img

# Define the transformations to be applied to each image
transform = transforms.Compose([
    # rescale=1. / 255,
])

# Define the dataset path
dataset = ImageDataset("./DIV2K_train_HR/", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %% Print the shapes
sample_1 = next(iter(dataset))
print(sample_1[0].shape)
print(sample_1[1].shape)

# %% Display the images
transforms.ToPILImage()(sample_1[0]).show()
transforms.ToPILImage()(sample_1[1]).show()
# %%
