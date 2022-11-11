import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def default_loader(path):    
    return Image.open(path)

def png_loader(path):
    t = Image.open(path)
    np_img = np.array(t)
    return np_img

# dataset txt : spectrogram_path + '\t' + wavefore_path + '\t' + T/F + '\n'
class CoughDataset(Dataset):
    def __init__(self, txt, transform=None, loader=png_loader):
        img_path = open(txt, 'r')
        imgs = []
        for line in img_path:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split('\t')
            imgs.append((words[0], words[1], words[2]))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

        self.trans_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.trans_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    
    def __getitem__(self, index):
        path_spec, path_wave, label = self.imgs[index]
        img_spec = self.loader(path_spec)
        img_wave = self.loader(path_wave)

        if self.transform is not None:
            if self.transform == 'train':
                img_spec = self.trans_train(img_spec)
                img_wave = self.trans_train(img_wave)

            elif self.transform in ['valid', 'test']:
                img_spec = self.trans_test(img_spec)
                img_wave = self.trans_test(img_wave)

        return {'img_spec': img_spec, 'img_wave': img_wave, 'target': label,
                'path_spec': path_spec, 'path_wave': path_wave}

    def __len__(self):
        return len(self.imgs)