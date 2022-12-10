import numpy as np
from PIL import Image
#import soundfile as sf 
#from scipy import interpolate
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def default_loader(path):    
    return Image.open(path)

def spec_2d_input_audio_loader(path):
    signal, samplerate = sf.read(path)
    dt = 1/samplerate
    t = np.arange(0, len(signal)*dt, dt)
    
    minsignal, maxsignal = signal.min(), signal.max()
    signal = 2 * (signal - minsignal)/(maxsignal - minsignal) -1
    
    samplingFrequency   = samplerate
    powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(signal, Fs=samplingFrequency, cmap="rainbow")    
    x_time, y_fre = np.meshgrid(time, freqenciesFound, sparse = True)
    f = interpolate.interp2d(x_time,y_fre,powerSpectrum,kind = 'linear')
    
    xnew = np.linspace(min(time), max(time),512)
    ynew = np.linspace(min(freqenciesFound), max(freqenciesFound),512)
    np_spec = f(xnew, ynew)
    return np_spec

def wave_1d_input_audio_loader(path):
    signal, samplerate = sf.read(path)
    dt = 1/samplerate
    t = np.arange(0, len(signal)*dt, dt)
    
    minsignal, maxsignal = signal.min(), signal.max()
    signal = 2 * (signal - minsignal)/(maxsignal - minsignal) -1
    
    t_new = np.linspace(min(t),max(t),160000)

    f = interpolate.interp1d(t,signal)
    
    y_new = f(t_new)
    np_wave = y_new
    return np_wave
def spec_2d_input_figure_loader(path):
    im = Image.open(path)
    im = im.convert("L")
    np_spec = np.array(im)
    return np_spec

def wave_1d_input_figure_loader(path):
    im = Image.open(path)
    im = im.convert("L")
    np_wave = np.array(im)
    return np_wave

# dataset txt : spectrogram_path + '\t' + wavefore_path + '\t' + T/F + '\n'
class CoughDataset(Dataset):
    def __init__(self, txt, transform=None, loader_spec=default_loader):
        
        img_path = open(txt, 'r')
        imgs = []
        for line in img_path:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split('\t')
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader_spec = loader_spec
        # self.loader_wave =  wave_1d_input_audio_loader

        self.trans_train = transforms.Compose([
            transforms.Grayscale(num_output_channels = 1),
            transforms.ColorJitter(brightness=0),
            transforms.Resize(size=(150,150)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.trans_test = transforms.Compose([
            transforms.Grayscale(num_output_channels = 1),
            transforms.Resize(size=(150,150)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    
    def __getitem__(self, index):
        path_spec, label = self.imgs[index]
        img_spec = self.loader_spec(path_spec)
        # img_wave = self.loader_wave(path_wave)

        if self.transform is not None:
            if self.transform == 'train':
                # img_spec = self.trans_train(img_spec)
                # img_wave = self.trans_train(img_wave)
                img_spec = self.trans_train(img_spec)

            elif self.transform in ['valid', 'test']:
                img_spec = self.trans_test(img_spec)
                # img_wave = self.trans_test(img_wave)

        return {'img_spec': img_spec, 'target': label, 'path_spec': path_spec}

    def __len__(self):
        return len(self.imgs)
