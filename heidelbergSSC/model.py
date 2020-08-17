import sys, os
import glob
import wget
import zipfile
import h5py
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn

# Dataset definition
class shdDataset(Dataset):
    def __init__(self, path, samplingTime=1e-3, sampleLength=1.5, mode='train'):
        # look if dataset is available in path. If not download it.
        os.makedirs(path, exist_ok=True)
        if mode == 'train':
            datasetPath = path + '/shd_train.h5'
            if len(glob.glob(datasetPath)) == 0:
                print('Starting download')
                wget.download('https://compneuro.net/datasets/shd_train.h5.zip', out=path)
                print('\nExtracting file')
                with zipfile.ZipFile(datasetPath+'.zip', 'r') as f:    
                    f.extractall(path)
        elif mode == 'test':
            datasetPath = path + '/shd_test.h5'
            if len(glob.glob(datasetPath)) == 0:
                print('Starting download')
                wget.download('https://compneuro.net/datasets/shd_test.h5.zip', out=path)
                print('\nExtracting file')
                with zipfile.ZipFile(datasetPath+'.zip', 'r') as f:    
                    f.extractall(path)
        elif mode == 'valid':
            datasetPath = path + '/shd_valid.h5'
            if len(glob.glob(datasetPath)) == 0:
                print('Starting download')
                wget.download('https://compneuro.net/datasets/shd_valid.h5.zip', out=path)
                print('\nExtracting file')
                with zipfile.ZipFile(datasetPath+'.zip', 'r') as f:    
                    f.extractall(path)
            pass
        else:
            raise Exception('Invialid mode: {}. Must be one of train/test/valid'.format(mode))

        self.data = h5py.File(datasetPath, 'r')
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        x = self.data['spikes']['units'][index]
        t = self.data['spikes']['times'][index]
        p = np.zeros(x.shape)
        label = int(self.data['labels'][index])

        TD = snn.io.event(x, None, p, t)

        inputSpikes = TD.toSpikeTensor(torch.zeros((1,1,700,self.nTimeBins)),
                        samplingTime=self.samplingTime)
        desiredClass = torch.zeros((20, 1, 1, 1))
        desiredClass[label,...] = 1
        return inputSpikes, desiredClass, label

    def __len__(self):
        return len(self.data['labels'])

# Dataset definition
class sscDataset(Dataset):
    def __init__(self, path, samplingTime=1e-3, sampleLength=1.5, mode='train'):
        # look if dataset is available in path. If not download it.
        os.makedirs(path, exist_ok=True)
        if mode == 'train':
            datasetPath = path + '/ssc_train.h5'
            if len(glob.glob(datasetPath)) == 0:
                print('Starting download')
                wget.download('https://compneuro.net/datasets/ssc_train.h5.zip', out=path)
                print('\nExtracting file')
                with zipfile.ZipFile(datasetPath+'.zip', 'r') as f:    
                    f.extractall(path)
        elif mode == 'test':
            datasetPath = path + '/ssc_test.h5'
            if len(glob.glob(datasetPath)) == 0:
                print('Starting download')
                wget.download('https://compneuro.net/datasets/ssc_test.h5.zip', out=path)
                print('\nExtracting file')
                with zipfile.ZipFile(datasetPath+'.zip', 'r') as f:    
                    f.extractall(path)
        elif mode == 'valid':
            datasetPath = path + '/ssc_valid.h5'
            if len(glob.glob(datasetPath)) == 0:
                print('Starting download')
                wget.download('https://compneuro.net/datasets/ssc_valid.h5.zip', out=path)
                print('\nExtracting file')
                with zipfile.ZipFile(datasetPath+'.zip', 'r') as f:    
                    f.extractall(path)
            pass
        else:
            raise Exception('Invialid mode: {}. Must be one of train/test/valid'.format(mode))

        self.data = h5py.File(datasetPath, 'r')
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        x = self.data['spikes']['units'][index]
        t = self.data['spikes']['times'][index]
        p = np.zeros(x.shape)
        label = int(self.data['labels'][index])

        TD = snn.io.event(x, None, p, t)

        inputSpikes = TD.toSpikeTensor(torch.zeros((1,1,700,self.nTimeBins)),
                        samplingTime=self.samplingTime)
        desiredClass = torch.zeros((35, 1, 1, 1))
        desiredClass[label,...] = 1
        return inputSpikes, desiredClass, label

    def __len__(self):
        return len(self.data['labels'])

        
# Network definition
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer

        self.conv = torch.nn.ModuleList([
            slayer.conv( 1,  8, (1, 3), padding=(0, 0), stride=(1, 2)),
            slayer.conv( 8, 16, (1, 3), padding=(0, 1), stride=(1, 2)),
            slayer.conv(16, 32, (1, 3), padding=(0, 1), stride=(1, 2)),
            slayer.conv(32, 64, (1, 3), padding=(0, 1), stride=(1, 2)),
        ])
        
        self.fc = torch.nn.ModuleList([
            slayer.dense(44*64, 512),
            slayer.dense(  512,  35),
        ])

        self.convDelay = torch.nn.ModuleList([
            slayer.delay(8),
            slayer.delay(16),
            slayer.delay(32),
            slayer.delay(64),
        ])

        self.delay = torch.nn.ModuleList([
            slayer.delay(512),
        ])

    def forward(self, spike):
        for (conv, delay) in zip(self.conv, self.convDelay):
            spike = self.slayer.spike(conv(self.slayer.psp(spike)))
            spike = delay(spike)

        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))

        for (fc, delay) in zip(self.fc, self.delay):
            spike = self.slayer.spike(fc(self.slayer.psp(spike)))
            spike = delay(spike)

        spike = self.slayer.spike(self.fc[-1](self.slayer.psp(spike)))

        return spike

def sscNet(netParams):
    net =  Network(netParams)
    
    for i, conv in enumerate(net.conv):
        conv.weight.data = torch.FloatTensor(np.load('Params/conv{}.npy'.format(i))).reshape(conv.weight.shape)

    for i, fc in enumerate(net.fc):
        fc.weight.data = torch.FloatTensor(np.load('Params/fc{}.npy'.format(i))).reshape(fc.weight.shape)

    for i, convDelay in enumerate(net.convDelay):
        convDelay.delay.data = torch.FloatTensor(np.load('Params/convDelay{}.npy'.format(i))).reshape(convDelay.delay.shape)

    for i, delay in enumerate(net.delay):
        delay.delay.data = torch.FloatTensor(np.load('Params/delay{}.npy'.format(i))).reshape(delay.delay.shape)

    return net

if __name__ == "__main__":
    device = torch.device('cuda:5')

    netParams = snn.params('network.yaml')
    net = sscNet(netParams).to(device)

    trainingSet = sscDataset(path='data', samplingTime=1e-3, sampleLength=1.5, mode='train')
    validationSet = sscDataset(path='data', mode='valid')
    testingSet = sscDataset(path='data', mode='test')

    validationLoader  = DataLoader(dataset=validationSet , batch_size=32, shuffle=True, num_workers=1)
    testLoader = DataLoader(dataset=testingSet, batch_size=32, shuffle=True, num_workers=1)

    stats = snn.utils.stats()

    for i, (input, target, label) in enumerate(validationLoader, 0):
        net.eval()

        with torch.no_grad():
            input  = input.to(device)
            target = target.to(device) 

            output = net.forward(input)

            stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.training.numSamples     += len(label)

        stats.print(
                0, i, 
                header= [
                    '[{}/{} ({:.0f}%)]'.format(i*validationLoader.batch_size, len(validationLoader.dataset), 100.0*i/len(validationLoader)),
                ],
            )

    for i, (input, target, label) in enumerate(testLoader, 0):
        net.eval()

        with torch.no_grad():
            input  = input.to(device)
            target = target.to(device) 

            output = net.forward(input)

            stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.testing.numSamples     += len(label)

        stats.print(
                0, i, 
                header= [
                    '[{}/{} ({:.0f}%)]'.format(i*testLoader.batch_size, len(testLoader.dataset), 100.0*i/len(testLoader)),
                ],
            )