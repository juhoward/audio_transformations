from glob import glob
import os
import torchaudio
import torch
import torchaudio.functional as F

class AudioDatasetCreator(object):
    def __init__(self,
                 data_dir:str,
                 output_dir:str,
                 augmentation_list:list[str],
                 rir_filename:str='./audio_samples/sample_rir.wav',
                 chatter_filename:str='./audio_samples/people-in-lounge-1.wav',
                 snr_dbs:int=20,
                 resample=True, sample_rate=44100, 
                 tempo=True, tempo_range=(0,0), 
                 pitch=True, pitch_range=(0,0),
                 noise=True, noise_range=(0,0)):
        self.aug_list = augmentation_list
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.resample = resample
        self.sample_rate = sample_rate
        self.tempo = tempo
        self.tempo_range = tempo_range
        self.pitch = pitch
        self.pitch_range = pitch_range
        self.noise = noise
        self.noise_range = noise_range
        self.snr_dbs = snr_dbs
        self.rir_filename = rir_filename
        self.rir = self.load_rir()
        self.chatter_filename = chatter_filename
        self.chatter = self.load_chatter()
        self.augmentations = {
            'rir' : self.add_rir,
            'chatter' : self.add_chatter,
            }

    def load_rir(self):
        rir_raw, sample_rate = torchaudio.load(self.rir_filename)
        if len(rir_raw.size()) > 1:
            rir_raw = rir_raw[0].unsqueeze(0)
        rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
        rir = rir / torch.linalg.vector_norm(rir, ord=2)
        return rir
    
    def load_chatter(self):
        chatter_raw, sample_rate = torchaudio.load(self.chatter_filename)
        # just use a single channel
        if len(chatter_raw.size()) > 1:
            chatter = chatter_raw[0].unsqueeze(0)
        # extend the chatter's length
        if chatter.size()[0] < 100000:
            chatter = torch.cat((chatter, chatter), dim=1)
        return chatter
    
    def add_chatter(self, waveform):
        chatter = self.chatter[:,:waveform.shape[1]]
        snr_dbs_tensor = torch.tensor([self.snr_dbs])
        return F.add_noise(waveform, chatter, snr_dbs_tensor)
    
    def add_rir(self, waveform):
        # apply room impulse response
        augmented = F.fftconvolve(waveform, self.rir)
        return augmented
    
    def load(self):
        '''
        torchaudio dataloader
        '''
        for p in self.paths:
            # return waveform for transformations & hold sample_rate in self
            waveform, self.sample_rate = torchaudio.load(p)
            print(p)
            yield waveform
    
    def augment(self, waveform):
        for i in self.aug_list:
            try:
                waveform = self.augmentations[i](waveform=waveform)
            except KeyError as err:
                print('Augmentation not available.')
                print(f'Please select from : {list(self.augmentations.keys())}')
        return waveform
    
    def save(self, waveform, idx):
        fpath = self.paths[idx]
        fname = fpath.split('/')[-1]
        os.makedirs(self.output_dir, exist_ok=True)
        aug_fname = self.output_dir + '/augmented_' + fname
        torchaudio.save(aug_fname, waveform, self.sample_rate)

    def generate_augmentations(self):
        self.paths = glob(self.data_dir + '/*.mp3')
        self.paths.sort()
        audio_reader = self.load()
        idx = 0
        for _ in range(len(self.paths)):
            waveform = next(audio_reader)
            print(waveform.shape)
            augmented = self.augment(waveform)
            self.save(augmented,idx)
            idx += 1

if __name__ == "__main__":
    data_dir = '/data/datasets/audio_samples/binary-one'
    output_dir = './audio_samples/output'
    # start with two augmentations for now
    augmentations_list = ['rir', 'chatter']
    # higher values == less background noise
    inverse_noise_volume = 20

    dset_creator = AudioDatasetCreator(data_dir, 
                                       output_dir, 
                                       augmentations_list, 
                                       snr_dbs=inverse_noise_volume)
    dset_creator.generate_augmentations()