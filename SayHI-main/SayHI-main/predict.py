import librosa
import numpy as np
import soundfile as sf

def feature_extractor(y, sr):
    print("entered")
    S = np.abs(librosa.stft(y))
    
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)
    tonnetz_var = np.var(tonnetz.T, axis=0)
    features = np.append(tonnetz_mean, tonnetz_var)
    print("1")
    spec_centroid = librosa.feature.spectral_centroid(y=y)
    spec_centroid_mean = np.mean(spec_centroid, axis = 1)
    spec_centroid_var = np.var(spec_centroid, axis = 1)
    features = np.append(features, [spec_centroid_mean, spec_centroid_var])
    print("2")
    mfcc = librosa.feature.mfcc(y=y)
    mfcc_mean = np.mean(mfcc.T, axis = 0)
    mfcc_var = np.var(mfcc.T, axis = 0)
    features = np.append(features, mfcc_mean)
    features = np.append(features, mfcc_var)
    print("3")
    spec_width = librosa.feature.spectral_bandwidth(y=y)
    spec_width_mean = np.mean(spec_width)
    spec_width_var = np.var(spec_width)
    features = np.append(features, [spec_width_mean, spec_width_var])
    print("4")
    spec_contrast = librosa.feature.spectral_contrast(y=y)
    spec_contrast_mean = np.mean(spec_contrast.T, axis = 0)
    spec_contrast_var= np.var(spec_contrast.T, axis = 0)
    features = np.append(features, spec_contrast_mean)
    features = np.append(features, spec_contrast_var)
    print("done")
    return features

ans = {
    "0": "Male",
    "1": "Female",
    "2": "Female",
    "3": "Male"
}

def final(filename):
        print("audio rcvd")
        y,sr = librosa.load(r'E:\Projects\SayHI\file.wav')
       # sf.write('stereo_file1.wav', reduced_noise, 48000, 'PCM_24')
        print("before before entering")
        y = librosa.to_mono(y)
        y = librosa.effects.harmonic(y)
        print("before entering")
        data = feature_extractor(y,sr)
        data = ans[filename.split('.')[0]]
        return data

if __name__ == "__main__":
      final()