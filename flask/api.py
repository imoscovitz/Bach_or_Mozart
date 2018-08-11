import numpy as np
import pickle
import pandas as pd
import librosa
import librosa.display
from pygame import mixer

import sys
sys.path.insert(0, '/Users/ilanmoscovitz/github/sf18_ds11/projects/03-mcnulty/')
import MTheory as mt

model_pipe = pickle.load(open('./model/model.pkl', 'rb'))

def make_prediction(audio_path):

    # Play the song while we wait
    mixer.init()
    mixer.music.load(audio_path)
    mixer.music.play()

    # Extract audio feature
    sample = audio_to_sample(audio_path)

    # Extract music theory features
    example = sample_to_example(sample)
    example['Mode'] = example['Mode'].map(lambda x: 0 if x=='Major' else 1)

    mt.display_chroma(sample.chords)

    prediction = str(model_pipe.predict(example)[0])
    probs = round(np.max(model_pipe.predict_proba(example)),2)

    result = {
            'composer': prediction,
            'probs': probs
        }
    if probs>.7:
        result['confidence'] = "I'm pretty sure."
    else:
        result['confidence'] = "At least I'm pretty sure."

    return result

def audio_to_sample(audio_path):
    """
    Extract a Sample from audio_path
    """
    # Extract audio data
    extractions = mt.parse_audio(audio_path, extraction_keys=['chroma'])
    # Build sample
    sample = mt.Sample(extractions, {'path':audio_path,'name':audio_path.split('/')[-1]})
    # Tonality
    sample.build_tonality_matches()
    return sample

def sample_to_example(sample):
    example_df = pd.DataFrame()

    def get_tempo(y, sr, hop_length=512):
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                                   hop_length=hop_length)[0]
        return int(tempo)

    def get_spectral_centroid(y, sr):
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_range = np.max(cent) - np.min(cent)
        return int(sc_range)

    def get_spectral_contrast(y, sr):
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        return contrast

    # Collect extractions
    y = sample.base['y']
    sr = sample.base['sr']
    chroma = sample.base['chroma']

    # Collect simple features
    tempo = get_tempo(y, sr)
    sc_range = get_spectral_centroid(y, sr)
    contrast = get_spectral_contrast(y, sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_var = np.var(contrast, axis=1)

    # Add features to df
    features = {'Tempo':tempo,
                'SCentr_Range':sc_range}

    for band in range(len(contrast_mean)):
        features['SContr_mean'+str(band)] = contrast_mean[band]
    for band in range(len(contrast_var)):
        features['SContr_var'+str(band)] = contrast_var[band]

    # Generate intervalic features
    interval_strengths = sample.interval_summary()
    interval_classes = sample.interval_class_summary()
    poly_mean = sample.mean_polyphony()
    poly_var = sample.var_polyphony()

    # Generate harmonic features
    sample.try_build_tonality_matches()
    mode = sample.mode
    tonal_clarity = sample.tonal_clarity

    # Add features to df
    features = {'Poly_mean':poly_mean,
                'Poly_var':poly_var,
                'Mode':mode,
                'Tonal_Clarity':tonal_clarity}

    intervals = [('Interval_'+str(i)) for i in list(range(1,7))]
    features.update(dict(zip(intervals, interval_strengths)))

    interval_class_names = ['Dissonance','Consonance','Perfect']
    features.update(dict(zip(interval_class_names, interval_classes)))

    example_df = example_df.append(features, ignore_index=True)

    return example_df

if __name__ == '__main__':
    print(make_prediction(example))
