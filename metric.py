
import librosa
from librosa.util import fix_length
import IPython.display as ipd
import soundfile as sf
from pypesq import pesq
from pystoi import stoi

def pesq_func(clean,noisy,sample_rate):
    min_length = min(len(clean), len(noisy))
    clean = clean[:min_length]
    noisy = noisy[:min_length]
    clean_re = librosa.resample(clean, orig_sr=sample_rate,target_sr=8000)
    noisy_re = librosa.resample(noisy, orig_sr=sample_rate,target_sr=8000)
    return pesq(clean_re, noisy_re, 8000)

def stoi_func(clean,noisy,sample_rate):
    min_length = min(len(clean), len(noisy))
    clean = clean[:min_length]
    noisy = noisy[:min_length]
    return stoi(clean, noisy, sample_rate)

man_union, man_sr = sf.read('man_union.wav')
man_voice, man_sr = sf.read('man_voice.wav')
man_noise, man_sr = sf.read('man_noise.wav')
man_denoised, man_sr = sf.read('man_denoised.wav')

woman_union, woman_sr = sf.read('woman_union.wav')
woman_voice, woman_sr = sf.read('woman_voice.wav')
woman_noise, woman_sr = sf.read('woman_noise.wav')
woman_denoised, woman_sr = sf.read('woman_denoised.wav')

print(pesq_func(man_voice,man_voice,sr))
print(pesq_func(man_voice,man_union,sr))
print(pesq_func(man_voice,man_denoised,sr))
print(stoi_func(man_voice,man_voice,sr))
print(stoi_func(man_voice,man_union,sr))
print(stoi_func(man_voice,man_denoised,sr))

print(pesq_func(woman_voice,woman_voice,sr))
print(pesq_func(woman_voice,woman_union,sr))
print(pesq_func(woman_voice,woman_denoised,sr))
print(stoi_func(woman_voice,woman_voice,sr))
print(stoi_func(woman_voice,woman_union,sr))
print(stoi_func(woman_voice,woman_denoised,sr))