import cv2
import numpy as np
import librosa

def solution(audio_path):
    
  y, sr = librosa.load(audio_path, sr=None) #sr=None preserves sampling rate
  n_fft = 2048 #%FFT points, adjust as you need
  hop_length = 512 #%Sliding amount for windowed FFT (adjust as needed)
  spec = librosa.feature.melspectrogram(y=y, sr=44100, n_fft=204, hop_length=512,n_mels=100, fmax=22000)
  # spec_db = perform power to decibel (db) image transformation of `spec'
  # decibel conversion → 10 log(
  #cv2.imshow(spec)



  log_mel_spectrogram = librosa.power_to_db(spec, ref=np.max)
  norm_spectrogram = cv2.normalize(log_mel_spectrogram, None, 0, 255, cv2.NORM_MINMAX)


  spectrogram_image = norm_spectrogram.astype(np.uint8)



  (T, threshInv) = cv2.threshold(spectrogram_image, 100, 255,
  cv2.THRESH_BINARY)

  count=0
  for i in range(threshInv.shape[0]):
    for j in range(threshInv.shape[1]):
      if threshInv[i][j]==255:
        count=count+1
  class_name = 'cardboard'

  if count<5000:
    class_name='cardboard'
  else:
    class_name='metal'
    
  return class_name
