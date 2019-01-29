import io
import os
import csv    
import subprocess
import shutil
import librosa
from pydub import AudioSegment

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types



#자르기를 시작하는 위치, seconds
offset = 60
# 자르는 크기, seconds
duration = 15

pDir = "C:/0.ITStudy/13.ML//" # 원본파일 위치
len = librosa.get_duration(filename="C:/0.ITStudy/13.ML/ASMR.wav") # 녹음된 원본파일

# if iteration > 1:
#     iteration = 1

# iteration = int(len/duration)
iteration = 2
count = 0
for i in range(iteration):
    y, sr = librosa.load(pDir + "ASMR.wav", sr = 16000, mono = True, offset = offset + i*duration, duration = duration)
    librosa.output.write_wav(pDir + "file" + str(i) + ".wav", y, sr)
    count = count + 1
print("파일 분할 완료!" + str(iteration) + "개 파일 생성")


for j in range(count):
    audio = AudioSegment.from_wav(pDir + "file" + str(j) + ".wav")
    audio = audio.set_channels(1)
    audio.export("C:/Users/Playdata/speech//" + str(j) + ".flac", format="flac")


client = speech.SpeechClient()
f = open('output.csv', 'w', encoding='utf-8', newline='')

for i in range(count):
    with io.open("{}.flac".format(i), 'rb') as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding='FLAC',
        sample_rate_hertz=16000,
        language_code='ko_KR'
    )

    response = client.recognize(config, audio)

    print('Waiting for operation to complete...')

    
    for k in response.results:
        alternatives = k.alternatives
        for alternative in alternatives:
            print('번역 : {}'.format(alternative.transcript))
            print('정확도 : {}'.format(alternative.confidence))

            wr = csv.writer(f)
            wr.writerow([alternative.transcript, alternative.confidence])

f.close()

