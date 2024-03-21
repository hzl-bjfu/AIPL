import librosa

def get_sampling_rate(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    return sr

# 你需要提供音频文件的路径
audio_file = "/home/hzl/code/investigation-phase-master/scripts/data/0009/111651_1.wav"

# 获取音频的采样率
sampling_rate = get_sampling_rate(audio_file)
print("采样率为:", sampling_rate)
