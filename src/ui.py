import os

import librosa
import pyrootutils
import soundfile
import streamlit as st
import torch
import torch.nn.functional as F  # noqa

from src.utils.constants import HOP_LENGTH, N_FFT
from src.utils.features import Loudness, get_f0

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def file_selector(folder_path="."):
    filenames = sorted(os.listdir(folder_path))
    selected_filename = st.selectbox("Select a file", filenames)
    return os.path.join(folder_path, selected_filename)


def get_saved_model(ckpt_path):
    from src.models.ddsp_module import DDSP

    model_ = DDSP().load_from_checkpoint(ckpt_path, map_location="cuda")
    model_.eval()

    return model_


def np_audio(np_array, samplerate=48000):
    soundfile.write("temp.wav", np_array.T, samplerate, "PCM_24")
    st.audio("temp.wav", format="audio/wav")
    os.remove("temp.wav")


class Preprocess:
    def __init__(self, device):
        self.ld = Loudness().to(device)

    def do(self, y):
        if (diff := len(y) % HOP_LENGTH) != 0:
            F.pad(y, (0, HOP_LENGTH - diff))

        audio = F.pad(y[None, None, :], (N_FFT // 2, N_FFT // 2))
        loudness = self.ld.get_amp(audio)
        f0 = get_f0(audio)

        return f0, loudness


processor = Preprocess("cuda")


def process(filepath):
    y_, _ = librosa.load(filepath, sr=48000)
    y_ = torch.from_numpy(y_).cuda()
    f0_, amp_ = processor.do(y_)

    return f0_, amp_


st.text("Select a folder")
folder_name = file_selector("/home/kureta/Music")
st.write(f"{folder_name} selected.")

st.text("Select a .wav file")
filename = file_selector(folder_name)
st.write(f"{filename} selected.")

with open(filename, "rb") as file:
    audio_data = file.read()

st.audio(audio_data, format="audio/wav")
f0, amp = process(filename)

st.text("Select a checkpoint file")
filename = file_selector("./checkpoints")
st.write(f"{filename} selected.")
st.text(f"{root}")

model = get_saved_model(filename).cuda()

with torch.inference_mode():
    y = model(f0, amp)

st.text(y.shape)
np_audio(y[0].cpu().numpy())
