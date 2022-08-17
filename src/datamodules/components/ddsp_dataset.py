from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import Dataset
from tqdm import tqdm

from src import utils
from src.utils.constants import HOP_LENGTH, N_FFT, SAMPLE_RATE
from src.utils.features import Loudness, get_f0

log = utils.get_pylogger(__name__)


def generate_data(wav_dir: Path, path: Path, example_duration: int, example_hop_length: int):
    ld = Loudness().cuda()

    wav_files = list(wav_dir.glob("*.[wav]*"))

    wav_data = []
    ld_data = []
    f0_data = []
    for wf in tqdm(wav_files):
        audio, _ = librosa.load(wf, sr=44100, mono=False, dtype="float32")
        audio = torch.from_numpy(audio).cuda()
        audio = AF.resample(audio.unsqueeze(0), 44100, SAMPLE_RATE)
        # This makes sure the entire chain of audio length is a multiple of `HOP_SIZE`
        if (diff := audio.shape[-1] % HOP_LENGTH) != 0:
            audio = F.pad(audio, (0, HOP_LENGTH - diff))

        wav_data.append(audio[0].cpu())

    # Concatenate all waves into a single wave
    wav_data = torch.cat(wav_data, dim=1)
    wav_data = wav_data.unfold(1, example_duration, example_hop_length)
    wav_data = wav_data.transpose(0, 1)

    for audio in tqdm(wav_data):
        audio = F.pad(audio.unsqueeze(0), (N_FFT // 2, N_FFT // 2)).cuda()
        loudness = ld.get_amp(audio)
        f0 = get_f0(audio)

        ld_data.append(loudness[0].cpu())
        f0_data.append(f0[0].cpu())

    ld_data = torch.stack(ld_data)
    f0_data = torch.stack(f0_data)

    data = {"audio": wav_data, "loudness": ld_data, "f0": f0_data}

    torch.save(data, path)

    return data


class DDSPDataset(Dataset):
    def __init__(self, path, wav_dir=None, example_duration=4, example_hop_length=1):
        super().__init__()
        example_duration *= SAMPLE_RATE
        example_hop_length *= SAMPLE_RATE
        if example_duration % HOP_LENGTH != 0:
            log.error("Example duration must be a multiple of `HOP_LENGTH` (in samples)")
            exit(1)
        if example_hop_length % HOP_LENGTH != 0:
            log.error("Example hop length must be a multiple of `HOP_LENGTH` (in samples)")
            exit(1)

        path = Path(path).expanduser()
        if wav_dir is not None:
            wav_dir = Path(wav_dir).expanduser()
        if path.is_file():
            if wav_dir is not None:
                log.warning(
                    "You have provided both saved features path and wave file directory."
                    f"If you want features to be regenerated, delete {path}"
                )
            features = torch.load(path)
        else:
            if wav_dir is None or not wav_dir.is_dir():
                raise FileNotFoundError(f"Wave files directory {wav_dir} does not exist.")

            features = generate_data(wav_dir, path, example_duration, example_hop_length)

        self.audio = features["audio"]
        self.loudness = features["loudness"]
        self.f0 = features["f0"]

    def __len__(self):
        return len(self.f0)

    def __getitem__(self, idx):
        return self.f0[idx], self.loudness[idx], self.audio[idx]
