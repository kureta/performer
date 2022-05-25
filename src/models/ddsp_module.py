import torch
import wandb
from pytorch_lightning import LightningModule

from src.models.components.controller import Controller
from src.models.components.filtered_noise import FilteredNoise
from src.models.components.harmonic_oscillator import HarmonicOscillator
from src.models.components.reverb import ConvolutionalReverb
from src.utils.constants import SAMPLE_RATE
from src.utils.multiscale_stft_loss import distance


class DDSP(LightningModule):
    def __init__(
        self,
        n_harmonics: int = 128,
        n_filters: int = 64,
        in_ch: int = 1,
        out_ch: int = 2,
        reverb_dur: int = 3,
        mlp_units: int = 512,
        mlp_layers: int = 3,
        gru_units: int = 512,
        gru_layers: int = 1,
        lr=0.003,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.controller = Controller(
            n_harmonics, n_filters, mlp_units, mlp_layers, gru_units, gru_layers
        )
        self.harmonics = HarmonicOscillator(n_harmonics, in_ch)
        self.noise = FilteredNoise(n_filters, in_ch)
        self.reverb = ConvolutionalReverb(reverb_dur, in_ch, out_ch)

    def forward(self, pitch, loudness):
        harm_ctrl, noise_ctrl = self.controller(pitch, loudness)
        harm = self.harmonics(*harm_ctrl)
        noise = self.noise(noise_ctrl)
        out = self.reverb(harm + noise)

        return out

    def training_step(self, batch, batch_nb):
        f0, amp, x = batch
        y = self(f0, amp)
        loss = distance(x, y)

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_nb):
        f0, amp, x = batch
        with torch.no_grad():
            y = self(f0, amp)
            loss = distance(x, y)

        self.log("val/loss", loss)
        if batch_nb < 4:
            wandb.log({f"{batch_nb}": wandb.Audio(y[0].cpu().numpy().T, sample_rate=SAMPLE_RATE)})

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
