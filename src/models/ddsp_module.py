import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule

from src.models.components.controller import Controller
from src.models.components.filtered_noise import FilteredNoise
from src.models.components.harmonic_oscillator import HarmonicOscillator
from src.models.components.reverb import ConvolutionalReverb
from src.utils.constants import SAMPLE_RATE
from src.utils.multiscale_stft_loss import distance


def time_plot(value, name):
    time = np.arange(len(value))
    data = np.stack([time, value]).T
    table = wandb.Table(columns=["time", name], data=data)
    plot = wandb.plot.line(table, "time", name)
    return plot


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
        # self.controller = TransformerController(in_ch, n_harmonics, n_filters, 512, 512)
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
            harm_ctrl, noise_ctrl = self.controller(f0, amp)
            harm = self.harmonics(*harm_ctrl)
            noise = self.noise(noise_ctrl)
            y = self.reverb(harm + noise)
            loss = distance(x, y)

        # Log all the things
        self.log("val/loss", loss)

        if wandb.run is None:
            return loss

        if batch_nb == 0:
            # Save impulse-response as audio, once per epoch
            ir = np.flip(self.reverb.ir[:, 0].cpu().numpy().T)
            ir = wandb.Audio(ir, sample_rate=SAMPLE_RATE)

            # Log reverb once
            wandb.log(
                {
                    "ir": ir,
                }
            )

        # Log other learned parameters, 4 samples each
        _, master_amp, overtone_amps = harm_ctrl
        if batch_nb < 4:
            # Generate audio
            audio = wandb.Audio(y[0].cpu().numpy().T, sample_rate=SAMPLE_RATE)
            # Generate inferred harmonic oscillator master amplitude plot
            loudness = master_amp[0, 0].cpu().numpy()
            loudness_plot = time_plot(loudness, "loudness")
            # Generate noise band controls
            im_noise = noise_ctrl[0, 0].cpu().numpy()
            im_noise /= im_noise.max()
            noise_plot = wandb.Image(im_noise * 255)
            # Generate overtone controls
            im_overtones = overtone_amps[0, 0].cpu().numpy()
            im_overtones /= im_overtones.max()
            overtone_plot = wandb.Image(im_overtones * 255)

            wandb.log(
                {
                    f"{batch_nb}": audio,
                    f"loudness_{batch_nb}": loudness_plot,
                    f"overtones_{batch_nb}": overtone_plot,
                    f"noise_bands_{batch_nb}": noise_plot,
                }
            )

        return loss

    def test_step(self, batch, batch_nb):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
