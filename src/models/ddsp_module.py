import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics import MeanAbsoluteError, MinMetric

from src.models.components.controller import Controller, TransformerController
from src.models.components.filtered_noise import FilteredNoise
from src.models.components.harmonic_oscillator import HarmonicOscillator
from src.models.components.reverb import ConvolutionalReverb
from src.utils.constants import SAMPLE_RATE
from src.utils.features import Loudness
from src.utils.multiscale_stft_loss import distance


def multiline_time_plot(values, name):
    time = np.arange(len(values[0]))
    return wandb.plot.line_series(
        xs=time,
        ys=values,
        keys=[f"{name}_{idx}" for idx in range(len(values))],
        title=name,
        xname="time",
    )


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
        weight_decay=0.1,
        transformer_units=512,
        transformer_hidden=512,
        transformer_heads=2,
        transformer_layers=3,
        controller="gru",
    ):
        super().__init__()
        self.save_hyperparameters()

        match controller:
            case "gru":
                self.controller = Controller(
                    n_harmonics, n_filters, mlp_units, mlp_layers, gru_units, gru_layers
                )
            case "transformer":
                self.controller = TransformerController(
                    in_ch,
                    n_harmonics,
                    n_filters,
                    n_units=transformer_units,
                    n_hidden=transformer_hidden,
                    n_heads=transformer_heads,
                    n_layers=transformer_layers,
                )
                pass
            case _:
                raise ValueError("Valid controllers are 'gru', 'transformer']")

        self.harmonics = HarmonicOscillator(n_harmonics, in_ch)
        self.noise = FilteredNoise(n_filters, in_ch)
        self.reverb = ConvolutionalReverb(reverb_dur, in_ch, out_ch)

        self.ld = Loudness()

        self.train_acc = MeanAbsoluteError()
        self.val_acc = MeanAbsoluteError()
        self.test_acc = MeanAbsoluteError()

        self.val_acc_best = MinMetric()

    def forward(self, pitch, loudness):
        harm_ctrl, noise_ctrl = self.controller(pitch, loudness)
        harm = self.harmonics(*harm_ctrl)
        noise = self.noise(noise_ctrl)
        out = self.reverb(harm + noise)

        return out

    def on_train_start(self):
        self.val_acc_best.reset()

    def training_step(self, batch, batch_nb):
        f0, amp, x = batch
        y = self(f0, amp)
        loss = distance(x, y)

        acc = self.train_acc(self.ld.get_amp(x), self.ld.get_amp(y))
        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss

    def training_epoch_end(self, outputs):
        self.train_acc.reset()

    def validation_step(self, batch, batch_nb):
        f0, amp, x = batch
        with torch.no_grad():
            harm_ctrl, noise_ctrl = self.controller(f0, amp)
            harm = self.harmonics(*harm_ctrl)
            noise = self.noise(noise_ctrl)
            y = self.reverb(harm + noise)
            loss = distance(x, y)

        acc = self.val_acc(self.ld.get_amp(x), self.ld.get_amp(y))
        self.log("val/loss", loss)
        self.log("val/acc", acc)

        if wandb.run is None:
            return loss

        if batch_nb == 0:
            # Save impulse-response as audio, once per epoch
            ir = np.flip(self.reverb.ir[:, 0].cpu().numpy().T)
            ir = wandb.Audio(ir, sample_rate=SAMPLE_RATE)
            audios = []
            amps = []
            overtone_images = []
            noise_images = []

            _, master_amp, overtone_amps = harm_ctrl
            for wav, m_amp, overtones, noises in zip(y, master_amp, overtone_amps, noise_ctrl):
                audios.append(wandb.Audio(wav.cpu().numpy().T, sample_rate=SAMPLE_RATE))
                # Generate inferred harmonic oscillator master amplitude plot
                amps.append(m_amp[0].cpu().numpy())
                # Generate noise band controls
                im_noise = noises[0].cpu().numpy()
                im_noise /= im_noise.max()
                im_noise *= 255
                noise_images.append(wandb.Image(im_noise))
                # Generate overtone controls
                im_overtones = overtones[0]
                im_overtones = 10 * torch.log10(
                    torch.maximum(
                        torch.tensor(1e-10, device=im_overtones.device), im_overtones**2
                    )
                )
                im_overtones = torch.maximum(im_overtones, im_overtones.max() - 80.0)
                im_overtones = (im_overtones + 80.0) / 80.0
                im_overtones *= 255
                overtone_images.append(wandb.Image(im_overtones))

            loudness_plots = multiline_time_plot(amps, "loudness")
            wandb.log(
                {
                    "ir": ir,
                    "pred_waves": audios,
                    "loudness_envelops": loudness_plots,
                    "overtones": overtone_images,
                    "noise_bands": noise_images,
                }
            )

        return loss

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute())
        self.val_acc.reset()

    def test_step(self, batch, batch_nb):
        f0, amp, x = batch
        with torch.no_grad():
            harm_ctrl, noise_ctrl = self.controller(f0, amp)
            harm = self.harmonics(*harm_ctrl)
            noise = self.noise(noise_ctrl)
            y = self.reverb(harm + noise)
            loss = distance(x, y)

        acc = self.test_acc(self.ld.get_amp(x), self.ld.get_amp(y))
        self.log("test/loss", loss)
        self.log("test/acc", acc)

        return loss

    def test_epoch_end(self, outputs):
        self.test_acc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
