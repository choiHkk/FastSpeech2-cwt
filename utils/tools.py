import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
from numba import jit, prange

matplotlib.use("Agg")


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


@jit(nopython=True)
def mas_width1(attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]): # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out


def to_device(data, device):
    (
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        cwt_specs,
        cwt_means,
        cwt_stds,
        uvs,
        energies,
        attn_priors,
    ) = data

    speakers = speakers.long().to(device)
    texts = texts.long().to(device)
    src_lens = src_lens.to(device)
    mels = mels.float().to(device)
    mel_lens = mel_lens.to(device)
    cwt_specs = cwt_specs.float().to(device)
    cwt_means = cwt_means.float().to(device)
    cwt_stds = cwt_stds.float().to(device)
    uvs = uvs.float().to(device)
    energies = energies.to(device)
    attn_priors = attn_priors.float().to(device)

    return (
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        cwt_specs,
        cwt_means,
        cwt_stds,
        uvs,
        energies,
        attn_priors,
    )


def to_device_inference(data, device):
    (
        speakers,
        texts,
        src_lens,
        max_src_len
    ) = data

    speakers = speakers.long().to(device)
    texts = texts.long().to(device)
    src_lens = src_lens.to(device)

    return (
        speakers,
        texts,
        src_lens,
        max_src_len,
    )



def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/energy_loss", losses[3], step)
        logger.add_scalar("Loss/duration_loss", losses[4], step)
        logger.add_scalar("Loss/ctc_loss", losses[5], step)
        logger.add_scalar("Loss/bin_loss", losses[6], step)
        logger.add_scalar("Loss/cwt_spec_loss", losses[7], step)
        logger.add_scalar("Loss/cwt_mean_loss", losses[8], step)
        logger.add_scalar("Loss/cwt_std_loss", losses[9], step)
        logger.add_scalar("Loss/uv_std_loss", losses[10], step)

    if fig is not None:
        logger.add_figure(tag, fig, step)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            step, 
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(
        dtype=lengths.dtype, device=lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[4][0, :mel_len].detach()#.transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach()#.transpose(0, 1)
    duration = predictions[5][0, :src_len].detach().cpu().numpy()
    
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[-2][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[-2][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        # stats = stats["pitch"] + stats["energy"][:2]
        stats = [None, None, None, None] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), energy),
            (mel_target.cpu().numpy(), energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer
        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction



def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    _, _, _, _, energy_min, energy_max = stats

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, energy = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

