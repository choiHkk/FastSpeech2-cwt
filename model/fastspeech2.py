import os
import json
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor, Linear
from utils.tools import get_mask_from_lengths
from .layers import Linear



class FastSpeech2(nn.Module):
    """ FastSpeech2 with alignment learning """

    def __init__(self, preprocess_config, model_config, train_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None, 
        mel_lens=None,
        max_mel_len=None,
        cwt_spec_targets=None,
        cwt_mean_target=None,
        cwt_std_target=None,
        uv=None,
        e_targets=None, 
        attn_priors=None, 
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None, 
        gen=False, 
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            g = self.speaker_emb(speakers)
            output = output + g.unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks, 
            attn_h, 
            attn_s, 
            attn_logprob
        ) = self.variance_adaptor(
            output, 
            src_lens, 
            src_masks,
            mels, 
            mel_lens, 
            mel_masks, 
            max_mel_len, 
            cwt_spec_targets,
            cwt_mean_target,
            cwt_std_target,
            uv,
            e_targets, 
            attn_priors, 
            g, 
            p_control,
            e_control,
            d_control,
            step, 
            gen, 
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output
        postnet_output, output = postnet_output.transpose(1,2), output.transpose(1,2)
        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_h, 
            attn_s, 
            attn_logprob
        )