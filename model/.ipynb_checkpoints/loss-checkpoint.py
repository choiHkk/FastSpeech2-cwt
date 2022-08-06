import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        self.binarization_loss_enable_steps = train_config['duration']['binarization_loss_enable_steps']
        self.binarization_loss_warmup_steps = train_config['duration']['binarization_loss_warmup_steps']
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()

    def forward(self, inputs, predictions, step):
        (
            _, 
            _, 
            _, 
            _, 
            mel_targets,
            _,
            _,
            cwt_spec_targets,
            cwt_mean_targets,
            cwt_std_targets,
            uv_targets,
            energy_targets,
            _,
        ) = inputs
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            duration_targets,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_hard, 
            attn_soft, 
            attn_logprob, 
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, :, :mel_masks.shape[1]]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        cwt_spec_targets.requires_grad = False
        cwt_mean_targets.requires_grad = False
        cwt_std_targets.requires_grad = False
        uv_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        
        (cwt_spec_predictions, cwt_mean_predictions, cwt_std_predictions) = pitch_predictions
        cwt_spec_predictions, uv_predictions = cwt_spec_predictions[:, :, :10], cwt_spec_predictions[:,:,-1]
        
        cwt_spec_loss = self.mae_loss(cwt_spec_predictions, cwt_spec_targets)
        cwt_mean_loss = self.mse_loss(cwt_mean_predictions, cwt_mean_targets)
        cwt_std_loss = self.mse_loss(cwt_std_predictions, cwt_std_targets)
        
        uv_std_loss = (
            F.binary_cross_entropy_with_logits(
                uv_predictions, uv_targets, reduction="none"
            ) * mel_masks.float()).sum() / mel_masks.float().sum()

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(1))
        
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        ctc_loss = self.sum_loss(
            attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens)
        if step < self.binarization_loss_enable_steps:
            bin_loss_weight = 0.
        else:
            bin_loss_weight = min((step-self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
        bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
        
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        
        total_loss = (
            mel_loss + 
            postnet_mel_loss + 
            duration_loss + 
            energy_loss + 
            ctc_loss + 
            bin_loss + 
            cwt_spec_loss + 
            cwt_mean_loss + 
            cwt_std_loss + 
            uv_std_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            energy_loss,
            duration_loss,
            ctc_loss, 
            bin_loss, 
            cwt_spec_loss, 
            cwt_mean_loss, 
            cwt_std_loss, 
            uv_std_loss
        )

    
class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()