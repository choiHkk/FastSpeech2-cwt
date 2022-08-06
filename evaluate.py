import argparse
import os

import torch
import yaml
import torch.nn as nn
# from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
# from dataset import Dataset
from data_utils import AudioTextDataset, AudioTextCollate, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    # dataset = Dataset(
    #     "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    # )
    dataset = AudioTextDataset(
        preprocess_config['path']['validation_files'], preprocess_config)
    
    batch_size = train_config["optimizer"]["batch_size"]
    collate_fn = AudioTextCollate()
    # loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=dataset.collate_fn,
    # )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=8, 
        pin_memory=True, 
        drop_last=False
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config, train_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(11)]
    # for batchs in loader:
    for batch in loader:
        # for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            # output = model(*(batch[1:]))
            output = model(*(batch), step=step, gen=False)

            # Cal Loss
            losses = Loss(batch, output, step=step)

            for i in range(len(losses)):
                loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, TL: {:.4f}, ML: {:.4f}, MPL: {:.4f}, EL: {:.4f}, DL: {:.4f}, CL: {:.4f}, BL: {:.4f}, CS: {:.4f}, CSM: {:.4f}, CSS: {:.4f}, US: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            step=step, 
            fig=fig,
            tag="Validation/step_{}".format(step),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            step=step, 
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_reconstructed".format(step),
        )
        log(
            logger,
            step=step, 
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_synthesized".format(step),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)