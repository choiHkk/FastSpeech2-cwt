## Introduction
1. FastSpeech2 오픈 소스와 한국어 데이터셋(KSS)을 사용해 빠르게 학습합니다.
2. 기존 오픈소스는 MFA기반 preprocessing을 진행한 상태에서 학습을 진행하지만 본 레포지토리에서는 alignment learning 기반 학습을 진행하고 preprocessing으로 인해 발생할 수 있는 디스크 용량 문제를 방지하기 위해 data_utils.py로부터 학습 데이터가 feeding됩니다.
3. 기존 오픈소스는 pitch를 그대로 사용하지만 논문에서는 pitch를 cwt를 통해 pitch spectrogram으로 변환하는 과정이 포함되어 있기 때문에 data_utils.py에 반영했습니다.
4. conda 환경으로 진행해도 무방하지만 본 레포지토리에서는 docker 환경만 제공합니다. 기본적으로 ubuntu에 docker, nvidia-docker가 설치되었다고 가정합니다.
5. GPU, CUDA 종류에 따라 Dockerfile 상단 torch image 수정이 필요할 수도 있습니다.
6. preprocessing 단계에서는 학습에 필요한 transcript와 stats 정도만 추출하는 과정만 포함되어 있습니다.
7. 그 외의 다른 preprocessing 과정은 필요하지 않습니다.

## Dataset
1. download dataset - https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
2. `unzip /path/to/the/kss.zip -d /path/to/the/kss`
3. `mkdir /path/to/the/FastSpeech2-cwt/data/dataset`
4. `mv /path/to/the/kss.zip /path/to/the/FastSpeech2-cwt/data/dataset`

## Docker build
1. `cd /path/to/the/FastSpeech2-cwt`
2. `docker build --tag FastSpeech2_cwt:latest .`

## Training
1. `nvidia-docker run -it --name 'FastSpeech2-cwt' -v /path/to/FastSpeech2-cwt:/home/work/FastSpeech2-cwt --ipc=host --privileged FastSpeech2_cwt:latest`
2. `cd /home/work/FastSpeech2-cwt`
3. `cd /home/work/FastSpeech2-cwt/hifigan`
4. `unzip generator_universal.pth.tar.zip .`
5. `cd /home/work/FastSpeech2-cwt`
6. `ln -s /home/work/FastSpeech2-cwt/data/dataset/kss`
7. `python preprocess.py ./config/kss/preprocess.yaml`
8. `python train.py -p ./config/kss/preprocess.yaml -m ./config/kss/model.yaml -t ./config/kss/train.yaml`
9. arguments
  * -p : preprocess config path
  * -m : model config path
  * -t : train config path
10. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Tensorboard losses
![FastSpeech2-cwt-tensorboard-losses](https://user-images.githubusercontent.com/69423543/183249577-e48d1b40-b14e-42b0-a51a-bfa3281e98d9.png)


## Tensorboard Stats
![FastSpeech2-cwt-tensorboard-trn-stats](https://user-images.githubusercontent.com/69423543/183249740-c8c64f79-1920-4ea6-8d39-bce8f2463935.png)
![FastSpeech2-cwt-tensorboard-val-stats](https://user-images.githubusercontent.com/69423543/183249668-1f4f5a3d-4930-427b-be6f-f6287e41cb1a.png)


## Reference
1. [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
2. [One TTS Alignment To Rule Them All](https://arxiv.org/pdf/2108.10447.pdf)
3. [FastSpeech2 github](https://github.com/ming024/FastSpeech2)
4. [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS)
