## Introduction
1. FastSpeech2 오픈 소스와 한국어 데이터셋(KSS)을 사용해 빠르게 학습합니다.
2. 기존 오픈소스는 MFA기반 preprocessing을 진행한 상태에서 학습을 진행하지만 본 레포지토리에서는 alignment learning 기반 학습을 진행하고 preprocessing으로 인해 발생할 수 있는 디스크 용량 문제를 방지하기 위해 data_utils.py로부터 학습 데이터가 feeding됩니다.
3. conda 환경으로 진행해도 무방하지만 본 레포지토리에서는 docker 환경만 제공합니다. 기본적으로 ubuntu에 docker, nvidia-docker가 설치되었다고 가정합니다.
4. GPU, CUDA 종류에 따라 Dockerfile 상단 torch image 수정이 필요할 수도 있습니다.
5. preprocessing 단계에서는 학습에 필요한 transcript와 stats 정도만 추출하는 과정만 포함되어 있습니다.
6. 그 외의 다른 preprocessing 과정은 필요하지 않습니다.

## Dataset
1. download dataset - https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
2. `unzip /path/to/the/kss.zip -d /path/to/the/kss`
3. `mkdir /path/to/the/FastSpeech2/data/dataset`
4. `mv /path/to/the/kss.zip /path/to/the/FastSpeech2/data/dataset`

## Docker build
1. `cd /path/to/the/FastSpeech2`
2. `docker build --tag FastSpeech2:latest .`

## Training
1. `nvidia-docker run -it --name 'FastSpeech2' -v /path/to/FastSpeech2:/home/work/FastSpeech2 --ipc=host --privileged FastSpeech2:latest`
2. `cd /home/work/FastSpeech2`
3. `cd /home/work/FastSpeech2/hifigan`
4. `unzip generator_universal.pth.tar.zip .`
5. `cd /home/work/FastSpeech2`
6. `python preprocess.py ./config/kss/preprocess.yaml`
7. `python train.py -p ./config/kss/preprocess.yaml -m ./config/kss/model.yaml -t ./config/kss/train.yaml`
8. arguments
  * -p : preprocess config path
  * -m : model config path
  * -t : train config path
9. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Tensorboard losses
![FastSpeech2-tensorboard-losses](https://user-images.githubusercontent.com/69423543/183047356-3fb819ee-dee1-40fb-9432-778a8b488202.png)

## Tensorboard Stats
![FastSpeech2-tensorboard-stats1](https://user-images.githubusercontent.com/69423543/183047576-55743354-1286-42d8-8b43-b95bbd71aea1.png)
![FastSpeech2-tensorboard-stats2](https://user-images.githubusercontent.com/69423543/183047734-796ed638-4fe1-405b-acc6-092420b835cd.png)

## Reference
1. [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
2. [One TTS Alignment To Rule Them All](https://arxiv.org/pdf/2108.10447.pdf)
3. [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS)
