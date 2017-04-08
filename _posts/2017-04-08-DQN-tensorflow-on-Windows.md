---
layout: post
comments: false
title: DQN-tensorflow on Windows 7 and 10
---


이 글의 목적은 고전 게임을 심층강화학습하는 코드인 [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)를 Windows 7이나 10에서 실행하는 방법을 소개하는 것입니다. 혹시 코드와 관련된 이론적인 배경이 궁금하시다면 [논문](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)을 직접 읽어보시거나 [천상혁님의 논문 리뷰](http://sanghyukchun.github.io/90/)를 참조하시길 바랍니다.

[DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow) 라는 제목에서 알 수 있듯이, 코드 내부에서 Python 라이브러리인 tensorflow 를 사용합니다. [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow) 여기에서 구할 수 있는 코드는 Linux 환경에서 `git clone https://github.com/devsisters/DQN-tensorflow.git` 하고나서 Python 2.7.x 환경에서 repository에서 나와있는대로 실행하면 잘 돌아갑니다.

하지만, `Windows 7`이나 `10`에서 위의 코드를 바로 돌리려고 하면 두 가지 종류의 오류 때문에 실행되지 않습니다.

  1.  Windows 파일 경로 길이 제한
  2.  Python 2와 3 호환성

## Windows 7 파일 경로 길이 제한
**로컬로 가져올 때:** 하지만, repository에 있는 checkpoints 안에 있는 하위 폴더 경로가 길어서 Windows 10에서는 경로명 제한을 해제한 후에 `clone` 해야하고 Windows 7에서는 압축파일을 받아서 적당한 곳에 직접 풀어서 진행해야합니다.

**실행할 때:** Windows 7에서는 경로 길이 제한 때문에 checkpoints 안에 매개변수값과 관련된 폴더를 만들다가 오류가 생깁니다. `dqn/base.py`에서 600-63 번째 줄에 있는 코드를 주석처리하면 이러한 오류는 발생하지 않습니다. 저는 60-63번째 줄을 주석처리하고 64번째 줄의 `return model_dir + '/'`을 `return model_dir + '20170408/'`로 바꿨습니다. [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)의 코드는 `config.py`에 있는 매개변수별로 폴더를 만들지만 이 글에서는 매개변수들을 바꾸지 않고 실행해보는 것 자체에 관심이 있기 때문에 경로를 위와 같이 하였습니다.


## Python 2 → Python 3
[DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow) 여기 있는 코드가 Python 3.x 에서 돌아가지 않으니 다음과 같이 바꿉니다.

1. `environment.py` 와 `agent.py`에 있는 `xrange`를 `range`로 바꿉니다.
2. `agent.py`에서 201과 248번째 줄의 `reduce` 를 사용하기 위해 2번째 줄에 `import functools`을 추가하고 `reduce`를 사용하는 부분을 `functools.reduce`로 바꿉니다.
3. `agent.py`에서 328번째 줄의 `self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)`을 `self._saver = tf.train.Saver([v for k, v in self.w.items()] + [self.step_op], max_to_keep=30)`로 바꿉니다.


## 실행
`python main.py --is_train=True --use_gpu=True --display=False`를 실행하면 다음과 같이 훈련을 시작합니다. 기본값은 벽돌깨기 게임(`Breakout-v0`)입니다. `--env_name=Breakout-v0`을 옵션으로 주지 않은 것에 주목하시길 바랍니다.

![]({{ site.baseurl }}/images/2017-04-08/Python3.5.2-amd64_tensorflow-gpu1.1.0rc_NVIDIA_960M_Windows10-part1.png)

NVIDIA 960M을 가지고 훈련/학습을 시작합니다. 중간에 보이는 `The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.`라는 메시지는 tensorflow-gpu 소스 코드를 받아서 직접 컴파일하면 나타나지 않습니다. 소스 코드를 받아서 적절한 옵션으로 컴파일하면 CPU가 지원하는 vectorization을 이용할 수 있습니다. Linux에서는 테스트해보았으나 Windows에서는 아직 해보지 않았습니다. 자세한 내용은 [Installing TensorFlow from Sources](https://www.tensorflow.org/install/install_sources)를 참조하시길 바랍니다.

![]({{ site.baseurl }}/images/2017-04-08/Python3.5.2-amd64_tensorflow-gpu1.1.0rc_NVIDIA_960M_Windows10-part2.png)

위의 사진은 학습이 진행되는 과정입니다. 중간에 checkpoints 가 저장됩니다. `config.py`에서 저장하는 빈도를 조절할 수 있습니다.

## 이 글에서 사용한 환경
- `Windows 7`과 `10` 환경 각각에서 테스트 했습니다.
- `Python 3.5.2`
- `tensorflow-gpu 1.1.0rc`
- NVIDIA 960M + CUDA 8.0.44 + cuDNN v5.1
- `MSYS2` + `CMAKE`로 미리 컴파일한 `atari-py`
- `pip install gym[atari]==0.7.0`
- [여기](http://www.lfd.uci.edu/~gohlke/pythonlibs/)에서 받을 수 있는 `PyOpenGL_accelerate‑3.1.1‑cp35‑cp35m‑win_amd64.whl`

`텐서플로우 0.10 + 우분투 16.04 + CUDA 8.0 + 파이썬 3.5 설치` 관련 내용은 ["파이썬-킴"님의 글](http://pythonkim.tistory.com/71)을 참조하시길 바랍니다.


## 기타 사항
이 글에서 언급한 변경 사항들은 [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)을 `fork`하여 [여기](https://github.com/danielykim/DQN-tensorflow)에 올려두었습니다.

