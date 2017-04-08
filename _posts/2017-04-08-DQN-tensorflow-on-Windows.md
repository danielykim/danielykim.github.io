---
layout: post
comments: false
title: DQN-tensorflow on Windows 7 and 10
---

이 글의 목적은 고전 게임을 심층강화학습하는 코드인 [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)를 Windows 7이나 10에서 실행하는 방법을 소개하는 것입니다. 혹시 코드와 관련된 이론적인 배경이 궁금하시다면 [논문](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)을 직접 읽어보시거나 [천상혁님의 논문 리뷰](http://sanghyukchun.github.io/90/)를 참조하시길 바랍니다.


[DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow) 라는 제목에서 알 수 있듯이, 내부에서 Python 라이브러리인 tensorflow 를 사용합니다. 


Windows 7과 10 환경 각각에서 Python 3.5.2와 tensorflow-gpu 1.1.0rc를 이용하여 진행했습니다.


## Windows 7 파일 경로 길이 제한
[DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow) 여기에서 구할 수 있는 코드는 Linux 환경에서 `git clone https://github.com/devsisters/DQN-tensorflow.git` 하고나서 Python 2.7.x 환경에서 repository에서 나와있는대로 실행하면 잘 돌아갑니다. 하지만, repository에 있는 checkpoints 안에 있는 하위 폴더 경로가 길어서 Windows 10에서는 경로명 제한을 해제한 후에 `clone` 해야하고 Windows 7에서는 압축파일을 받아서 적당한 곳에 직접 풀어서 진행해야합니다.

Windows 7에서는 경로 길이 제한 때문에 checkpoints 안에 매개변수값과 관련된 폴더를 만들다가 오류가 생깁니다. 이러한 오류는 `dqn/base.py`에서 600-63 번째 줄에 있는 코드를 주석처리하면 발생하지 않습니다. 저는 60-63번째 줄을 주석처리하고 64번째 줄의

```Python
return model_dir + '/'
```

을 아래와 같이 임의로 바꿨습니다.

```Python
return model_dir + '20170408/'
```


## Python 2 → Python 3
[DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow) 여기 있는 코드가 Python 2.7.x 를 지원하니 다음과 같이 Python 3.x 로 바꿉니다.

1. `environment.py` 와 `agent.py`에 있는 `xrange`를 `range`로 바꿉니다.

2. `agent.py`에서 201과 248번째 줄의 `reduce` 를 사용하기 위해 2번째 줄에 

```Python
import functools
```

을 추가하고 `reduce`를 사용하는 부분을 `functools.reduce`로 바꿉니다.

3. `agent.py`에서 328번째 줄의 

```Python
self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)
```

을

```Python
self._saver = tf.train.Saver([v for k, v in self.w.items()] + [self.step_op], max_to_keep=30)
```

으로 바꿉니다.

