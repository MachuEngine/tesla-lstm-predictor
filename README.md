# tesla-lstm-predictor
이 프로젝트는 LSTM(Long Short-Term Memory) 모델을 사용하여 Tesla (TSLA) 주가를 예측하는 예제입니다. Yahoo Finance에서 TSLA의 역사적 데이터를 다운로드하여 데이터 전처리, 시퀀스 생성, 모델 학습, 평가 및 시각화를 수행합니다.

### 주요 기능
데이터 다운로드 및 전처리:
Yahoo Finance에서 TSLA 주가 데이터를 다운로드하고, `MinMaxScaler`를 사용하여 주요 피처(`Close`, `Volume`, `Open`, `High`, `Low`)를 스케일링합니다.

시퀀스 데이터 생성:
일정 시간 단위로 시퀀스 데이터를 생성하여 LSTM 모델의 입력으로 사용합니다.

LSTM 모델 정의 및 학습:
구성 파일(`config.yaml`)에 지정된 하이퍼파라미터에 따라 LSTM 모델을 정의하고, 학습합니다.

모델 평가 및 시각화:
학습된 모델을 평가하고, 예측 결과와 실제 데이터를 시각화합니다.

모듈화:
코드가 `src` 디렉토리 내 여러 모듈(`utils`, `dataset`, `model`, `train`, `eval`, `visualization`)로 구성되어 있어, 유지보수 및 확장이 용이하도록 하였습니다. 


### 파일 구조
```bash
.
├── configs
│   └── config.yaml       # 학습 및 로깅 관련 설정 파일
├── main.py               # 프로젝트 진입점
└── src
    ├── dataset.py        # 데이터 다운로드 및 전처리 모듈
    ├── eval.py           # 모델 평가 모듈
    ├── model.py          # LSTM 모델 정의 모듈
    ├── train.py          # 모델 학습 모듈
    ├── utils.py          # 유틸리티 함수 (설정 로드, 로깅 설정 등)
    └── visualization.py  # 결과 시각화 모듈

```

### 설치 방법
1. 레포지토리 클론
터미널에서 아래 명령어를 실행하여 레포지토리를 클론합니다.

```bash
git clone https://github.com/MachuEngine/tesla-lstm-predictor.git
cd tesla-lstm-predictor
```

2. 의존성 설치
Python 3.10 이상 버전을 사용하고, 필요한 패키지를 설치합니다.
```
pip install -r requirements.txt

```

### 설정
* configs/config.yaml
이 파일은 데이터 경로, 학습 하이퍼파라미터(에포크 수, 학습률, 입력/출력 크기, hidden size 등), 로깅 설정, 체크포인트 저장 간격 등을 정의합니다.

```yaml
data:
  path: "./data"

train:
  num_epochs: 200
  learning_rate: 0.001
  input_size: 5
  hidden_size: 64
  num_layers: 2
  output_size: 1

logging:
  level: INFO
  file: "./logs/train.log"

checkpoint:
  directory: "./checkpoints"
  save_interval: 5
```

### 사용 방법

1. 메인 스크립트 실행
터미널에서 다음 명령어를 실행하여 프로젝트를 시작합니다.

```bash
python main.py
```
main.py에서는 src/utils.py의 설정 로드 및 로깅 설정,
src/dataset.py의 데이터 다운로드 및 전처리,
src/model.py의 LSTM 모델 생성,
src/train.py의 모델 학습,
src/eval.py의 평가,
src/visualization.py의 결과 시각화를 순차적으로 호출합니다.

2. 결과 확인
학습이 완료되면, 터미널에 학습 손실과 예측 결과가 출력되며, 시각화 창이 열려 예측 값과 실제 값의 비교 그래프를 확인할 수 있습니다.

### 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.