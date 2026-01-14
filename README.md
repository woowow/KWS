# KWS 데이터 수집

이 프로젝트는 **키워드 스팟팅(KWS: Keyword Spotting)** 학습/평가용 음성 데이터를 수집합니다.  
각 참여자는 지정된 **화자 번호(`spkXX`)** 로 구분되며, 정해진 **라벨(labels)** 별로 30번씩 녹음합니다(Unknown은 70번).

---

## 1) 수집 대상(라벨)

### 명령어(키워드)
- `next`
- `prev`
- `stop`
- `play`

### 비명령어(unknown)
- `unknown` : 아무 말 / 요리하다가 나올만한 말 포함 가능 (오탐 방지 목적)

---

## 2) 설치 (requirements.txt)

프로젝트 루트에서 아래 순서대로 실행하세요.

### 2.1 가상환경

**Windows (PowerShell)**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```
### 2.2 의존성 설치
pip install -r requirements.txt

## 3) 실행 위치 / 기본 실행 형태

kws-project/train/에서 실행

## 4) 옵션(인자) 설명

ex)
```bash
python scripts/record_kws.py --spk spk02 --labels unknown --append --count 150
```

### 4.1 --spk

화자 ID(사람 구분용)

예: spk01, spk02, spk03 …

반드시 본인에게 배정된 번호로 실행

### 4.2 --labels

수집할 라벨(클래스) 목록

모델은 “오디오 → 라벨” 분류를 학습하므로, 각 녹음 파일이 어떤 라벨인지 정확히 붙어야 함

예시:

--labels next prev stop play : 명령어 키워드 녹음

--labels unknown : 명령어가 아닌 말(잡담/요리 중 말 등) 녹음

KWS는 실제 환경에서 명령어가 아닌 말들을 명령어로 착각하는 경우가 많음

그래서 unknown 데이터를 충분히 넣어야 오탐이 줄어듬

### 4.3 --count

각 라벨당 녹음 횟수

ex) --labels next prev stop play --count 50
→ next 50 + prev 50 + stop 50 + play 50 (총 200개)

### 4.4 --append

이미 같은 spk로 저장된 데이터가 있으면 덮어쓰지 않고 뒤에 추가 저장

중간에 끊어서 다시 진행할 수도 있으므로 웬만하면 항상 --append 사용

## 5) 무엇을 어떻게 녹음하나요?

### 5.1 명령어 라벨: next / prev / stop / play

각 명령어를 아래 발화 스타일을 포함해서 다양하게 녹음해주세요.

평소 말하듯

빠르게

천천히

또박또박

살짝 뭉퉁그려서 (여러 발음)

***각 단계별 발화 단어는 아래와 같습니다.***

next: "다음 단계"

prev: "이전 단계"

stop: "일시 정지"

play: "이어 하기"

### 5.2 unknown 라벨: unknown (명령어가 아닌 말)

아무 말을 녹음합니다.
요리하다가 나올만한 말이어도 괜찮습니다.

아무 문장 (예: “오늘 뭐 먹지”, “배고프다”)

요리 관련 말 (예: “소금 어디있지”, “물 좀 더 넣자”, “불 줄여야겠다”)

감탄/추임새 (예: “아 뜨거”, “음 맛있다”)

---
## ⚠️ 주의사항

**unknown에는 next/prev/stop/play 단어를 넣지 말아주세요.**

### 명령어를 입력하면 argsment 정보가 출력되고 바로 녹음 준비를 시작합니다.

### 말할 준비! 3... 2... 1... 이후 RECORDING (1.50s)... 이 나오면 1.5초간 녹음이 진행됩니다.

### ⚠️ 본인이 타이밍을 놓쳤다거나 잘 못 말했다면 재녹음하거나 반드시 경로로 이동해서 직접 삭제해주세요.
---

## 6) 실행 명령어 (그대로 복붙하고 본인번호만 수정해서 사용)

### "다음 단계" 녹음
python scripts/record_kws.py --spk spk본인번호 --labels next --append --count 30
### "이전 단계" 녹음
python scripts/record_kws.py --spk spk본인번호 --labels prev --append --count 30
### "일시 정지" 녹음
python scripts/record_kws.py --spk spk본인번호 --labels stop --append --count 30
### "이어 하기" 녹음
python scripts/record_kws.py --spk spk본인번호 --labels play --append --count 30

### 아무 소리 녹음
python scripts/record_kws.py --spk spk02 --labels unknown --append --count 70


## 7) 녹음 완료 시

kws-project/train/data/raw/spk/ 안에 본인이 녹음한 파일들이 저장되어있습니다.

각 파일들을 압축해서 MM 보내주세요.