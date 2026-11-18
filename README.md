# Layer Fusion Study — ModernBERT-base (Track A)

고정된 `ModernBERT-base`의 여러 층 표현(임베딩 벡터 1개 + 22개 층 = 23개 층 벡터)을 어떻게 결합하느냐에 따라 분류 오차율이 어떻게 달라지는지 검증하는 심층학습 기말 프로젝트 저장소입니다.

기준 문서:
- 요구사항 및 테스트 경계: [`docs/SPEC.md`](docs/SPEC.md)
- 공용 용어집: [`GLOSSARY.md`](GLOSSARY.md)
- 설계 결정(ADR): [`docs/adr/`](docs/adr/)
- 작업 분해 및 Task별 진행 상황: [`docs/03-plan.md`](docs/03-plan.md)

## 역할 분담
- **모델 담당 (고용민)**: 결합 방식(`B0`, `F1`~`F4`, `B0+`), 층별 정규화, 분류기, 가설 판정·분석
- **인프라 담당 (차가원)**: 서버 환경, 층 벡터 추출·캐시, 순차 실행 대기열, 실행 기록 관리

## 폴더 구조
```text
├── GLOSSARY.md                # 용어 단일 기준
├── AGENTS.md / CLAUDE.md      # AI 에이전트 공통 규칙
├── configs/                   # 실험 행렬, 판정 규칙(11/30 고정), 1주차 환경 설정
├── docs/                      # SPEC.md(요구사항 단일 기준), ADR, WBS 계획·진행 현황(03-plan.md)
├── src/layer_fusion/          # 핵심 패키지 (contracts, layers, data, inputs, extraction, providers, fusion, runner, judgment, reporting)
├── scripts/                   # verify.py, sync_skills.py, CLI 진입점
├── tests/                     # 단위 테스트 및 Seam ①②③ 테스트
├── data/cache/                # 추출된 층 벡터 캐시 (git 미추적)
└── results/                   # 실행 레코드, 보고서 산출물, 재현 트랙 결과 (git 미추적)
```

## 설치 및 검증 (로컬 CPU)
**Python 3.11**을 사용합니다(CI와 동일, `.python-version`). 3.13 이상은 torch 등 일부 패키지 호환을 보장하지 않습니다.

```bash
python3.11 -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.lock     # 두 사람·GPU 서버가 같은 버전을 쓰도록 잠근 목록
pip install -e . --no-deps
python scripts/verify.py             # [verify] ALL PASSED 가 나오면 완료
```

- `requirements.txt`는 버전 범위(사람이 고치는 파일), `requirements.lock`은 정확한 버전(생성 파일)입니다. 의존성을 바꿀 때는 `requirements.txt`를 고친 뒤 아래 명령으로 잠금 파일을 다시 만들고 함께 커밋합니다.
  ```bash
  uv pip compile requirements.txt --universal --python-version 3.11 -o requirements.lock
  ```
- 잠금 파일은 macOS(arm64)·Linux·Windows 공통입니다. Linux GPU 서버에서 CUDA 버전에 맞는 torch가 필요하면 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)의 `--index-url`로 torch만 잠금 파일과 같은 버전으로 다시 설치합니다.
- `transformers` 버전은 T-02a(1주차 환경 확인)에서 확정한 뒤 `configs/week1_env.yaml`과 잠금 파일에 함께 반영합니다.

확인된 환경 (2026-11-15): macOS arm64 / Python 3.11.15 / torch 2.15.0 (MPS 사용 가능) / transformers 5.23.0. 새 환경에서 잠금 파일로 설치한 뒤 `verify` 통과.

## GPU 서버 (공용, 개발 컨테이너)
연구실 서버 `metis`(RTX 5090 1장, 32GB)는 두 사람이 나눠 씁니다. 각자 자기 서버 계정으로 접속해 **자기 폴더·자기 컨테이너**를 씁니다. 설정은 `.devcontainer/devcontainer.json` 하나를 같이 쓰고, 컨테이너 이름은 서버 계정 이름으로 자동으로 붙습니다(`layer-fusion-<계정>`).

처음 한 번 (서버 터미널):
```bash
nvidia-smi                                    # GPU를 누가 쓰고 있는지 확인
mkdir -p ~/.cache/huggingface                 # 모델 가중치 보관 (컨테이너와 공유)
git clone https://github.com/minyi-k03/DeepLearning-Project.git ~/DeepLearning-Project
```
그다음 VS Code(Remote-SSH)에서 **파일 → 폴더 열기 → `~/DeepLearning-Project`** → **F1 → `Dev Containers: Reopen in Container`**. 처음에는 패키지 설치로 10~20분 걸립니다. 컨테이너 안 터미널에서 확인합니다.
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/verify.py
```

GPU 사용 규칙 (ADR 0009):
- **디버깅 실행**(`run_kind=debug`)은 둘이 동시에 돌려도 됩니다. 메모리 32GB를 넘지 않게 합니다.
- **본 실험**(학습 시간을 재는 실행)은 인프라 담당(차가원)이 대기열로 한 번에 하나씩 돌립니다. 그동안 다른 GPU 작업은 하지 않습니다.
- 컨테이너마다 CPU 8코어·메모리 24GB 상한이 걸려 있습니다(서버 20코어·61GB). 넘으면 내 컨테이너 안의 프로세스만 종료됩니다.
- GPU 메모리는 도커로 제한할 수 없습니다. 실행 전에 `nvidia-smi`로 다른 사람의 사용량을 확인합니다.
- 다 쓰면 VS Code 창을 닫거나 `docker stop layer-fusion-$USER`로 컨테이너를 멈춥니다.
- 다른 프로젝트의 컨테이너는 건드리지 않습니다.

## Skills 동기화
프로젝트 스킬 원본은 `.agents/skills/`에 두며, `.claude/skills/`는 `scripts/sync_skills.py`로 동기화합니다.
```bash
python scripts/sync_skills.py --check
```

| 역할 | 스킬 | 출처 | 비고 |
| --- | --- | --- | --- |
| — | (미설치) | — | 필요 시 팀 검토 후 `.agents/skills/`에 추가 |
