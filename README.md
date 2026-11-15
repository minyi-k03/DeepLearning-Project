# Layer Fusion Study — ModernBERT-base (Track A)

고정된 `ModernBERT-base`의 여러 층 표현(임베딩 벡터 1개 + 22개 층 = 23개 층 벡터)을 어떻게 결합하느냐에 따라 분류 오차율이 어떻게 달라지는지 검증하는 심층학습 기말 프로젝트 저장소입니다.

기준 문서:
- 요구사항 및 테스트 경계: [`docs/SPEC.md`](docs/SPEC.md)
- 공용 용어집: [`GLOSSARY.md`](GLOSSARY.md)
- 설계 결정(ADR): [`docs/adr/`](docs/adr/)
- 작업 분해 및 진행 상황: [`docs/03-plan.md`](docs/03-plan.md), [`docs/progress.md`](docs/progress.md)

## 역할 분담
- **모델 담당 (고용민)**: 결합 방식(`B0`, `F1`~`F4`, `B0+`), 층별 정규화, 분류기, 가설 판정·분석
- **인프라 담당 (차가원)**: 서버 환경, 층 벡터 추출·캐시, 순차 실행 대기열, 실행 기록 관리

## 폴더 구조
```text
├── GLOSSARY.md                # 용어 단일 기준
├── AGENTS.md / CLAUDE.md      # AI 에이전트 공통 규칙
├── configs/                   # 실험 행렬, 판정 규칙(11/30 고정), 1주차 환경 설정
├── docs/                      # SPEC.md(요구사항 단일 기준), ADR, WBS 계획(03-plan.md), 진행 현황(progress.md)
├── src/layer_fusion/          # 핵심 패키지 (contracts, layers, data, inputs, extraction, providers, fusion, runner, judgment, reporting)
├── scripts/                   # verify.py, sync_skills.py, CLI 진입점
├── tests/                     # 단위 테스트 및 Seam ①②③ 테스트
├── data/cache/                # 추출된 층 벡터 캐시 (git 미추적)
└── results/                   # 실행 레코드, 보고서 산출물, 재현 트랙 결과 (git 미추적)
```

## 설치 및 검증 (로컬 CPU)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/verify.py
```

## Skills 동기화
프로젝트 스킬 원본은 `.agents/skills/`에 두며, `.claude/skills/`는 `scripts/sync_skills.py`로 동기화합니다.
```bash
python scripts/sync_skills.py --check
```

| 역할 | 스킬 | 출처 | 비고 |
| --- | --- | --- | --- |
| — | (미설치) | — | 필요 시 팀 검토 후 `.agents/skills/`에 추가 |
