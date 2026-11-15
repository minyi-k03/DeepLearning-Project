# progress.md

| ID | Task | 담당 | 브랜치 | 상태 | 마지막 verify | 메모 |
| --- | --- | --- | --- | --- | --- | --- |
| T-01 | 저장소 구조·계약·하네스 구축 | 공통 | `feat/T-01a-env-lock` | 진행 | 2026-11-15 통과 | T-01a: Python 3.11 + `requirements.lock`, 차가원 Mac에서 새 환경 verify 통과 — 고용민 컴퓨터 확인 대기. T-01b: CODEOWNERS 활성화 완료, main 보호·테스트 PR 확인 대기 |
| T-02 | 1주차 환경 확인 (`week1_env.yaml`) | 인프라·모델 | — | 대기 | — | `transformers` 5.x 마지막 층 벡터 정의·IMDB 최대 길이 확인 필요 |
| T-03 | 데이터·입력 구성 모듈 (`data.py`, `inputs.py`) | 모델·인프라 | — | 대기 | — | |
| T-04 | 층 벡터 추출기·캐시·공급원 (`extraction.py`, `providers.py`) | 인프라 | — | 대기 | — | |
| T-05 | 결합 헤드 (`fusion.py`) | 모델 | — | 대기 | — | F4 게이트 구체 형태 확정 필요 |
| T-06 | 실행 모듈·대기열·재현 트랙 (`runner.py`) | 인프라 | — | 대기 | — | |
| T-07 | 판정 규칙 고정·판정 모듈 (`judgment.py`) | 모델 | — | 대기 | — | 11/30 고정 예정 |
| T-08 | 재현 트랙 및 본 실험 순차 실행 | 인프라 | — | 대기 | — | |
| T-09 | 보고 모듈·보고서·발표 | 모델·공통 | — | 대기 | — | 12/14 최종 발표 |

## 세션 로그
- 2026-11-15 · 초기 스캐폴딩 생성: `SPEC.md`, `GLOSSARY.md`, `docs/adr/0001`–`0009`를 바탕으로 패키지 구조, 설정 템플릿, 테스트 스켈레톤, 하네스 파일 생성.
- 2026-11-15 · T-01a(차가원): Python 3.11 venv, `requirements.lock`(uv, universal) 생성, 새 환경에서 잠금 파일 설치 후 `python scripts/verify.py` ALL PASSED(5 passed, 26 skipped). README 설치 절차 갱신.
