# progress.md

| ID | Task | 담당 | 브랜치 | 상태 | 마지막 verify | 메모 |
| --- | --- | --- | --- | --- | --- | --- |
| T-01 | 저장소 구조·계약·하네스 구축 | 공통 | main (초기 스캐폴딩) | 진행 | 미실행 | `SPEC.md` 기반 폴더·스켈레톤 생성 완료, Python 실행 환경 확인 필요 |
| T-02 | 1주차 환경 확인 (`week1_env.yaml`) | 인프라·모델 | `feat/T-02b-imdb-length` | 진행 | 2026-11-15 통과 | T-02b 완료(리뷰 대기): IMDB 최대 3,270 토큰 ≤ 4,096 → Lfull 잘림 0건. T-02a(transformers 버전·최종 정규화 확인)·T-02c(교수님 승인) 남음 |
| T-03 | 데이터·입력 구성 모듈 (`data.py`, `inputs.py`) | 모델·인프라 | — | 대기 | — | |
| T-04 | 층 벡터 추출기·캐시·공급원 (`extraction.py`, `providers.py`) | 인프라 | — | 대기 | — | |
| T-05 | 결합 헤드 (`fusion.py`) | 모델 | — | 대기 | — | F4 게이트 구체 형태 확정 필요 |
| T-06 | 실행 모듈·대기열·재현 트랙 (`runner.py`) | 인프라 | — | 대기 | — | |
| T-07 | 판정 규칙 고정·판정 모듈 (`judgment.py`) | 모델 | — | 대기 | — | 11/30 고정 예정 |
| T-08 | 재현 트랙 및 본 실험 순차 실행 | 인프라 | — | 대기 | — | |
| T-09 | 보고 모듈·보고서·발표 | 모델·공통 | — | 대기 | — | 12/14 최종 발표 |

## 세션 로그
- 2026-11-15 · 초기 스캐폴딩 생성: `SPEC.md`, `GLOSSARY.md`, `docs/adr/0001`–`0009`를 바탕으로 패키지 구조, 설정 템플릿, 테스트 스켈레톤, 하네스 파일 생성.
- 2026-11-15 · T-02b(차가원): `scripts/measure_imdb_lengths.py`로 IMDB train/test 50,000건 측정. 최대 3,270(train)/2,971(test) 토큰 → `week1_env.yaml` `imdb_length_check` 채움(상한 4,096, 잘림 0건). 결과 `docs/measurements/imdb_length_check.json`. verify ALL PASSED.
