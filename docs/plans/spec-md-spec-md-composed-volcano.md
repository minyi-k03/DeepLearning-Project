# 계획: `docs/03-plan.md` 작성 (아키텍처 · 담당 · Task 분해)

## Context

`SPEC.md`(ready-for-agent), `GLOSSARY.md`, ADR 0001·0003~0009가 확정되었고 `src/layer_fusion/` 스켈레톤·설정 템플릿·skip된 seam 테스트까지 만들어져 있다(`docs/progress.md` T-01 진행). 그러나 `AGENTS.md`가 세션 시작 시 읽으라고 지정한 `docs/03-plan.md`가 아직 없고, `progress.md`의 T-01~T-09는 PR 하나로 병합·검증하기엔 너무 크다. 가이드(`03-capstone-project.md` §4-3)의 4단계 산출물 형식대로 **아키텍처 + 담당 표 + 마일스톤 + Task 목록 표**를 `docs/03-plan.md` 한 파일에 만든다. 12/14 발표에서 거꾸로 계산하고, 11/30 판정 규칙 고정(ADR 0007)을 축으로 삼는다.

## 작업 범위

- **새로 만드는 파일: `docs/03-plan.md` 하나** (AGENTS.md가 참조하는 경로. 별도 `03-architecture.md`는 만들지 않고 아키텍처를 이 파일 §1에 둔다).
- `docs/progress.md`, 코드, 설정, 테스트는 **수정하지 않는다**. (진행 상태 변경은 progress.md에서만, Task 삭제·완료 조건 변경 금지 규칙 준수. 하위 Task 행 추가는 승인 후 별도 작업으로 제안)
- 저장소가 아직 git이 아니므로 커밋은 하지 않는다(T-01b로 Task화).
- **Task ID 방침**: 기존 T-01~T-09를 상위 Task(묶음)로 유지하고 `T-04a`처럼 하위 Task로 쪼갠다. 테스트 skip 사유(`TODO(T-05)` 등)와 progress.md ID가 그대로 맞는다.
- `SPEC.md` §결정 필요 사항 값은 지어내지 않고, 어떤 Task를 막는지·언제까지 정할지만 적는다.

## `docs/03-plan.md` 구성 (작성할 내용)

### §0 머리말
근거 문서, 상태(초안 → 팀 승인 후 사람만 표 수정), 상태 추적은 `progress.md`에서만.

### §1 아키텍처
- **모듈 흐름도(ASCII)**:
  `data.py → inputs.py → extraction.py → (캐시) → providers.py → [실행 seam] runner.py(fusion.py 사용) → ResultRecord(JSON) → [판정 seam] judgment.py(+decision_rules.yaml) → reporting.py`
  파인튜닝/재현: `BackboneProvider → runner.py`, 재현 트랙은 `track=reproduction`으로 판정 입력에서 제외.
- **모듈 ↔ 파일 ↔ 담당 ↔ SPEC 모듈 번호 표** (11개 모듈을 기존 파일에 매핑: `layers.py`는 기구현·테스트 통과 상태).
- **두 seam 계약**: `run_experiment(RunConfig, LayerVectorProvider) -> ResultRecord`, `judge_hypotheses(records, DecisionRules) -> JudgmentReport` (`src/layer_fusion/contracts.py` 기준). 둘 다 CPU·가짜 공급원/합성 레코드로 테스트.
- **의존 규칙**: judgment/reporting은 torch·모델을 import하지 않음 / runner는 공급원 인터페이스만 앎 / 처리 순서(최종 정규화 → 패딩 제외 토큰 평균 → 층별 정규화) 고정 / 계약 변경은 SPEC·contracts.py·테스트 먼저.
- **설정 파일 역할**: `week1_env.yaml`(1주차 확정값), `experiment_matrix.yaml`(실행 목록), `decision_rules.yaml`(11/30 locked).

### §2 담당 표
| 담당 | 범위 |
|---|---|
| 고용민(모델) | 데이터 모듈, 결합 헤드, 판정 모듈, 보고 모듈, 분석·해석 |
| 차가원(인프라) | 입력 구성, 1주차 환경 고정, 추출·캐시·공급원, 실행 모듈·대기열·재현 트랙, 본 실험 순차 실행(ADR 0009) |
| 공통 기반 | 차가원: 환경·의존성 잠금, `scripts/verify.py`, `configs/week1_env.yaml`, `providers.py` 인터페이스 / 고용민: `contracts.py`, `layers.py`, `judgment.py`, `configs/decision_rules.yaml` — 모두 상대 리뷰 필수(AGENTS.md §4) |
| 병합 | PR 작성자가 상대 승인 후 병합. 공통 파일 PR은 상대 + 다른 계열 모델 리뷰 |
| 에이전트 | 담당자가 Task마다 선택(가이드 §4-0) |

### §3 마일스톤 (12/14에서 역산)
| M | 기간 | 목표 | 게이트 |
|---|---|---|---|
| M0 | 11/15–11/17 | 하네스·git·환경 잠금, verify 통과(skip 포함) | verify ALL PASSED |
| M1 | 11/16–11/22 (1주차) | 1주차 확인, CPU 모듈(데이터·입력·결합 헤드·공급원·판정 코어) | week1_env.yaml 채움, Seam ①② 일부 un-skip |
| M2 | 11/23–11/30 | 추출·캐시, 실행 seam, 대기열, 디버깅 E2E, 재현 트랙 실행, 판정 규칙 고정 | **11/30: decision_rules.yaml `locked` + 실행 경로 코드 기능 동결** |
| M3 | 12/1–12/9 | 본 실험 순차 실행 MNLI → SST-2 → IMDB | 결과 레코드 수집 완료 |
| M4 | 12/8–12/12 | 판정·보고 산출물·보고서·발표 | 보고서 4–6쪽 초안 |
| 버퍼 | 12/12–12/13 | 재실행·수정 | 12/14 발표 |

### §4 Task 목록 표
열: `ID | 제목 | 스토리(US) | M | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 파일 | 위험 | 병렬`

- **T-01 공통 기반**
  - T-01a Python 환경·의존성 잠금, README 설치 절차 (인프라, M0, verify 통과, 공통○, 보통)
  - T-01b git 초기화·main 보호·CODEOWNERS(공통 파일 8개) (인프라, M0, 높음)
  - T-01c `contracts.py` 계약 검토·합의 (모델, 리뷰 인프라, M0, 높음)
  - T-01d 문서 정합성: ADR 0002 파일 누락 복구, ADR 0003 "SST-2 판정" 문구 정리 (공통, M0, 낮음)
- **T-02 1주차 확인**
  - T-02a transformers 5.x·어텐션·컴파일 고정 + `verify_final_norm_contract` 구현, Seam③ 첫 테스트 un-skip (인프라, US24·50, GPU, 높음)
  - T-02b IMDB 최대 토큰 길이 측정 → Lfull 상한·잘린 비율 기록 (인프라, US15·16, Blocked T-03b의 토크나이저 부분 또는 단독 스크립트)
  - T-02c 교수님 MNLI·SST-2 승인 확인 → `dataset_approval` 기입 (팀, US10·11, 수동)
- **T-03 데이터·입력**
  - T-03a 데이터 모듈: 3개 데이터셋 doc_id 부여, MNLI matched/mismatched, 튜닝 분할 10%, 소량 표본(시드별·클래스당 1,000·ID 기록) (모델, US1·3~9, CPU 단위 테스트, 병렬○)
  - T-03b 입력 구성: MNLI [SEP] 단일 입력, 앞+뒤 자르기 L128/L512, Lfull 상한, 길이·구간·시야 비율 (인프라, US2·12~19, `test_head_tail_truncation_budgets` un-skip, 병렬○)
- **T-04 추출·캐시·공급원**
  - T-04a 공급원 인터페이스 + `FakeLayerVectorProvider.make_synthetic` + 메타 불일치 오류 (인프라, US26, CPU, `test_mismatched_provider_metadata_raises_error`, 공통○, 병렬○)
  - T-04b 추출기·캐시 writer·`CacheProvider` ([N,23,768] + doc_id·라벨 + 생성 설정) (인프라, US21~27, Blocked T-02a·T-03b, Seam③ 나머지 5개 un-skip, 높음)
- **T-05 결합 헤드**
  - T-05a 층별 정규화 + B0/F1/F2(전체·전역만·지역만)/F3 + 파라미터 수 + 층 가중치 노출 (모델, US28~32·35·37, Seam① 3개 un-skip, 병렬○)
  - T-05b F4 게이트 + B0+ 파라미터 맞춤 (모델, US33·34·36, Blocked: F4 형태·가중치 정규화·허용 차이 결정, `test_trainable_parameter_counts...`)
- **T-06 실행 모듈**
  - T-06a 실행 seam(백본 고정): LR 3후보 튜닝 분할 선택, 동일 에폭·조기 종료, MNLI matched/mismatched, 학습 시간·곡선·환경 정보 기록 (인프라, US39~43·52~54, Blocked T-04a·T-05a·T-03a, Seam① 나머지 un-skip, 높음)
  - T-06b 결과 레코드 직렬화 + `experiment_matrix.yaml` → RunConfig 전개 + `run_experiment.py` CLI (인프라, US48, Blocked T-06a)
  - T-06c 대기열 실행기: main 동시 실행 차단(잠금), run_kind/track 기록, `run_queue.py` (인프라, US47~49·51, ADR 0009, 높음)
  - T-06d 파인튜닝 경로: `BackboneProvider` + 결합 헤드 학습 (인프라·모델, US44, Blocked T-04b·T-05a, GPU)
  - T-06e 재현 트랙 실행기: 논문 설정 파인튜닝 ×3, `track=reproduction` (인프라, US45·46, Blocked T-02a, 논문 하이퍼파라미터 확인 필요)
- **T-07 판정**
  - T-07a 판정 규칙 문서 고정: `decision_rules.yaml` TBD 전부 채우고 `status: locked` (팀, US55, **기한 11/30**, 공통○, 높음)
  - T-07b 판정 코어: 규칙 로더, 짝지은 McNemar, 표준편차 규칙, 지지/효과 없음, debug·repro 제외, 탐색적 무판정 (모델, US56~59, Seam② 6개 un-skip, 합성 레코드, 병렬○)
  - T-07c 가설별 판정·분석: H1·H2·H3·H4′·H5a·H5b(섞기/가중치 학습 분해), 길이 구간 오차율·최소 표본, 무게중심·전역 비율·시드 일관성, F4 길이–가중치 순위 상관 (모델, US60~72, Seam② 나머지 3개, Blocked T-07b, 규칙값은 T-07a)
- **T-08 실행**
  - T-08a 디버깅 E2E 스모크(소규모, run_kind=debug) (인프라, M2, Blocked T-04b·T-06a~c)
  - T-08b 층 벡터 캐시 생성: MNLI·SST-2·IMDB×3 입력 설정 (인프라, M2, Blocked T-04b·T-02b)
  - T-08c 재현 트랙 실행 MNLI·SST-2 ×3 (인프라, M2, Blocked T-06e) — 본 실험 시작 전 완료
  - T-08d 본 실험 MNLI 고정 전체 B0~B0+ (H1·H4′) (인프라, M3, Blocked T-07a·T-08b)
  - T-08e MNLI 소량 (H2) (M3, Blocked T-08d와 독립, 결합 방식 목록 결정 필요)
  - T-08f MNLI 파인튜닝 B0 vs 가장 좋은 결합 방식 (H3) (M3, Blocked T-08d 결과 + T-06d + 선정 기준)
  - T-08g SST-2 고정 (H4 탐색) (M3, SST-2 승인 시만)
  - T-08h IMDB L128/L512/Lfull × B0·F1·F2, 전역/지역 ablation, Lfull F4 (H5a·H5b·H5c) (M3)
- **T-09 보고**
  - T-09a 보고 모듈: 결과표(오차율±표준편차·파라미터 수·학습 시간), 학습 곡선, 층 가중치 그림, 길이 구간·시야 비율 표 (모델, US73·74, 합성 레코드로 M2부터, 병렬○)
  - T-09b 판정 실행·보고서 4–6쪽(한계 절 US75 포함) (공통, M4)
  - T-09c 발표 자료 (공통, M4)

병렬 표시: M1에서 T-03a ∥ T-03b ∥ T-05a ∥ T-04a ∥ T-07b ∥ T-02a 동시 진행 가능.

### §5 의존 그래프(ASCII)·임계 경로
`T-01a → T-02a → T-04b → T-08b → (T-07a 11/30) → T-08d → T-08f → T-09b`
GPU 작업은 ADR 0009에 따라 인프라 1인 순차.

### §6 결정 필요 사항 → 막는 Task·기한 표
SPEC §결정 필요 사항 각 항목을 "막는 Task / 정할 시점(1주차 · 11/30 · 구현 단계) / 기록 위치(week1_env.yaml · decision_rules.yaml · SPEC)"로 매핑. 값은 쓰지 않음. 추가로 재현 트랙의 논문 하이퍼파라미터(SPEC에 값 없음)를 T-06e 선행 확인 항목으로 표기.

### §7 위험·충돌 보고
- ADR 0002가 SPEC·AGENTS에 인용되나 `docs/adr/`에 없음 → T-01d
- ADR 0003 vs ADR 0007·GLOSSARY의 H4 판정 표현 불일치(SPEC 추가 메모와 동일) → T-01d
- 저장소가 git이 아님 → 브랜치 규칙 적용 불가 → T-01b
- M3 GPU 예산(MNLI 파인튜닝 B0·best × 시드 수 + 재현 6회)이 문서에 추정치 없음 → 일정 밀리면 MNLI > SST-2 > IMDB(ADR 0003)
- T-08f는 T-08d 결과에 의존(순차) → 선정 기준을 11/30에 먼저 고정
- 캐시 용량(MNLI float32 약 28GB) → 저장 정밀도 결정이 T-08b를 막음

## 확인 방법

- 작성 후 `python scripts/verify.py` 실행(문서만 추가하므로 기존 결과와 같아야 함 — 실제 실행 결과를 그대로 보고).
- 수동 점검: SPEC 사용자 스토리 1~75가 모두 어느 Task에 매핑되는지, 테스트 파일의 skip 사유 Task ID(T-02~T-07)가 하위 Task와 일치하는지, `§결정 필요 사항`의 값을 지어낸 곳이 없는지.
- 완료 보고는 AGENTS.md §9 형식, 공통 파일 변경 없음 명시, progress.md 하위 Task 반영은 사람 승인 사항으로 남김.
