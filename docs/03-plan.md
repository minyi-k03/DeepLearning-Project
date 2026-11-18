# 03-plan — 아키텍처 · 담당 · 작업 분해 (층 결합 연구, Track A)

> 근거: `docs/SPEC.md`(기준), `GLOSSARY.md`, `docs/adr/0001`·`0003`~`0009`, `docs/03-capstone-project.md` §4-3
> 상태: **초안 (팀 승인 대기)**. 승인 이후 Task 표의 **상태 열**과 §8 진행 기록만 갱신한다(에이전트 포함). 완료 조건·마감·담당 등 나머지는 사람만 고치고, Task는 삭제하지 않는다. 진행 상태는 이 문서에서만 확인한다.
> 용어는 `GLOSSARY.md`를 따른다. `docs/SPEC.md` §결정 필요 사항의 값은 여기서 정하지 않는다(§6에서 어느 Task를 막는지만 적음).

---

## 0. 읽는 법

- Task ID는 **T-01~T-09를 상위 Task(묶음)로 두고**, PR 하나로 병합·검증할 수 있는 크기로 `T-04a`처럼 하위 Task를 나눴다. 테스트 skip 사유(`TODO(T-05)` 등)의 ID와 맞는다.
- "US"는 `docs/SPEC.md` 사용자 스토리 번호다.
- 담당은 사람만 적는다. Task마다 쓸 에이전트는 담당자가 고른다.
- 일정은 모두 날짜로 적는다. SPEC의 "1주차 확인"은 **11/16(월)–11/22(일)**로 잡았다(가정. 다르면 팀이 고친다).
- Task 표의 "마감"은 **그 날짜까지 완료 조건을 채운다**는 뜻이다(코드 Task는 PR 병합까지). 시작일은 Blocked by가 풀리면 담당자가 정한다. 마일스톤 기간은 §3에 있다.
- "상태"는 아래 다섯 값 중 하나로 시작하고, 필요하면 괄호로 브랜치·PR·막힌 이유를 덧붙인다. 눈에 띄도록 기호를 붙이고, 대기 외에는 굵게 쓴다.
  - ⬜ 대기 / 🔄 **진행** / 👀 **리뷰 대기** / ✅ **완료** / ⛔ **막힘**
  - `완료` = 완료 조건 충족 + 병합(코드 Task). 상위 Task(T-01~T-09) 표제의 상태는 하위 Task를 요약한 값이다.
- 마감이 밀리면 해당 Task의 상태 열과 §8 진행 기록에 적고, 뒤따르는 Task 마감을 같이 다시 잡는다. 11/30(판정 규칙 고정·기능 동결)과 12/14(발표)는 움직이지 않는다.

---

## 1. 아키텍처

### 1-1. 모듈 흐름

```text
                    configs/week1_env.yaml        configs/experiment_matrix.yaml
                            │                                │
 ┌──────────┐   ┌───────────▼──┐   ┌───────────────┐   ┌─────▼───────────────────────────┐
 │ data.py  │──▶│  inputs.py   │──▶│ extraction.py │──▶│ 층 벡터 캐시 [N, 23, 768]        │
 │ 문서 ID·  │   │ [SEP] 단일입력│   │ 패딩 제외 토큰 │   │ + doc_id·라벨 순서 + 생성 설정   │
 │ 튜닝분할· │   │ 앞+뒤 자르기  │   │ 평균, 23개     │   └─────┬───────────────────────────┘
 │ 소량표본  │   │ 길이·길이구간 │   └───────────────┘         │
 └──────────┘   └──────────────┘                            ▼
                                        ┌────────────── providers.py ──────────────┐
                                        │ CacheProvider(백본 고정)                  │
                                        │ BackboneProvider(파인튜닝·재현 트랙)       │
                                        │ FakeLayerVectorProvider(테스트)           │
                                        │ 생성 설정 ≠ 실행 설정 → CacheConfigMismatchError │
                                        └──────────────────┬───────────────────────┘
                                                           ▼
  ╔═════════════════ 실행 seam ═════════════════╗
  ║ runner.run_experiment(RunConfig, Provider)  ║   fusion.py: 층별 정규화 → 결합 방식 → 분류기
  ║        → ResultRecord                       ║   layers.py: 전역/지역 층, 층 범위 (단일 정의)
  ╚══════════════════════╤══════════════════════╝
         run_queue(본 실험, 1개씩) / 디버깅 / 재현 트랙
                         ▼
               결과 레코드 (JSON, 실행당 1개)        configs/decision_rules.yaml (11/30 locked)
                         │                                      │
  ╔══════════════════════▼═══════ 판정 seam ═════════════════════▼═╗
  ║ judgment.judge_hypotheses(records, DecisionRules) → JudgmentReport ║
  ║ (run_kind=debug, track=reproduction 레코드는 입력에서 제외)          ║
  ╚══════════════════════════════╤═════════════════════════════════╝
                                 ▼
                    reporting.py: 결과표·학습 곡선·층 가중치 그림·길이 구간 표
```

처리 순서는 고정이다(SPEC 모듈 6): (백본 안) 마지막 층에 최종 정규화 → 층마다 패딩 제외 토큰 평균 → 층 벡터 단위 층별 정규화 → 결합 → 분류기.

### 1-2. 모듈 ↔ 파일 ↔ 담당

| SPEC 모듈 | 파일 | 담당 | 현재 상태 |
| --- | --- | --- | --- |
| 1 데이터 | `src/layer_fusion/data.py` | 고용민 | 스켈레톤 |
| 2 입력 구성 | `src/layer_fusion/inputs.py` | 차가원 | 길이 구간·시야 비율 구현, 자르기 미구현 |
| 3 층 구조 정의 | `src/layer_fusion/layers.py` | 고용민 | 구현·테스트 통과 (US38) |
| 4 층 벡터 추출기 | `src/layer_fusion/extraction.py`, `scripts/extract_cache.py` | 차가원 | 스켈레톤 |
| 5 층 벡터 공급원 | `src/layer_fusion/providers.py` | 차가원 | 인터페이스·메타 비교 일부 |
| 6 결합 헤드 | `src/layer_fusion/fusion.py` | 고용민 | 스켈레톤 |
| 7 실행 모듈 | `src/layer_fusion/runner.py`, `scripts/run_experiment.py` | 차가원 | 스켈레톤 |
| 8 재현 트랙 실행기 | `runner.run_reproduction_track` | 차가원 | 스켈레톤 |
| 9 대기열 실행기 | `runner.run_sequential_queue`, `scripts/run_queue.py` | 차가원 | 스켈레톤 |
| 10 판정 모듈 | `src/layer_fusion/judgment.py` | 고용민 | 스켈레톤 |
| 11 보고 모듈 | `src/layer_fusion/reporting.py`, `scripts/judge_and_report.py` | 고용민 | 스켈레톤 |
| 계약 | `src/layer_fusion/contracts.py` | 고용민 (공통) | 초안 |

### 1-3. 두 seam 계약 (`contracts.py` 기준)

| seam | 입력 | 출력 | 테스트 방식 |
| --- | --- | --- | --- |
| ① 실행 | `RunConfig` + `LayerVectorProvider` | `ResultRecord` | `FakeLayerVectorProvider`, CPU |
| ② 판정 | `list[ResultRecord]` + `DecisionRules` | `JudgmentReport` | 손으로 만든 레코드, CPU, 순수 함수 |
| ③ 추출 확인 | 실제 ModernBERT-base, 짧은 입력 몇 개 | 층 벡터·캐시 | 고정한 transformers 버전, GPU 또는 CPU |

### 1-4. 의존 규칙

- `judgment.py`·`reporting.py`는 torch·모델을 import하지 않는다(결과 레코드만 읽음).
- `runner.py`는 공급원 인터페이스만 안다. 캐시 파일 형식·모델 로딩은 `providers.py`·`extraction.py` 안에 둔다.
- 층 번호·전역/지역·층 범위는 `layers.py`에서만 정의한다.
- 판정 기준값은 `configs/decision_rules.yaml`에서만 읽는다(코드에 숫자로 두지 않음).
- 계약을 바꿀 때는 `docs/SPEC.md` → `contracts.py` → 테스트 → 의존 모듈 순서로 바꾼다(AGENTS.md §4).

### 1-5. 설정 파일

| 파일 | 역할 | 확정 시점 |
| --- | --- | --- |
| `configs/week1_env.yaml` | transformers 버전, 어텐션·컴파일, Lfull 상한, 승인 여부, 학습 예산 | 11/16–11/22 (학습 예산은 늦어도 11/30) |
| `configs/experiment_matrix.yaml` | 실행 목록 (SPEC §실험 행렬) | 미정 항목은 11/30까지 |
| `configs/decision_rules.yaml` | 판정 규칙 문서 | **11/30 `status: locked`**, 이후 변경 금지 |

---

## 2. 담당 표

| 담당 | 책임 범위 |
| --- | --- |
| **고용민 (모델)** | 데이터 모듈, 결합 헤드(B0·F1~F4·B0+), 판정 모듈, 보고 모듈, 분석·해석 |
| **차가원 (인프라)** | 환경 확인·고정(11/16–11/22), 입력 구성, 추출·캐시·공급원, 실행 모듈·대기열·재현 트랙, 본 실험 순차 실행(ADR 0009) |
| **공통 기반 — 차가원** | 의존성 잠금·실행 환경, `scripts/verify.py`, `configs/week1_env.yaml`, `providers.py` 인터페이스 |
| **공통 기반 — 고용민** | `contracts.py`, `layers.py`, `judgment.py`, `configs/decision_rules.yaml` |
| **공통 (둘 다)** | `docs/SPEC.md`, `GLOSSARY.md`, `AGENTS.md`, `configs/experiment_matrix.yaml` |
| **병합** | PR 작성자가 상대 팀원 승인 후 병합. 공통 파일·계약 PR은 상대 팀원 + 다른 계열 모델 리뷰 |

- 공통 파일(AGENTS.md §4 목록)은 상대 리뷰 없이 병합하지 않는다.
- GPU를 쓰는 **본 실험 실행**은 차가원 한 사람이 대기열로 한 번에 하나씩 돌린다. **디버깅 실행**은 각자 동시에 돌려도 된다(ADR 0009).

---

## 3. 마일스톤 (12/14에서 역산)

| M | 기간 | 목표 | 게이트 |
| --- | --- | --- | --- |
| **M0** | 11/15–11/17 | git·환경 잠금, 계약 합의, 문서 정합성 | `python scripts/verify.py` 통과(skip 포함) |
| **M1** | 11/16–11/22 | 환경 확인(SPEC "1주차 확인"), CPU 모듈(데이터·입력·결합 헤드·공급원·판정 코어) | `week1_env.yaml` 환경 확인 항목 채움, Seam ①② 일부 un-skip |
| **M2** | 11/23–11/30 | 추출·캐시, 실행 seam, 대기열, 디버깅 E2E, 캐시 생성, 재현 트랙 실행, 판정 규칙 고정 | **11/30: `decision_rules.yaml` locked + 실행 경로 코드 기능 동결** (data·inputs·extraction·providers·fusion·runner) |
| **M3** | 12/1–12/9 | 본 실험 순차 실행: MNLI → SST-2 → IMDB | 모든 본 실험 결과 레코드 수집 |
| **M4** | 12/8–12/12 | 판정 실행, 보고 산출물, 보고서 4–6쪽, 발표 자료 | 보고서 초안 |
| **버퍼** | 12/12–12/13 | 실패 실행 재실행, 보고서 수정 | **12/14 최종 발표** |

- 기능 동결 뒤 실행 경로 코드는 버그 수정만 한다. 고친 경우 영향받는 본 실험 실행을 다시 돌린다.
- `judgment.py`·`reporting.py`는 11/30 뒤에도 고칠 수 있지만, 판정 **기준값**은 바꾸지 않는다(ADR 0007).

---

## 4. Task 목록

열 설명: **상태** = §0의 상태 값과 기호(에이전트도 갱신 가능) / **공통** = 공통 파일 수정 여부 / **위험** = 낮음·보통·높음 / **병렬** = 같은 마일스톤의 다른 Task와 동시 진행 가능 여부

### T-01 공통 기반 — ✅ **완료**

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-01a | ✅ **완료** (PR #2 병합, 차가원 Mac·GPU 서버·고용민 PC 확인) | Python 환경·의존성 잠금, README 설치 절차 | — | 11/17(화) | 차가원 | — | 새 환경에서 README대로 설치 후 verify 통과 (transformers 버전은 T-02a에서 확정 후 잠금 갱신) | verify, 두 사람 컴퓨터에서 실행 | 예 | 보통 | ○ |
| T-01b | ✅ **완료** (CODEOWNERS·main 보호·PR 흐름 확인 완료) | git 저장소 초기화, main 보호, CODEOWNERS(AGENTS.md §4 공통 파일) | — | 11/16(월) | 차가원 | — | 개인 브랜치(YongMin·Gawoun) → PR → main 흐름 작동, 공통 파일에 리뷰어 지정 | 테스트 PR 1건 | 예 | 높음 | ○ |
| T-01c | ✅ **완료** (PR #4 병합) | `contracts.py` 계약 검토·합의 (SPEC §데이터 형태와 필드 대조) | 52 | 11/17(화) | 고용민 (리뷰 차가원) | — | `RunConfig`·`ResultRecord`·`CacheMetadata`·`DecisionRules`가 SPEC 필드를 모두 담음 | verify, 상대 리뷰 | 예 | 높음 | ○ |
| T-01d | ✅ **완료** (PR #9 병합: ADR 0002 복구, ADR 0003·0005 문구 정리) | 문서 정합성: ADR 0002 파일 누락 복구, ADR 0003 "SST-2 판정" 문구 정리 | — | 11/17(화) | 공통 | — | ADR 0001~0009 파일이 모두 있고 H4 표현이 ADR 0007·GLOSSARY와 일치 | 문서 리뷰 | 예 | 낮음 | ○ |

### T-02 환경 확인 (11/16–11/22) — ✅ **완료**

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-02a | ✅ **완료** (PR #5: 5.23.0·sdpa·compile 끔 고정, CPU·GPU 최종 정규화 확인) | transformers 5.x 버전·어텐션 구현·컴파일 설정 고정, `verify_final_norm_contract` 구현 | 24, 50 | 11/20(금) | 차가원 | T-01a | `week1_env.yaml` `runtime_lock` 채움, 23번째 층 벡터 = 최종 정규화 출력의 토큰 평균 확인 | `test_23_layer_vectors_and_final_norm_contract` un-skip | 예 | 높음 | ○ |
| T-02b | ✅ **완료** (PR #3 병합) | IMDB 최대 토큰 길이 측정 → Lfull 상한·잘린 문서 비율 기록 | 15, 16 | 11/18(수) | 차가원 | T-01a | `imdb_length_check` 채움 (4,096 이하면 자르지 않음, 넘으면 4,096 + 비율) | 측정 스크립트 실행 로그 | 예 | 낮음 | ○ |
| T-02c | ✅ **완료** (승인 확정, `dataset_approval` 반영) | 교수님 MNLI·SST-2 사용 승인 확인 | 10, 11 | 11/20(금) | 공통 | — | `dataset_approval` 채움. 거절 시 ADR 0003 비상 계획 적용 | 수동 (회신 기록) | 예 | 높음 | ○ |

### T-03 데이터·입력 — 🔄 **진행**

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-03a | ⬜ 대기 | 데이터 모듈: MNLI·SST-2·IMDB 공통 로더, 문서 ID, MNLI matched/mismatched, 튜닝 분할 10%, 소량 표본(시드마다 새로·클래스당 1,000·ID 기록), 데이터셋 제외/AG News 전환이 설정만으로 되게 | 1, 3–11 | 11/19(목) | 고용민 | T-01c | 평가셋 = MNLI·SST-2 validation, IMDB test. 같은 시드 → 같은 표본, 다른 시드 → 다른 표본 | 단위 테스트(작은 가짜 데이터), `test_small_data_sampling_...` un-skip(T-06a와 함께) | 아니오 | 보통 | ○ |
| T-03b | ✅ **완료** (PR #8 병합: 앞+뒤 자르기·MNLI 결합·길이 구간) | 입력 구성: MNLI [SEP] 단일 입력, 앞+뒤 자르기(L128 32+94, L512 128+382, 특수 토큰 포함 128/512), Lfull 상한, 토큰 길이·길이 구간·시야 비율 | 2, 12–14, 17–19 | 11/21(토) | 차가원 | T-01a | 자르기 길이가 정확하고 모든 IMDB 문서에 길이·길이 구간이 붙음 | `test_head_tail_truncation_budgets` un-skip, 기존 길이 구간·시야 비율 테스트 | 아니오 | 보통 | ○ |

### T-04 추출·캐시·공급원 — 🔄 **진행**

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-04a | ✅ **완료** (PR #8 병합: 가짜 공급원·실행 전 설정 검사) | 공급원 인터페이스 확정, `FakeLayerVectorProvider.make_synthetic`, 생성 설정 불일치 오류 | 26 | 11/19(목) | 차가원 | T-01c | 가짜 공급원으로 실행 seam 테스트를 작성할 수 있음, 설정이 다르면 `CacheConfigMismatchError` | `test_mismatched_provider_metadata_raises_error` un-skip | 예 | 보통 | ○ |
| T-04b | ⬜ 대기 | 층 벡터 추출기·캐시 writer·`CacheProvider`: [N, 23, 768] + doc_id·라벨 순서 + 생성 설정(모델·버전·길이·자르기 방식) | 21–23, 25, 27 | 11/24(화) | 차가원 | T-02a, T-03b | 패딩 제외 평균, 짧은 문서는 L512 = Lfull, 순서 일치 | Seam ③ 나머지 5개 un-skip (`tests/test_extraction.py`) | 아니오 | 높음 | — |

### T-05 결합 헤드 — ⬜ 대기

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-05a | ⬜ 대기 | 층별 정규화(학습 파라미터 없는 LayerNorm) + B0·F1·F2(전체/전역만/지역만, γ 없음)·F3 + 학습 파라미터 수 + 층 가중치 노출 | 28–32, 35, 37 | 11/21(토) | 고용민 | T-01c | 전역만 8개·지역만 14개 가중치, 임베딩 벡터 제외 | `test_per_layer_norm_...`, `test_f2_has_no_global_scale_gamma`, `test_f2_global_and_local_subsets_...` un-skip | 아니오 | 보통 | ○ |
| T-05b | ⬜ 대기 | F4(입력별 게이팅) + B0+(2층 MLP, 은닉 MNLI 26) 파라미터 수 맞춤 | 33, 34, 36 | 11/26(목) | 고용민 | T-05a, §6의 F4 형태·허용 차이 결정 | F4 문서별 층 가중치 노출, B0+와 F4 파라미터 수 차이가 허용 범위 안 | `test_trainable_parameter_counts_and_b0_plus_matched_to_f4` un-skip | 아니오 | 보통 | — |

### T-06 실행 모듈 — ⬜ 대기

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-06a | ⬜ 대기 | 실행 seam(백본 고정): 학습률 3후보를 튜닝 분할로 선택, 동일 에폭·조기 종료, 평가셋 예측, MNLI matched/mismatched 오차율, 학습 시간·학습 곡선·환경 정보·층 가중치·소량 표본 ID 기록 | 39–43, 50, 52–54 | 11/25(수) | 차가원 (리뷰 고용민) | T-03a, T-04a, T-05a | 평가셋 라벨을 바꿔도 학습률 선택 불변, 레코드 필드 완비 | Seam ① 나머지(`tests/test_execution_seam.py`) un-skip | 아니오 | 높음 | — |
| T-06b | ⬜ 대기 | 결과 레코드 JSON 저장/로드, `experiment_matrix.yaml` → `RunConfig` 전개, `scripts/run_experiment.py` CLI | 48, 52 | 11/26(목) | 차가원 | T-06a | CLI 한 줄로 실행 → 레코드 파일 생성, 다시 읽어 판정 seam에 넣을 수 있음 | CLI 디버깅 실행 + 왕복 직렬화 테스트 | 아니오 | 보통 | — |
| T-06c | ⬜ 대기 | 대기열 실행기: 본 실험 동시 실행 차단(잠금), `run_kind`/`track` 기록, 우선순위 MNLI > SST-2 > IMDB, `scripts/run_queue.py` | 47–49, 51 | 11/27(금) | 차가원 | T-06b | 본 실험 2개를 동시에 시작하면 하나가 거부됨, 디버깅 실행은 대기열 밖에서 debug로 기록 | 단위 테스트 + 가짜 공급원 대기열 실행 | 아니오 | 높음 | — |
| T-06d | ⬜ 대기 | 파인튜닝 경로: `BackboneProvider` + 결합 헤드 동시 학습 (B0, 가장 좋은 결합 방식) | 44 | 11/28(토) | 차가원 · 고용민 | T-04b, T-05a | 백본 공급원으로 실행 seam이 같은 형식의 레코드를 냄 | 짧은 디버깅 실행 (GPU) | 아니오 | 높음 | ○ |
| T-06e | ⬜ 대기 | 재현 트랙 실행기: 논문 설정 그대로 전체 데이터 파인튜닝 ×3, `track=reproduction` | 45, 46 | 11/26(목) | 차가원 | T-02a, 논문 하이퍼파라미터 확인(§6) | 결과가 재현 트랙으로 표시되고 판정 입력에서 빠짐 | 짧은 디버깅 실행 + 판정 제외 테스트 | 아니오 | 보통 | ○ |

### T-07 판정 — ⬜ 대기

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-07a | ⬜ 대기 | **판정 규칙 문서 고정**: `decision_rules.yaml`의 TBD 전부 채우고 `status: locked` | 55 | **11/30(월)** 고정 기한 | 공통 (작성 고용민) | — | §6의 "11/30" 항목이 모두 값을 가짐, 본 실험 결과 레코드가 하나도 없는 상태에서 고정 | 두 사람 승인, 로더가 locked 아니면 본 실험 판정 거부 | 예 | 높음 | ○ |
| T-07b | ⬜ 대기 | 판정 코어: 규칙 로더, 시드 평균·표준편차, 문서 ID로 짝지은 McNemar, 지지/효과 없음, debug·재현 트랙 제외, 탐색적 가설 무판정 | 56–59 | 11/24(화) | 고용민 | T-01c | 규칙값은 인자로 받아 합성 레코드로 검증 | Seam ② 6개 un-skip (`tests/test_judgment_seam.py`) | 예 | 높음 | ○ |
| T-07c | ⬜ 대기 | 가설별 판정·분석: H1·H3·H2·H4′·H5a·H5b(섞기 효과 F1−B0 / 가중치 학습 효과 F2−F1), 길이 구간별 오차율·최소 표본, ≤512 구간 L512 = Lfull 확인, 무게중심·전역 층 가중치 비율·시드 일관성, H4 SST-2 McNemar, 전역/지역 ablation 표, F4 길이–가중치 순위 상관(전체·라벨별) | 60–72 | 11/28(토) | 고용민 | T-07b (기준값은 T-07a) | 손 계산 예(22층에 가중치 전부 → 무게중심 22, 비율 1)와 일치 | Seam ② 나머지 3개 un-skip + 추가 단위 테스트 | 예 | 보통 | — |

### T-08 실행 — ⬜ 대기

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-08a | ⬜ 대기 | 디버깅 E2E 스모크: 소규모 데이터로 추출 → 실행 → 판정 → 보고 (`run_kind=debug`) | — | 11/28(토) | 차가원 | T-04b, T-06c, T-07b | 파이프라인 끝까지 통과, 디버깅 레코드가 결과표에 안 나옴 | 실제 실행 로그 | 아니오 | 보통 | — |
| T-08b | ⬜ 대기 | 층 벡터 캐시 생성: MNLI, SST-2, IMDB × L128/L512/Lfull | 21 | 11/27(금) | 차가원 | T-04b, T-02b, 캐시 정밀도 결정 | 캐시마다 생성 설정 기록, 형태·순서 검사 통과 | 캐시 검사 스크립트 | 아니오 | 보통 | — |
| T-08c | ⬜ 대기 | 재현 트랙 실행: MNLI·SST-2 × 3회 | 45 | 11/30(월) | 차가원 | T-06e | 오차율이 공개 수치(10.9% / 4.0%)와 나란히 기록됨, 본 실험 시작 전 완료 | 재현 트랙 레코드 | 아니오 | 보통 | — |
| T-08d | ⬜ 대기 | 본 실험: MNLI 백본 고정·전체, B0·F1·F2·F3·F4·B0+ | 60, 63 | 12/2(수) | 차가원 | T-07a, T-08a, T-08b | 시드 수(§6)만큼 레코드 수집 | 대기열 로그, 레코드 수 검사 | 아니오 | 높음 | — |
| T-08e | ⬜ 대기 | 본 실험: MNLI 백본 고정·소량 | 62 | 12/3(목) | 차가원 | T-08d와 독립(대기열 순서만), 결합 방식 목록 결정 | 시드마다 표본 ID 기록 | 대기열 로그 | 아니오 | 보통 | — |
| T-08f | ⬜ 대기 | 본 실험: MNLI 파인튜닝, B0 vs 가장 좋은 결합 방식 | 61 | 12/6(일) | 차가원 | T-08d 결과, T-06d, 선정 기준(T-07a) | 선정 기준대로 고른 결합 방식으로 실행 | 대기열 로그 | 아니오 | 높음 | — |
| T-08g | ⬜ 대기 | 본 실험: SST-2 백본 고정 (승인 시에만) | 70 | 12/7(월) | 차가원 | T-02c, 결합 방식 목록 결정(F2 포함) | F2 층 가중치 기록 | 대기열 로그 | 아니오 | 낮음 | — |
| T-08h | ⬜ 대기 | 본 실험: IMDB L128/L512/Lfull × B0·F1·F2, F2 전역만/지역만, Lfull F4 | 64–66, 71, 72 | 12/9(수) | 차가원 | T-08b, T-05b | 레코드에 문서별 길이·길이 구간 포함 | 대기열 로그 | 아니오 | 보통 | — |

### T-09 보고 — ⬜ 대기

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-09a | ⬜ 대기 | 보고 모듈: 결과표(오차율 평균 ± 표준편차 / 학습 파라미터 수 / 학습 시간), 학습 곡선, 층 가중치 그림, 길이 구간 표(문서 수 포함), 시야 비율 표 | 20, 73, 74 | 11/27(금) | 고용민 | T-01c | 합성 레코드로 표·그림 생성, debug·재현 트랙 제외 | 단위 테스트 + 산출물 눈 확인 | 아니오 | 낮음 | ○ |
| T-09b | ⬜ 대기 | 판정 실행 + 보고서 4–6쪽 (표·그래프·해석 짝, 재현 트랙 별도 절, 한계 절) | 46, 73–75 | 12/11(금) | 공통 | T-08d~h, T-07c, T-09a | 확증 가설은 지지/효과 없음, 탐색적 가설은 경향만 | 상대 리뷰 + 다른 계열 모델 검토 1회 | 아니오 | 보통 | — |
| T-09c | ⬜ 대기 | 발표 자료 | — | 12/12(토) | 공통 | T-09b | 12/14 발표 가능 | 리허설 | 아니오 | 낮음 | — |

### 병렬로 할 수 있는 묶음

- **11/15–11/17**: T-01a ∥ T-01b ∥ T-01c ∥ T-01d
- **11/16–11/22**: 고용민 T-03a → T-05a → T-05b, T-07b / 차가원 T-02a ∥ T-02b ∥ T-03b ∥ T-04a / 공통 T-02c
- **11/23–11/30**: 고용민 T-07c ∥ T-09a ∥ T-07a 초안 / 차가원 T-04b → T-06a → T-06b → T-06c → T-08a, T-06d ∥ T-06e → T-08b·T-08c (GPU는 순차)

---

## 5. 의존 그래프와 임계 경로

```text
T-01a ─┬─▶ T-02a ─┬─▶ T-04b ─┬─▶ T-08b ─┐
       │          │          ├─▶ T-06d ─┼──────────────────▶ T-08f ─┐
       │          └─▶ T-06e ─┼──────────┼─▶ T-08c             ▲      │
       ├─▶ T-02b ─────────────┼─────────▶ T-08b               │      │
       └─▶ T-03b ────────────▶┘                               │      │
T-01c ─┬─▶ T-03a ─┐                                           │      │
       ├─▶ T-04a ─┼─▶ T-06a ─▶ T-06b ─▶ T-06c ─▶ T-08a ─▶ T-08d ─┤      ├─▶ T-09b ─▶ T-09c
       ├─▶ T-05a ─┘                                ▲          ├─▶ T-08e
       │     └─▶ T-05b ────────────────────────────┼──────────┴─▶ T-08h
       ├─▶ T-07b ─▶ T-07c ─────────────────────────┼─────────────────────┘
       └─▶ T-09a ──────────────────────────────────┼─────────────────────┘
                              T-07a (11/30 locked) ┘
```

**임계 경로**: T-01a → T-02a → T-04b → T-08b → (T-07a 11/30) → T-08d → T-08f → T-09b → 12/14

- T-08f(파인튜닝)는 T-08d 결과로 "가장 좋은 결합 방식"을 고른 뒤에야 시작할 수 있다. 그래서 MNLI 백본 고정 실행을 본 실험 첫 순서(12/1부터)로 두고, 선정 기준은 11/30에 미리 고정한다.
- GPU를 쓰는 T-08b·T-08c·T-08d~h는 ADR 0009에 따라 차가원이 하나씩 순차로 돌린다.

---

## 6. 결정 필요 사항 → 막는 Task

`docs/SPEC.md` §결정 필요 사항의 각 항목이 어느 Task를 막는지만 적는다. 값은 정하지 않는다. SPEC에 따라 **모두 본 실험 실행 전(늦어도 11/30)에** 고정한다.

| 항목 | 정할 날짜 | 기록 위치 | 막는 Task |
| --- | --- | --- | --- |
| transformers 버전, 층별 출력 마지막 값 정의 | 11/16–11/22 | `week1_env.yaml` | T-02a → T-04b |
| 어텐션 구현, 컴파일 설정 | 11/16–11/22 | `week1_env.yaml` | T-02a → T-04b, T-06a |
| Lfull 상한 | 11/16–11/22 | `week1_env.yaml` | T-02b → T-08b(IMDB Lfull) |
| MNLI·SST-2 승인 | 11/16–11/22 | `week1_env.yaml` | T-08d~g |
| 최소 표본 기준, 미달 구간 표시 | 11/23–11/30 (11/30 고정) | `decision_rules.yaml` | T-07c 결과, T-09b |
| 시드 일관성 기준값 | 11/23–11/30 (11/30 고정) | `decision_rules.yaml` | T-07c 결과 |
| McNemar 유의수준, 다중 비교 보정 | 11/23–11/30 (11/30 고정) | `decision_rules.yaml` | T-07b·T-07c 결과 |
| 여러 시드의 McNemar 결합 방법 | 11/23–11/30 (11/30 고정) | `decision_rules.yaml` | T-07b |
| 무게중심·전역 층 가중치 비율에서 임베딩 벡터 처리 | 11/23–11/30 (11/30 고정) | `decision_rules.yaml` | T-07c |
| 핵심 비교 목록(5시드 대상), 파인튜닝 시드 수 | 11/23–11/30 (11/30 고정) | `decision_rules.yaml`, `experiment_matrix.yaml` | T-08d, T-08f |
| H1 판정 결합 방식 | 11/23–11/30 (11/30 고정) | `decision_rules.yaml` | T-07c |
| H5a 판정 결합 방식 | 11/23–11/30 (11/30 고정) | `decision_rules.yaml` | T-07c |
| "가장 좋은 결합 방식" 선정 기준 | 11/23–11/30 (11/30 고정) | `decision_rules.yaml` | T-08f |
| F4 구체 형태(게이트 구조·입력 벡터) | 11/16–11/30 (구현 중) | `docs/SPEC.md` | T-05b |
| F2·F4 층 가중치 정규화 방식(softmax 여부) | 11/16–11/30 (구현 중) | `docs/SPEC.md` | T-05a, T-05b |
| 토큰 평균에 특수 토큰 포함 여부 | 11/16–11/30 (구현 중) | `week1_env.yaml` | T-04b |
| 학습률 후보 3개, 에폭 수, 조기 종료, 배치 크기, 파인튜닝 학습률 범위 | 11/16–11/30 (구현 중) | `week1_env.yaml` | T-06a(값 없이 구현 가능) → T-08d |
| 튜닝 분할 방식(하나로 고정 여부) | 11/16–11/30 (구현 중) | `docs/SPEC.md` | T-03a |
| 소량 표본 범위(튜닝 분할 포함 여부), 소량 조건의 튜닝 분할 | 11/16–11/30 (구현 중) | `docs/SPEC.md` | T-03a |
| MNLI 소량·SST-2에서 돌릴 결합 방식 목록 | 11/16–11/30 (구현 중) | `experiment_matrix.yaml` | T-08e, T-08g |
| 학습 시간 측정 범위(추출 시간 포함 여부) | 11/16–11/30 (구현 중) | `week1_env.yaml` | T-06a |
| 파인튜닝 조건 파라미터 수 범위(백본 포함 여부) | 11/16–11/30 (구현 중) | `docs/SPEC.md` | T-06d, T-09a |
| B0+와 F4 파라미터 수 허용 차이 | 11/16–11/30 (구현 중) | `docs/SPEC.md` | T-05b |
| AG News 전환 시 B0+ 은닉 크기 | MNLI 거절 시 즉시 | `docs/SPEC.md` | T-05b (MNLI 거절 시만) |
| 캐시 저장 정밀도(float32 약 28GB / float16 약 14GB, MNLI 학습셋) | 11/16–11/30 (구현 중) | `week1_env.yaml` | T-04b, T-08b |
| **(추가)** 재현 트랙의 논문 하이퍼파라미터 — SPEC에 값이 없음 | 11/16–11/22 | `experiment_matrix.yaml` `reproduction_track` | T-06e, T-08c |

---

## 7. 위험과 문서 충돌

| 항목 | 영향 | 대응 |
| --- | --- | --- |
| ADR 0002가 `docs/SPEC.md`·`AGENTS.md`에서 인용되지만 `docs/adr/`에 파일이 없음 | 근거 추적 불가 | T-01d에서 복구 (내용은 팀이 작성, 에이전트가 지어내지 않음) |
| `decision_rules.yaml`의 `target_fusion_rule`(H1·H5a), `core_comparisons_for_5_seeds`, `finetune_num_seeds`, `best_fusion_selection_rule`은 T-01c에서 `DecisionRules`에 `None` 기본값 필드로 추가됨. 단 형태가 SPEC에서 미정이라 4개는 `Any | None`(임시)이고, `hypotheses`의 `type`/`priority`는 계약에 필드가 없음 | T-07b가 규칙값을 타입 없이 읽거나 가설 유형·우선순위를 코드에 하드코딩할 수 있음 | T-07a에서 값·형태를 고정할 때 타입을 좁히고, T-07b 규칙 로더 작성 전에 `hypotheses` type/priority를 계약에 넣을지 로더가 YAML을 직접 읽을지 팀이 결정 (계약 변경이므로 SPEC → contracts → 테스트 순서) |
| ADR 0003 "SST-2는 층 가중치로 판정" vs ADR 0007·GLOSSARY "H4는 탐색적" | 판정표 해석 혼동 | SPEC은 ADR 0007을 따름. T-01d에서 ADR 0003 문구 정리 |
| 저장소가 아직 git이 아님 | 브랜치·리뷰 규칙(AGENTS.md §4) 적용 불가 | T-01b를 첫 작업으로 (11/15–11/17) |
| 본 실험 기간(12/1–12/9) GPU 시간 추정치가 문서에 없음 (MNLI 파인튜닝 × 시드, 재현 6회, IMDB Lfull 추출) | 12/9까지 본 실험이 끝나지 않을 수 있음 | T-08a 디버깅 실행에서 실행당 시간을 재고 일정 재점검. 밀리면 MNLI > SST-2 > IMDB(ADR 0003) |
| T-08f가 T-08d 결과에 순차 의존 | 임계 경로가 길어짐 | MNLI 백본 고정을 본 실험 맨 앞(12/1~)에, 선정 기준을 11/30에 고정 |
| 캐시 용량 | 디스크 부족 시 T-08b 중단 | 정밀도 결정을 11/23 전후에, 캐시는 커밋 금지(AGENTS.md §7) |
| 판정 규칙 고정 지연 | 본 실험 시작 불가 | T-07a 기한 11/30, 본 실험 판정은 locked 규칙만 받음 |
| SST-2 미승인 / MNLI 거절 | 실험 행렬 변경 | SST-2는 설정에서 제외만, MNLI 거절 시 AG News(ADR 0003) |

---

## 8. 진행 기록

Task별 상태는 §4 표의 "상태" 열이 기준이고, 이 절에는 세션 로그와 마지막 verify만 남긴다. 옛 `docs/progress.md`의 로그를 옮겼다(T-01c 항목에는 남은 결정 한 문장을 덧붙였다).

**마지막 verify**: 2026-11-18 Gawoun(PR #8), Python 3.11 및 로컬 모델 캐시 사용 (40 passed, 22 skipped). torch·transformers 없는 환경에서는 실제 모델 테스트 3~4개가 skip된다.

- 2026-11-15 · 초기 스캐폴딩 생성: `SPEC.md`, `GLOSSARY.md`, `docs/adr/0001`–`0009`를 바탕으로 패키지 구조, 설정 템플릿, 테스트 스켈레톤, 하네스 파일 생성.
- 2026-11-15 · T-01a(차가원): Python 3.11 venv, `requirements.lock`(uv, universal) 생성, 새 환경에서 잠금 파일 설치 후 `python scripts/verify.py` ALL PASSED(5 passed, 26 skipped). README 설치 절차 갱신.
- 2026-11-15 · T-02b(차가원): `scripts/measure_imdb_lengths.py`로 IMDB train/test 50,000건 측정. 최대 3,270(train)/2,971(test) 토큰 → `week1_env.yaml` `imdb_length_check` 채움(상한 4,096, 잘림 0건). 결과 `docs/measurements/imdb_length_check.json`. verify ALL PASSED.
- 2026-11-18 · T-01c(고용민): SPEC §데이터 형태와 `contracts.py` 대조. `RunConfig`·`ResultRecord`·`CacheMetadata`는 SPEC 필드를 모두 담음. `DecisionRules`에 `decision_rules.yaml`의 규칙 5개(`h1_target_fusion_rule`, `h5a_target_fusion_rule`, `core_comparisons_for_5_seeds`, `finetune_num_seeds`, `best_fusion_selection_rule`)가 타입 필드 없이 `extra`로만 들어가던 것을 `None` 기본값 필드로 추가(값은 TBD 그대로). `tests/test_contracts.py` 7개 추가. verify ALL PASSED(12 passed, 26 skipped). 팀 확인 필요: `lr_tuning_scores` 키가 JSON에서 문자열이 됨(T-06b), `early_stopping_patience`가 int로 가정됨, `validate_metadata_match`가 SPEC 4항목보다 엄격(차가원 확인). 추가로 남은 범위: 가설별 `type`/`priority`(확증/탐색 구분)는 `DecisionRules` 타입 필드가 없음 — 계약에 넣을지 판정 로더가 YAML을 직접 읽을지 결정 필요.
- 2026-11-18 · T-01c 교차검증 반영(고용민): 다른 에이전트의 코드리뷰를 실제 파일과 대조해 타당한 항목만 반영. (1) `h1_target_fusion_rule`·`h5a_target_fusion_rule`·`core_comparisons_for_5_seeds`·`best_fusion_selection_rule` 타입을 `str`/`list[str]`에서 `Any | None`으로 완화(SPEC §11/30이 대표 방식 하나인지 여러 방식인지 열어 둠, 값은 `None` 그대로). (2) `tests/test_contracts.py` `SPEC_FIELDS`에 SPEC §11/30 고정 항목 7개와 `token_length`·`length_bucket`·`truncated_doc_ratio`(US 16·17) 추가. 반영하지 않음: `genre_split` 검사(SPEC에 해당 필드 없음), `hypotheses` type/priority(위 T-01c 항목의 미결정 사항 그대로, 팀 결정 필요). §7의 `DecisionRules` 5필드 행이 옛 상태("`contracts.py`에 필드 없음")로 남아 있어, 사용자 요청에 따라 현재 상태(필드 추가됨, 타입 임시, `hypotheses` type/priority 미결정)로 고침.
- 2026-11-18 · T-02c(고용민): 교수님의 MNLI·SST-2 사용 승인이 확정되었다는 사용자 확인에 따라 `configs/week1_env.yaml` `dataset_approval`의 `mnli_approved`·`sst2_approved`를 `true`로 채우고 상태를 완료로 바꿈. ADR 0003 비상 계획(AG News 전환)·SST-2 제외는 적용하지 않음. 회신 원문은 이 저장소에 기록이 없으므로 필요하면 팀이 출처(메일 등)를 따로 남길 것. `configs/`는 공통 파일이라 병합 전 차가원 리뷰 필요.
- 2026-11-18 · 진행 관리 통합(고용민): `docs/progress.md`를 폐지하고 Task별 상태를 이 문서 §4의 "상태" 열과 이 절로 합침. AGENTS.md §3·CLAUDE.md·README의 참조를 고침. 팀 확인 필요: 아직 병합되지 않은 브랜치(`feat/T-02a-runtime-lock` 등)가 옛 `progress.md`를 수정해 두었으면 병합 때 충돌하며, 그때는 삭제를 받아들이고 해당 상태·로그를 이 문서에 옮긴다.
- 2026-11-18 · T-02a(차가원): `check_final_norm_contract`/`verify_final_norm_contract`·`masked_token_mean` 구현, `scripts/check_runtime_lock.py`로 CPU 확인 → `docs/measurements/runtime_check_cpu.json`. hidden_states 23개, 마지막 값 = `final_norm(마지막 층 원출력)` = `last_hidden_state`(층 노름 21·22층 ≈ 3,760 vs 23번째 ≈ 27). `transformers==5.23.0` 고정(lock 변화 없음). Seam ③ 첫 테스트 un-skip(가중치가 로컬 캐시에 없으면 skip). attn·compile은 GPU 서버 측정 대기. 참고: ADR 0005가 근거로 든 `tie_last_hidden_states` 속성은 5.23.0 config에 없음 — 동작은 동일, 팀 보고.
- 2026-11-18 · T-02a GPU 측정(차가원): 공용 서버 metis(RTX 5090, torch 2.15.0+cu132)의 개인 개발 컨테이너에서 `scripts/check_runtime_lock.py --device cuda --attn sdpa flash_attention_2 --compile off on --benchmark-steps 20` 실행 → `docs/measurements/runtime_check_cuda.json`. sdpa는 compile 끔/켬 모두 최종 정규화 계약 통과(23개, 마지막 = `final_norm(원출력)`), forward(32×512, fp32) 평균 0.1071초 / 0.0971초(compile이 약 9% 빠름). flash_attention_2는 패키지 미설치로 측정 불가. Seam ③ 최종 정규화 테스트 서버에서도 PASSED. compile 켬은 처음에 `stdlib.h` 없음으로 실패 → devcontainer에 `libc6-dev` 추가(`209d435`). `attn_implementation`·`torch_compile` 값은 팀 결정 대기(`week1_env.yaml`은 아직 null).
- 2026-11-18 · T-01a 완료(차가원): 고용민 PC에서 README 절차대로 설치 후 verify 통과 확인(팀원 보고). 차가원 Mac(11/15·11/18), GPU 서버 컨테이너(11/18)와 함께 완료 조건 충족. Windows용 `py -3.11` 안내는 PR #5에 포함.
- 2026-11-18 · T-02a 완료(차가원): 고용민 PR #5 승인으로 `attn_implementation: "sdpa"`, `torch_compile: false` 결정 → `configs/week1_env.yaml` `runtime_lock`의 버전·마지막 층 정의·어텐션·컴파일 4개 값 채움(`cache_dtype`은 T-04b 전 별도 결정). 근거 정정: compile 측정은 forward만·512 고정 길이·1회라 파인튜닝 이득은 미측정 → T-08a에서 파인튜닝 실행 시간 실측 후 필요하면 11/30 전에 재검토. ADR 0005의 `tie_last_hidden_states` 문구는 T-01d에서 정리하기로 함.
- 2026-11-18 · 정리(차가원): T-01b 완료 조건 문구를 개인 브랜치 규칙(AGENTS.md §4)에 맞게 "기능 브랜치·PR 흐름" → "개인 브랜치(YongMin·Gawoun) → PR → main 흐름"으로 수정(차가원 승인, 고용민 요청). 병합이 끝난 원격 브랜치 `feat/T-01a-env-lock`·`feat/T-02b-imdb-length`·`feat/T-02a-runtime-lock`·`feat/T-01-devcontainer` 삭제(내용이 main에 있는 것 확인 후).
- 2026-11-18 · T-03b(차가원): `inputs.build_tokenized_inputs` 구현 — MNLI·SST-2는 `[CLS] a [SEP] (b [SEP])` 그대로, L128/L512는 내용 토큰 앞 32+뒤 94 / 앞 128+뒤 382로 특수 토큰 포함 정확히 128/512, Lfull은 상한까지 자르지 않음. 문서별 `token_lengths`(자르기 전)·`length_buckets`·`truncated`·`truncated_doc_ratio` 반환. `truncation_rule_name` 추가(T-04b `CacheMetadata.truncation_rule`용). transformers 5.x에 `build_inputs_with_special_tokens`가 없어 특수 토큰을 `cls/sep_token_id`로 직접 붙임. `test_head_tail_truncation_budgets`(가짜 토크나이저)·`test_imdb_head_tail_truncation_lengths`(실제 ModernBERT 토크나이저, 캐시 없으면 skip) un-skip. 실제 IMDB test 200건: L128 잘림 90.5%, L512 14%, Lfull 0%, 앞·뒤 디코딩 확인. **결정 필요**: Lfull 상한 초과 문서의 자르기 방식은 ADR 0006에 없어 `ValueError`로 멈추게 둠(IMDB 최대 3,270 < 4,096이라 현재 영향 없음). 참고: IMDB 원문에 `<br />` 태그가 그대로 있음 — 전처리 여부는 T-03a(데이터 모듈)에서 팀 확인 필요(T-02b 길이 측정도 원문 기준).
- 2026-11-18 · T-04a(차가원): `providers.check_provider_for_run` 추가 — `input_setting`이 다르면 오류, **본 실험(`run_kind=main`)은 `expected_cache_meta` 필수**(디버깅만 생략 허용), 기대 메타가 있으면 `CacheMetadata` 전체 비교(ADR 0005). 기존에는 기대 메타가 없으면 검사를 건너뛰어 다른 설정 캐시가 조용히 통과할 수 있었음(US 26). `FakeLayerVectorProvider.make_synthetic` 구현(train/eval, [N,23,768] float32, seed 결정적, 클래스 균형·클래스별 평균 차이로 학습 신호, MNLI형 eval은 matched/mismatched 반씩, `doc_records` 지정 가능, 기본 메타 = `week1_env.yaml` 고정값). `run_experiment`는 학습 전에 `validate_for_run`만 수행(나머지는 T-06a). `test_mismatched_provider_metadata_raises_error` un-skip, `tests/test_providers.py` 11개 추가. verify ALL PASSED(39 passed, 22 skipped). **팀 확인 필요**: `CacheMetadata`에 데이터셋 이름이 없어 MNLI 캐시를 SST-2 실행에 써도 메타 검사로 못 잡음 — SPEC 캐시 생성 설정에도 없어 계약 변경 여부 논의 필요.
- 2026-11-18 · PR #8 리뷰 반영(차가원): 고용민 교차검증 반영 — `check_provider_for_run`의 `input_setting` 비교에서 `is not None` 조건 제거(MNLI·SST-2 실행에 IMDB 캐시가 들어오는 경우도 차단, 테스트 추가), 실제 ModernBERT 토크나이저로 MNLI 직접 결합 `[CLS] a [SEP] b [SEP]` = `tokenizer(a, b)` 확인 테스트 추가. 리뷰에서 팀 합의: Lfull 상한 초과 시 오류 유지, IMDB `<br />` 원문 유지, `CacheMetadata`에 데이터셋 필드 추가(계약 변경, 고용민이 SPEC부터 반영), 본 실험 `expected_cache_meta` 필수(T-06b에서 자동 생성), **캐시 float16 저장·로드 시 float32**(디버깅 추출에서 최대 절댓값 1회 확인 후 확정), **토큰 평균에 특수 토큰 포함(true)** → `week1_env.yaml` 기록은 T-04b에서.
- 2026-11-18 · T-03b·T-04a 완료(차가원): PR #8 고용민 승인 후 병합(`aee4f10`). /code-review(표준·명세) 반영분 포함. T-04b로 넘긴 항목: `TokenizedInputs` 데이터클래스, `truncation_rule` 문자열 형식 확인, 문서별 시야 비율, 데이터셋 필드·IMDB `input_setting` 검사(계약 변경), `expected_cache_meta` 필수 규칙 SPEC 반영, `cache_dtype`·`include_special_tokens_in_pool` 기록.
- 2026-11-18 · T-01d 완료 표시(차가원, 담당 공통): PR #9(고용민) 병합 확인 후 상태 ✅, T-01 묶음 제목 ✅로 갱신.
