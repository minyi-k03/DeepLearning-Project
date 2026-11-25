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
  - **상태 변경은 그 Task를 끝내는 PR 안에 함께 넣는다.** PR을 올릴 때 상태를 `✅ **완료** (PR #N 병합 시)`처럼 미리 적고 상위 Task 표제도 같이 갱신해, 병합과 동시에 main에 반영되게 한다(2026-11-18 차가원·고용민 합의). 리뷰 중 막히면 👀/⛔로 되돌린다.
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

### T-03 데이터·입력 — ✅ **완료**

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-03a | ✅ **완료** (PR #11 병합 시: 로더·튜닝 분할·소량 표본, 교차검증 반영) | 데이터 모듈: MNLI·SST-2·IMDB 공통 로더, 문서 ID, MNLI matched/mismatched, 튜닝 분할 10%, 소량 표본(시드마다 새로·클래스당 1,000·ID 기록), 데이터셋 제외/AG News 전환이 설정만으로 되게 | 1, 3–11 | 11/19(목) | 고용민 | T-01c | 평가셋 = MNLI·SST-2 validation, IMDB test. 같은 시드 → 같은 표본, 다른 시드 → 다른 표본 | 단위 테스트(작은 가짜 데이터), `test_small_data_sampling_...` un-skip(T-06a와 함께) | 아니오 | 보통 | ○ |
| T-03b | ✅ **완료** (PR #8 병합: 앞+뒤 자르기·MNLI 결합·길이 구간) | 입력 구성: MNLI [SEP] 단일 입력, 앞+뒤 자르기(L128 32+94, L512 128+382, 특수 토큰 포함 128/512), Lfull 상한, 토큰 길이·길이 구간·시야 비율 | 2, 12–14, 17–19 | 11/21(토) | 차가원 | T-01a | 자르기 길이가 정확하고 모든 IMDB 문서에 길이·길이 구간이 붙음 | `test_head_tail_truncation_budgets` un-skip, 기존 길이 구간·시야 비율 테스트 | 아니오 | 보통 | ○ |

### T-04 추출·캐시·공급원 — ✅ **완료** (PR #13 병합 시)

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-04a | ✅ **완료** (PR #8 병합: 가짜 공급원·실행 전 설정 검사) | 공급원 인터페이스 확정, `FakeLayerVectorProvider.make_synthetic`, 생성 설정 불일치 오류 | 26 | 11/19(목) | 차가원 | T-01c | 가짜 공급원으로 실행 seam 테스트를 작성할 수 있음, 설정이 다르면 `CacheConfigMismatchError` | `test_mismatched_provider_metadata_raises_error` un-skip | 예 | 보통 | ○ |
| T-04b | ✅ **완료** (PR #13 병합 시: PR #12 추출기·캐시 writer·`CacheProvider` + PR #13 CLI·float16 실측·`week1_env.yaml`·SPEC 결정 기록. 후속: `CacheMetadata` 데이터셋 필드는 계약 PR 후) | 층 벡터 추출기·캐시 writer·`CacheProvider`: [N, 23, 768] + doc_id·라벨 순서 + 생성 설정(모델·버전·길이·자르기 방식) | 21–23, 25, 27 | 11/24(화) | 차가원 | T-02a, T-03b | 패딩 제외 평균, 짧은 문서는 L512 = Lfull, 순서 일치 | Seam ③ 나머지 5개 un-skip (`tests/test_extraction.py`) | 아니오 | 높음 | — |

### T-05 결합 헤드 — ✅ **완료**

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-05a | ✅ **완료** (PR #14 병합 시: B0·F1·F2·F3·층별 정규화, F2 softmax SPEC 기록) | 층별 정규화(학습 파라미터 없는 LayerNorm) + B0·F1·F2(전체/전역만/지역만, γ 없음)·F3 + 학습 파라미터 수 + 층 가중치 노출 | 28–32, 35, 37 | 11/21(토) | 고용민 | T-01c | 전역만 8개·지역만 14개 가중치, 임베딩 벡터 제외 | `test_per_layer_norm_...`, `test_f2_has_no_global_scale_gamma`, `test_f2_global_and_local_subsets_...` un-skip | 아니오 | 보통 | ○ |
| T-05b | ✅ **완료** (PR #18 병합 시: F4 게이트·B0+·파라미터 수 허용 차이 1%·역할 기준 가중치 감쇠 SPEC 기록. 후속: 러너의 F4 문서별 가중치 연동은 차가원 T-06d) | F4(입력별 게이팅) + B0+(2층 MLP, 은닉 MNLI 26) 파라미터 수 맞춤 | 33, 34, 36 | 11/26(목) | 고용민 | T-05a, §6의 F4 형태·허용 차이 결정 | F4 문서별 층 가중치 노출, B0+와 F4 파라미터 수 차이가 허용 범위 안 | `test_trainable_parameter_counts_and_b0_plus_matched_to_f4` un-skip | 아니오 | 보통 | — |

### T-06 실행 모듈 — 🔄 **진행**

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-06a | ✅ **완료** (PR #15 병합 시: 실행 seam·학습 루프, 학습률 선택·조기 종료·소량 표본·학습 시간 SPEC 결정 기록) | 실행 seam(백본 고정): 학습률 3후보를 튜닝 분할로 선택, 동일 에폭·조기 종료, 평가셋 예측, MNLI matched/mismatched 오차율, 학습 시간·학습 곡선·환경 정보·층 가중치·소량 표본 ID 기록 | 39–43, 50, 52–54 | 11/25(수) | 차가원 (리뷰 고용민) | T-03a, T-04a, T-05a | 평가셋 라벨을 바꿔도 학습률 선택 불변, 레코드 필드 완비 | Seam ① 나머지(`tests/test_execution_seam.py`) un-skip | 아니오 | 높음 | — |
| T-06b | ✅ **완료** (PR #19 병합 시: 결과 레코드 JSON·행렬 전개·`run_experiment.py` debug CLI) | 결과 레코드 JSON 저장/로드, `experiment_matrix.yaml` → `RunConfig` 전개, `scripts/run_experiment.py` CLI | 48, 52 | 11/26(목) | 차가원 | T-06a | CLI 한 줄로 실행 → 레코드 파일 생성, 다시 읽어 판정 seam에 넣을 수 있음 | CLI 디버깅 실행 + 왕복 직렬화 테스트 | 아니오 | 보통 | — |
| T-06c | ✅ **완료** (PR #21 병합: 대기열 실행기·본 실험 GPU 잠금·학습률 재사용·이어하기) | 대기열 실행기: 본 실험 동시 실행 차단(잠금), `run_kind`/`track` 기록, 우선순위 MNLI > SST-2 > IMDB, `scripts/run_queue.py` | 47–49, 51 | 11/27(금) | 차가원 | T-06b | 본 실험 2개를 동시에 시작하면 하나가 거부됨, 디버깅 실행은 대기열 밖에서 debug로 기록 | 단위 테스트 + 가짜 공급원 대기열 실행 | 아니오 | 높음 | — |
| T-06d | ✅ **완료** (PR #22 병합 시: 파인튜닝 경로·GPU 디버깅 실행·RNG 재현성. 후속: SPEC 제안 2건 고용민 리뷰, GPU 비결정성 허용 여부 팀 결정) | 파인튜닝 경로: `BackboneProvider` + 결합 헤드 동시 학습 (B0, 가장 좋은 결합 방식) | 44 | 11/28(토) | 차가원 · 고용민 | T-04b, T-05a | 백본 공급원으로 실행 seam이 같은 형식의 레코드를 냄 | 짧은 디버깅 실행 (GPU) | 아니오 | 높음 | ○ |
| T-06e | 🔄 **진행** (구현·CPU 검증·metis GPU 짧은 실행 완료, PR 전 — PR #22 병합 후 PR) | 재현 트랙 실행기: 논문 설정 그대로 전체 데이터 파인튜닝 ×3, `track=reproduction` | 45, 46 | 11/26(목) | 차가원 | T-02a, 논문 하이퍼파라미터 확인(§6) | 결과가 재현 트랙으로 표시되고 판정 입력에서 빠짐 | 짧은 디버깅 실행 + 판정 제외 테스트 | 아니오 | 보통 | ○ |

### T-07 판정 — 🔄 **진행**

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-07a | ⬜ 대기 | **판정 규칙 문서 고정**: `decision_rules.yaml`의 TBD 전부 채우고 `status: locked` | 55 | **11/30(월)** 고정 기한 | 공통 (작성 고용민) | — | §6의 "11/30" 항목이 모두 값을 가짐, 본 실험 결과 레코드가 하나도 없는 상태에서 고정 | 두 사람 승인, 로더가 locked 아니면 본 실험 판정 거부 | 예 | 높음 | ○ |
| T-07b | ✅ **완료** (PR #17 병합 시) | 판정 코어: 규칙 로더, 시드 평균·표준편차, 문서 ID로 짝지은 McNemar, 지지/효과 없음, debug·재현 트랙 제외, 탐색적 가설 무판정 | 56–59 | 11/24(화) | 고용민 | T-01c | 규칙값은 인자로 받아 합성 레코드로 검증 | Seam ② 6개 un-skip (`tests/test_judgment_seam.py`) | 예 | 높음 | ○ |
| T-07c | ⬜ 대기 | 가설별 판정·분석: H1·H3·H2·H4′·H5a·H5b(섞기 효과 F1−B0 / 가중치 학습 효과 F2−F1), 길이 구간별 오차율·최소 표본, ≤512 구간 L512 = Lfull 확인, 무게중심·전역 층 가중치 비율·시드 일관성, H4 SST-2 McNemar, 전역/지역 ablation 표, F4 길이–가중치 순위 상관(전체·라벨별) | 60–72 | 11/28(토) | 고용민 | T-07b (기준값은 T-07a) | 손 계산 예(22층에 가중치 전부 → 무게중심 22, 비율 1)와 일치 | Seam ② 나머지 3개 un-skip + 추가 단위 테스트 | 예 | 보통 | — |

### T-08 실행 — 🔄 **진행**

| ID | 상태 | 제목 | US | 마감 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 | 위험 | 병렬 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-08a | ⬜ 대기 | 디버깅 E2E 스모크: 소규모 데이터로 추출 → 실행 → 판정 → 보고 (`run_kind=debug`) | — | 11/28(토) | 차가원 | T-04b, T-06c, T-07b | 파이프라인 끝까지 통과, 디버깅 레코드가 결과표에 안 나옴 | 실제 실행 로그 | 아니오 | 보통 | — |
| T-08b | ✅ **완료** (11/20 metis에서 캐시 5개 생성·검사 통과, PR #16 병합 시) | 층 벡터 캐시 생성: MNLI, SST-2, IMDB × L128/L512/Lfull | 21 | 11/27(금) | 차가원 | T-04b, T-02b, 캐시 정밀도 결정 | 캐시마다 생성 설정 기록, 형태·순서 검사 통과 | 캐시 검사 스크립트 | 아니오 | 보통 | — |
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

**마지막 verify**: 2026-11-25 Gawoun-t06e(T-06e PR #22 리뷰 반영 후, 로컬 macOS), Python 3.11 및 로컬 모델 캐시 사용 ALL PASSED (452 passed, 4 skipped: 판정 seam T-07c 3개, CUDA 전용 1개). torch·transformers 없는 환경에서는 실제 모델 테스트 3~4개가 skip된다.

**마지막 verify**: 2026-11-25 Gawoun-t06e(T-06e GPU 디버깅·레코드 권한 수정 후, metis 개인 개발 컨테이너), Python 3.11.18, ALL PASSED (447 passed, 3 skipped: 판정 seam T-07c 3개).

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
- 2026-11-18 · T-01d 완료 표시(고용민): `7b93320`(PR #9로 main 병합)에서 `docs/adr/0002-modernbert-base-single-backbone.md` 복구, ADR 0003의 SST-2/H4를 탐색적(경향 비교, 판정 없음)으로 정리, ADR 0005의 존재하지 않는 `tie_last_hidden_states` 문구를 transformers 5.23.0 측정 근거로 교체. 확인: `docs/adr/`에 0001~0009 파일 9개 모두 존재, ADR 0003·0007·GLOSSARY "탐색적 가설" 표현 일치. 상태 열만 갱신이 빠져 있었음. 이로써 T-01a~d가 모두 완료되어 T-01 상위 표제도 완료로 바꿈. 팀 확인 필요: §7 첫 행("ADR 0002 파일 없음")과 ADR 0003 행은 해결된 내용이라 사람이 정리해야 함(에이전트는 §7을 수정하지 않음).
- 2026-11-18 · T-03b(차가원): `inputs.build_tokenized_inputs` 구현 — MNLI·SST-2는 `[CLS] a [SEP] (b [SEP])` 그대로, L128/L512는 내용 토큰 앞 32+뒤 94 / 앞 128+뒤 382로 특수 토큰 포함 정확히 128/512, Lfull은 상한까지 자르지 않음. 문서별 `token_lengths`(자르기 전)·`length_buckets`·`truncated`·`truncated_doc_ratio` 반환. `truncation_rule_name` 추가(T-04b `CacheMetadata.truncation_rule`용). transformers 5.x에 `build_inputs_with_special_tokens`가 없어 특수 토큰을 `cls/sep_token_id`로 직접 붙임. `test_head_tail_truncation_budgets`(가짜 토크나이저)·`test_imdb_head_tail_truncation_lengths`(실제 ModernBERT 토크나이저, 캐시 없으면 skip) un-skip. 실제 IMDB test 200건: L128 잘림 90.5%, L512 14%, Lfull 0%, 앞·뒤 디코딩 확인. **결정 필요**: Lfull 상한 초과 문서의 자르기 방식은 ADR 0006에 없어 `ValueError`로 멈추게 둠(IMDB 최대 3,270 < 4,096이라 현재 영향 없음). 참고: IMDB 원문에 `<br />` 태그가 그대로 있음 — 전처리 여부는 T-03a(데이터 모듈)에서 팀 확인 필요(T-02b 길이 측정도 원문 기준).
- 2026-11-18 · T-04a(차가원): `providers.check_provider_for_run` 추가 — `input_setting`이 다르면 오류, **본 실험(`run_kind=main`)은 `expected_cache_meta` 필수**(디버깅만 생략 허용), 기대 메타가 있으면 `CacheMetadata` 전체 비교(ADR 0005). 기존에는 기대 메타가 없으면 검사를 건너뛰어 다른 설정 캐시가 조용히 통과할 수 있었음(US 26). `FakeLayerVectorProvider.make_synthetic` 구현(train/eval, [N,23,768] float32, seed 결정적, 클래스 균형·클래스별 평균 차이로 학습 신호, MNLI형 eval은 matched/mismatched 반씩, `doc_records` 지정 가능, 기본 메타 = `week1_env.yaml` 고정값). `run_experiment`는 학습 전에 `validate_for_run`만 수행(나머지는 T-06a). `test_mismatched_provider_metadata_raises_error` un-skip, `tests/test_providers.py` 11개 추가. verify ALL PASSED(39 passed, 22 skipped). **팀 확인 필요**: `CacheMetadata`에 데이터셋 이름이 없어 MNLI 캐시를 SST-2 실행에 써도 메타 검사로 못 잡음 — SPEC 캐시 생성 설정에도 없어 계약 변경 여부 논의 필요.
- 2026-11-18 · T-03a(고용민): `data.py` 구현 — `DATASET_SPECS`(MNLI·SST-2·IMDB·AG News, 데이터셋 이름만으로 전환), `rows_to_records`(doc_id `mnli-validation_matched-000012` 형식, MNLI matched/mismatched를 `genre_split`으로 태그, 범위 밖 라벨은 `ValueError`), `load_dataset_splits`(`{"train","eval"}`, HF `datasets`는 함수 안에서 늦게 import, 원문 그대로 유지), `make_tuning_split`(클래스 비율 유지·시드 결정적·튜닝 레코드 `split="tuning"`), `sample_small_dataset`(클래스당 `per_class`·지역 `random.Random(seed)`·입력 순서 무관·클래스 부족 시 `ValueError`). `tests/test_data.py` 21개 추가, `test_small_data_sampling_...` un-skip. verify ALL PASSED(68 passed, 25 skipped). **미확인**: 이 환경에 `datasets`가 없어 실제 HF 다운로드 경로(`load_dataset_splits`)는 실행해 보지 못함 — 설치된 환경에서 3개 데이터셋 건수·라벨 분포 확인 필요. **SPEC 미정 항목은 값을 정하지 않음**: 튜닝 분할을 하나로 고정할지는 호출자가 넘기는 `seed`로, 소량 표본이 튜닝 분할을 포함할지는 호출자가 넘기는 `train_records`로 결정(T-06a). 튜닝 분할이 클래스 비율 유지(stratified)인 것은 구현 선택이므로 팀 확인 필요.
- 2026-11-18 · T-03a 교차검증 반영(고용민): 다른 에이전트 리뷰 9건을 실제 코드·실행으로 대조, 8건 반영. (1) `make_tuning_split`·`sample_small_dataset`이 같은 시드에서 같은 난수열을 써 **소량 표본이 튜닝 분할 안에 통째로 들어가던 문제**(MNLI 규모 재현: 클래스 0 1,000/1,000 중복) → 용도별 난수열(`random.Random("layer_fusion.data:<용도>:<seed>")`)로 분리, 수정 후 겹침 298/3,000(≈10%, 독립 기대값). (2) `seed`를 두 함수 모두 필수 키워드 인자로(빠뜨린 시드가 0으로 조용히 재사용되지 않게), `sample_small_dataset`에 필수 `num_classes` 추가(빈 입력·빠진 클래스 차단). (3) 빈 입력·중복 `doc_id`·허용되지 않은 split(튜닝 분할은 `train`만, 소량 표본은 `train`/`tuning`만, `eval` 거부) 오류. (4) 튜닝 분할도 `doc_id` 정렬 후 추출(입력 순서 무관). (5) 클래스가 작아 튜닝 분할이 0건 또는 전체가 되면 오류(`round(0.1×5)=0`). (6) 가짜 `datasets` 모듈 주입으로 `load_dataset_splits` 테스트, AG News 변환 테스트 추가. (7) `test_small_data_sampling_...` seam 테스트는 runner를 거치지 않아 다시 skip(T-06a에서 un-skip, 데이터 수준은 `tests/test_data.py`가 검증). 반영하지 않음: `FakeLayerVectorProvider.make_synthetic(doc_records=...)`가 `train`/`eval` 외 split(`tuning`)을 조용히 버리는 점 — `providers.py`(차가원 담당)와 T-06a 설계 문제라 **팀 확인 필요**: 캐시는 `train`/`eval`만 있으므로 T-06a에서 튜닝 분할을 `train` 배치의 doc_id 부분집합으로 고를지, 공급원에 `tuning` split을 둘지 결정.
- 2026-11-18 · PR #8 리뷰 반영(차가원): 고용민 교차검증 반영 — `check_provider_for_run`의 `input_setting` 비교에서 `is not None` 조건 제거(MNLI·SST-2 실행에 IMDB 캐시가 들어오는 경우도 차단, 테스트 추가), 실제 ModernBERT 토크나이저로 MNLI 직접 결합 `[CLS] a [SEP] b [SEP]` = `tokenizer(a, b)` 확인 테스트 추가. 리뷰에서 팀 합의: Lfull 상한 초과 시 오류 유지, IMDB `<br />` 원문 유지, `CacheMetadata`에 데이터셋 필드 추가(계약 변경, 고용민이 SPEC부터 반영), 본 실험 `expected_cache_meta` 필수(T-06b에서 자동 생성), **캐시 float16 저장·로드 시 float32**(디버깅 추출에서 최대 절댓값 1회 확인 후 확정), **토큰 평균에 특수 토큰 포함(true)** → `week1_env.yaml` 기록은 T-04b에서.
- 2026-11-18 · T-03b·T-04a 완료(차가원): PR #8 고용민 승인 후 병합(`aee4f10`). /code-review(표준·명세) 반영분 포함. T-04b로 넘긴 항목: `TokenizedInputs` 데이터클래스, `truncation_rule` 문자열 형식 확인, 문서별 시야 비율, 데이터셋 필드·IMDB `input_setting` 검사(계약 변경), `expected_cache_meta` 필수 규칙 SPEC 반영, `cache_dtype`·`include_special_tokens_in_pool` 기록.
- 2026-11-18 · T-01d 완료 표시(차가원, 담당 공통): PR #9(고용민) 병합 확인 후 상태 ✅, T-01 묶음 제목 ✅로 갱신.
- 2026-11-18 · 규칙 추가(차가원·고용민 합의): 상태가 PR 병합보다 한 단계씩 늦게 main에 반영되던 문제(T-01d·T-03b·T-04a) 때문에, §0에 "상태 변경은 Task를 끝내는 PR 안에 함께 넣는다" 규칙을 추가. §0은 사람만 고치는 부분이라 두 사람 합의를 근거로 수정.
- 2026-11-18 · T-03a PR #11(고용민): §0 규칙에 따라 PR 안에서 T-03a를 `✅ 완료 (PR #11 병합 시)`로, T-03b가 이미 완료라 T-03 묶음 제목도 ✅로 갱신. 리뷰 중 막히면 👀/⛔로 되돌린다.
- 2026-11-18 · 튜닝 분할 방식 팀 결정(고용민·차가원, PR #11 리뷰): **라벨 층화**로 정함 — 클래스마다 `round(0.1 × 클래스 크기)`건(.5는 짝수 쪽으로 반올림), 0건이거나 클래스 전체가 되면 오류. 차가원이 코드·건수를 직접 확인: MNLI 39,270 / SST-2 6,735 / IMDB 2,500(세 데이터셋 모두 .5로 떨어지는 클래스 없음). 장르 층화는 아님(GLUE MNLI 열이 `premise, hypothesis, label, idx`뿐). SST-2는 클래스 비율이 학습 44.2 : 55.8 / 평가 49.1 : 50.9로 불균형이라 튜닝 분할에도 그대로 남으며, 보고서 한계 절에 적는다. **여전히 열린 항목(SPEC 미정, 값은 정하지 않음)**: 튜닝 분할을 시드마다 다시 뽑을지는 차가원 T-06a 설계(시드별 표본을 한 번 만들어 모든 결합 방식이 공유)와 연결되므로 **T-06a 시작 전(11/25 전)에** 함께 정함. 소량 표본이 튜닝 분할 문서를 포함할지·소량 조건의 튜닝 분할도 같은 시점에 정함.
- 2026-11-19 · T-04b PR 1(차가원): `TokenizedInputs` 데이터클래스·배치 토크나이징·문서별 시야 비율(US 18), `extract_layer_vectors`(층마다 즉시 패딩 제외 평균, 길이순 배치 후 원래 순서 복원, memmap 출력 가능, 고정 transformers 버전 확인, float16 범위 검사·왕복 오차 보고), `save_layer_cache`(`<dir>/metadata.json` + `<split>/vectors.npy` 등, `<split>.partial`에 쓰고 성공 시 이름 변경, 기존 분할 덮어쓰기 거부), `CacheProvider`(memmap 로드). `max_length_cap`은 설정별 고정값(128/512/상한). `make_synthetic`은 train/eval 외 split 거부(PR #11 합의). Seam ③ 남은 4개 un-skip, `/code-review` 반영. verify ALL PASSED(62 passed, 18 skipped). **PR 2(데이터 모듈 PR #11 병합 후)**: `scripts/extract_cache.py`, float16 범위 실측, `week1_env.yaml`의 `cache_dtype`·`include_special_tokens_in_pool` 기록, 시야 비율·토큰 길이 저장.
- 2026-11-19 · T-04b PR #12 리뷰 반영(차가원): 고용민 교차검증 반영 — 층 벡터에 NaN/inf가 있으면 배치마다 멈춤(float32·float16 모두, `check_float16_range`도 NaN 차단: 누적 `max()`가 NaN을 버려서 손상 캐시가 통과할 수 있었음), `CacheProvider.get_split`이 형태 [N, 23, 768]·dtype(메타데이터와 일치)·`token_lengths`/`genre_splits` 길이를 확인하고 `.partial` 이름은 거부, 장르 없는 데이터(SST-2·IMDB)는 `genre_splits=None`, 밖에서 넘긴 모델의 `attn_implementation`이 기록값과 다르면 오류. §8 상태 요약 절은 되돌림(상태 기준은 §4 표 하나). **PR #8 합의 변경 기록**: "로드 시 float32" 대신 `CacheProvider`는 memmap으로 저장 dtype 그대로 돌려주고 소비자(T-06a)가 미니배치마다 float32로 올림 — MNLI 학습 캐시가 float32 약 28GB / float16 약 14GB라 24GB 메모리 제한에서 통째로 float32 변환이 불가하기 때문. `truncation_rule` 이름(`head_32_tail_94`·`head_128_tail_382`·`none`)은 고용민 확인으로 확정. `CacheMetadata` 데이터셋 필드는 고용민 계약 PR(SPEC → contracts → tests) 병합 후 T-04b PR 2에서 맞춤. PR 2로 넘긴 정리: `_max_length_cap`을 `inputs.py`로, `save_layer_cache`가 묶음 타입을 받게, `ExtractionResult.report` 데이터클래스화.
- 2026-11-19 · T-04b PR 2(차가원): `scripts/extract_cache.py` 구현(`layer_fusion.cache_build`) — 고정 설정은 `week1_env.yaml`에서만 읽고 `null`이면 멈춤, `--limit` 디버깅 캐시는 `data/cache_debug/`(git 제외), `--report-only`는 float32 추출 보고만. writer를 `begin_split`(`<split>.partial/vectors.npy` memmap) → 추출기가 배치마다 디스크에 직접 씀 → `finish_split`(인덱스 파일·`report.json` 쓰고 이름 변경) / 실패 시 `abort_split`으로 나눠 MNLI 캐시를 메모리에 올리거나 두 번 쓰지 않음. 문서별 시야 비율(`view_ratios.npy`, US 18)·토큰 길이 저장, `LayerVectorBatch.view_ratios` 추가(T-09a에서 사용 가능). PR #12에서 넘긴 정리 3건 반영(`inputs.max_length_cap`, `ExtractionReport` 데이터클래스 + 추출 시간, `finish_split(result)`). **float16 실측**(CPU, `docs/measurements/float16_range_check.json`): 표본 MNLI 2,000·SST-2 1,000·IMDB 3설정 300건 + 가장 짧은 문서 300건(짧을수록 값이 큼) → 최대 |값| 7,994(SST-2, 층 21·차원 251 이상치), float16 최대 65,504 대비 약 8.2배 여유, 반올림 상대 오차 ≤ 2^-11(약 4.9e-4, PR #13 리뷰에서 2.4e-4를 정정). PR #8 합의 조건(실측 후 확정)을 채워 `cache_dtype: "float16"`, `include_special_tokens_in_pool: true` 기록(공유 파일, 고용민 리뷰로 확정). §0 규칙에 따라 T-04b를 `✅ 완료 (PR #13 병합 시)`로 표시. **후속**: `CacheMetadata` 데이터셋 필드(고용민 계약 PR 후).
- 2026-11-19 · T-04b PR #13 /code-review 반영(차가원): 표준·명세 두 축 리뷰 결과를 모두 반영. (1) **SPEC §결정 필요 사항에 두 결정 기록** — 토큰 평균에 특수 토큰 포함(true), 캐시 저장 정밀도 float16(yaml만 바뀌고 SPEC이 미정으로 남아 AGENTS.md §2·§4와 충돌하던 것). 공통 파일이라 고용민 PR 승인이 곧 확정. (2) **반올림 상대 오차 수치 정정**: ≤ 2^-12(2.4e-4) → ≤ 2^-11(약 4.9e-4, float16 가수부 10비트, yaml·측정 JSON). 결론(float16)은 그대로. (3) **실측 재현 가능**: `--shortest N`(가장 짧은 입력 N건)과 보고서의 `max_abs_slot`·`max_abs_dim`·`min_input_length` 추가. 재실행으로 MNLI 6,374.5654 일치, SST-2는 길이 4 동률이 많아 N=300은 임의 부분집합(7,942.9)이고 N=3000에서 7,993.5947 일치. IMDB(L512) 가장 짧은 300건도 새로 측정: 3,050.9(같은 층 21·차원 251). (4) **debug 캐시가 본 캐시 폴더에 섞이지 않게**: `--limit`/`--shortest` 표본은 `data/cache_debug/` 아래에만 쓸 수 있음(`CacheMetadata`에 표본 표시가 없어 폴더가 유일한 구분, AGENTS.md §4). (5) **본 캐시는 고정 `cache_dtype`만**: `--dtype`가 다르면 거부, 덮어쓰기는 debug 표본에서만. (6) 정리: 길이 검사 `check_split_lengths` 하나로 writer·`CacheProvider` 공유, 색인 파일 묶음 `SplitIndex`로 `_finish` 인자 9→5개, `ExtractionEnv.cache_metadata`, CLI가 `CACHE_DTYPES`·`cache_name` 재사용. T-04 표제 ✅, 상태 표기 `PR #13`. verify ALL PASSED(133 passed, 18 skipped). **PR #14와 병합 순서 주의**: 두 PR 모두 SPEC §결정 필요 사항(인접 줄)과 §8 끝에 추가하므로 나중 병합 쪽에서 충돌 가능, 양쪽 내용을 모두 남기면 됨.
- 2026-11-19 · T-04b PR #13 고용민 리뷰(변경 요청) 반영(차가원): (1) **Windows에서 verify 2건 실패 수정** — 열린 `np.memmap` 핸들 때문에 `.partial` 폴더 이름 변경이 `WinError 5`로 실패하고, 실패 경로의 `rmtree(ignore_errors=True)`는 조용히 실패해 `.partial`이 남았음(macOS·Ubuntu CI는 POSIX라 통과). `close_memmap`을 추가해 writer가 쓴 memmap을 모두 명시적으로 닫음: `_write_vectors`(flush 후 닫기, 복사 경로는 finally), `_finish`(행 수 확인용 매핑 닫기), `save_layer_cache`(`begin_split` 반환값), `abort_split(..., out)`(지우기 전에 닫기, `build_cache` 실패 경로). `abort_split`은 폴더가 남으면 경고를 냄. `finish_split` 뒤에는 `result.vectors`(닫힌 memmap)를 쓰면 안 되고 `CacheProvider`로 다시 읽음. **테스트**: Windows PC 없이 재현하려고 `tests/conftest.py`에 `windows_file_locks` 픽스처 추가 — 열린 memmap이 있는 폴더의 rename·rmtree를 Windows처럼 막음. 수정 전 코드에서 리뷰의 2건(+ 실제 모델을 쓰는 `build_cache` 테스트 1건)이 그대로 실패하는 것을 확인한 뒤 고침. 캐시 테스트 3개 모듈 전체에 적용. (2) T-04 표제·§8 문구·상대 오차는 앞 커밋(`a5ab584`)에서 반영됨, 이번에 §8 PR 2 로그의 2.4e-4와 docstring 2곳의 'T-04b PR 2에서 확정' 문구도 정리. verify ALL PASSED(135 passed, 18 skipped). **실제 Windows 재확인은 고용민 PC에서 필요.**
- 2026-11-19 · T-05a 구현(고용민): `fusion.py`에 `PerLayerNorm`(학습 파라미터 0), B0(슬롯 22)·F1(23개 평균)·F2(전체/전역만 8/지역만 14, 임베딩 제외)·F3(슬롯 19~22 이어붙이기), `count_trainable_parameters`, `extract_layer_weights`(F2만 반환, 나머지 `None`) 구현. F4·B0+는 `NotImplementedError`(T-05b), 층 부분집합은 F2 외에는 `ValueError`. **F2 층 가중치 정규화는 softmax(층당 로짓 1개, 0으로 초기화 → 시작 시 F1과 동일)로 구현**하고 같은 PR에서 SPEC §결정 필요 사항·모듈 6에 F2 결정을 기록(F4는 미정, T-05b에서 게이트 구조와 함께 확인). SPEC은 공통 파일이라 **차가원 PR 승인이 곧 결정 승인** — 반려되면 구현도 함께 고친다. `tests/test_fusion.py` 신규, `test_execution_seam.py`의 3개 un-skip(파라미터 수 테스트는 T-05b로 사유 변경). verify ALL PASSED(117 passed, 22 skipped). `.venv`는 이 PC에서 새로 만들어 `requirements.lock` 설치(torch 2.15.0+cpu). 아직 PR 전이라 🔄 — PR을 올릴 때 §0 규칙대로 `✅ **완료** (PR #N 병합 시)`와 T-05 표제를 함께 갱신한다(T-05b가 남아 있으므로 T-05 표제는 🔄 유지).
- 2026-11-19 · T-05a PR #14(고용민): §0 규칙에 따라 PR 안에서 T-05a를 `✅ 완료 (PR #14 병합 시)`로 갱신. T-05b가 남아 있어 T-05 묶음 제목은 🔄 유지. 리뷰 중 막히면 👀/⛔로 되돌린다. PR #14는 SPEC 결정(F2 = softmax)을 함께 담은 공통 파일 PR이라 차가원 승인이 곧 결정 승인이다.
- 2026-11-19 · T-05a 교차검증 반영(고용민): 다른 에이전트 리뷰를 실제 코드·실행과 대조. 반영: (1) §8의 `2026-11-18 · 튜닝 분할 방식 팀 결정` 줄이 작업 트리에서 빠져 있던 것 복구(`3ec044e` 커밋이 T-05a 편집 뒤에 들어와 그대로 커밋하면 삭제될 상태였음). (2) 테스트 보강 — 비균일 F2 가중합이 수동 계산과 일치·F1과 달라짐, 모든 층이 같으면 F2 = B0(숨은 스케일 없음), LOCAL_ONLY 제외 슬롯 불변, F3 19~22 각 슬롯 반응, B0·F1·F2·F3 모두 층별 a·x+b 불변, batch 1, 0이 아닌 기울기, 노이즈 시드 고정. 결함 주입(F2가 가중치 무시·F3 슬롯 누락·숨은 γ)이 모두 새 테스트에 걸리는 것 확인(기존 테스트는 F2 가중치 무시를 못 잡음). (3) seam 테스트 2개를 내부 속성(`named_parameters` 이름·`head.slots`) 대신 공개 함수·출력으로 검사(SPEC "바깥 행동만 검사"), T-06a에서 `run_experiment` 경유 재검토 메모. (4) `hidden_size < 1` 오류, 기본값 `layers.HIDDEN_SIZE`, F1 슬롯을 `slots_for_subset(ALL)`로. 반영 안 함: `fusion.py`의 torch 지연 import(`nn.Module` 정의 자체가 torch이고 `__init__`·판정·보고 모듈은 import하지 않음), float16 입력 처리(PR #12 리뷰 변경 합의에 따라 `CacheProvider`는 저장된 float16 memmap을 그대로 넘기고 소비자(T-06a 러너)가 헤드 입력 전 미니배치마다 float32로 변환하므로 `fusion.py`는 float32 입력 전제 유지 — PR #14 리뷰에서 정정). softmax는 위 항목대로 SPEC에 기록. verify ALL PASSED(134 passed, 22 skipped).
- 2026-11-19 · T-05a PR #14 리뷰 반영(고용민): 차가원 리뷰 확인 — 요청한 3건(F2=softmax 결정·SPEC 문구, F4 미정 범위, 슬롯 매핑) 모두 동의. 위 §8 교차검증 항목의 float16 입력 전제를 PR #12 리뷰 합의(`CacheProvider`는 float16 memmap 그대로 반환, T-06a 러너가 헤드 입력 전 미니배치마다 float32로 변환; `fusion.py` 수정 없음)에 맞게 정정. 병합 순서상 PR #13과 병합 시 `docs/SPEC.md` §결정 필요 사항(F2 softmax·특수 토큰 포함 인접 두 줄)과 `docs/03-plan.md` §8 끝은 양쪽 모두 유지 예정.
- 2026-11-19 · T-06a 구현(차가원): `runner.run_experiment`(백본 고정) + `training.py`(`RowView`·`train_head`·`evaluate`) 구현. **SPEC §결정 필요 사항 4건을 결정 제안으로 기록(공통 파일, 고용민 PR 리뷰 승인이 곧 결정 승인)**: (1) 튜닝 분할은 상수 시드 0으로 하나만 떼고, 학습률은 설정 조합마다 선택 실행(첫 시드)에서 후보 3개를 모두 학습해 튜닝 오차율로 고르며 나머지 시드는 `RunConfig.fixed_lr`로 재사용(ADR 0007 "고른 학습률로 시드를 반복", 시드 표준편차에 학습률 선택 흔들림이 섞이지 않게). 동률은 튜닝 손실 → 후보 순서. (2) 조기 종료는 튜닝 분할 오차율(엄격 감소만 개선, 최적 에폭으로 복원), 학습은 90% — GLOSSARY 튜닝 분할 정의에 "조기 종료" 추가. (3) 소량 표본은 90%에서 클래스당 1,000건, 튜닝 분할은 전체 조건과 공유(H2가 학습 데이터 양만 바꾸게, 한계 절에 기록). (4) `train_seconds`는 보고하는 모델의 학습 루프만, 학습률 탐색 전체는 `lr_search_seconds`로 따로(`extract_time_included_in_train_time`은 여전히 null). 구현 기본값: AdamW 기본 설정·cross-entropy. **계약 변경**: `RunConfig.fixed_lr`·`lr_selection_seed`, `ResultRecord.lr_selection_seed`·`lr_search_seconds`(SPEC §데이터 형태 → contracts → `test_contracts.py` 순). 캐시는 미니배치마다 정렬한 행을 가져와 float32로 올림(PR #12 합의), 전역 torch RNG는 건드리지 않음(`fork_rng`). 본 실험은 학습률 후보 정확히 3개·에폭·patience·배치 크기 필수, 파인튜닝은 `NotImplementedError`(T-06d). Seam ① 5개 un-skip, `tests/test_runner.py` 신규 21개(선택 실행 = 같은 시드의 `fixed_lr` 재실행 모델 일치, 조기 종료 복원, IMDB 길이 구간·잘린 비율, float16 입력 등). `make_synthetic`은 모든 학습률에서 튜닝 오차 0이라 선택 테스트가 공허해져서, 클래스가 겹치는 `noisy_provider` 픽스처를 `conftest.py`에 추가. verify ALL PASSED(212 passed, 10 skipped). **후속**: 선택 실행 → `fixed_lr` 전달 순서(T-06b/c), 실제 MNLI memmap 학습 속도(T-08a), F4 문서별 층 가중치(T-05b).
- 2026-11-19 · T-06a PR #15(차가원): §0 규칙에 따라 PR 안에서 T-06a를 `✅ 완료 (PR #15 병합 시)`로 갱신. T-06b~e가 남아 있어 T-06 묶음 제목은 🔄 유지. PR #15는 SPEC·GLOSSARY·contracts 공통 파일 변경을 담고 있어 고용민 승인이 곧 결정 1–4 승인이다. 리뷰 중 막히면 👀/⛔로 되돌린다.
- 2026-11-19 · T-06a PR #15 리뷰 반영(차가원): 고용민 변경 요청 반영. **결정 제안 수정(SPEC §결정 필요 사항)**: (1) 설정 조합 키에 백본 조건 추가, 재사용 실행은 `lr_selection_seed`로 선택 실행 레코드를 참조(T-06b·T-07·T-09). (2) 조기 종료 개선 판정을 (튜닝 오차율, 튜닝 손실) 순서 비교로 바꿈(학습률 동률 규칙과 같음, IMDB·SST-2 오차율 동률 대비), SPEC US 3 문구를 GLOSSARY와 맞춤. (3·4) 튜닝 분할 공유는 유지, `train_seconds`는 학습 단계만 재고 에폭별 튜닝 평가 시간은 새 필드 `ResultRecord.tuning_eval_seconds`로 분리(MNLI 소량은 학습 3,000건에 튜닝 39,270건이라 섞으면 전체/소량 비교 왜곡). (5) AdamW 가중치 감쇠 0.01은 2차원 이상 가중치 행렬에만 적용, bias·F2 `layer_logits`는 0(감쇠가 F2를 F1 쪽으로 당기는 편향 제거). **발산 처리**: 학습·튜닝 손실이 유한하지 않으면 그 후보는 발산으로 선택에서 제외(점수 `inf`), 모두 발산하거나 `fixed_lr` 발산이면 `TrainingDivergedError`, 평가 손실이 유한하지 않으면 오류. **필수 수정**: 학습률 `isfinite` 검사, patience ≥ 1, 재현 트랙 거부 + `run_id`에 `repro-` 접두, IMDB는 `input_setting`·`token_lengths` 필수이고 IMDB가 아니면 `input_setting` 금지, MNLI 장르는 matched/mismatched만(없으면 학습 전에 오류), 모든 CUDA 디바이스 RNG 보존, `train_head` 입력 검사(`assert` 제거). 정리: `fits` 튜플을 `_Candidate`로, conftest가 `layers` 상수 사용. 리뷰 사례(`lr=inf`·`nan`, `patience=0`, 재현 트랙 `run_id`)를 다시 실행해 모두 거부되는 것 확인. verify ALL PASSED(228 passed, 11 skipped, CUDA 전용 1개 포함).
- 2026-11-20 · T-08b 층 벡터 캐시 생성(차가원, metis GPU 컨테이너): 선행 조건(T-04b·T-02b·float16 결정)이 모두 끝나 Task ID 순서보다 먼저 실행(GPU 여유 시간 활용, 본 실험 아님). 5개 캐시 모두 [N, 23, 768] float16, NaN/inf 0, doc_id·라벨 순서 = `load_dataset_splits`, `.partial` 잔여 없음. 생성 설정: transformers 5.23.0, sdpa, compile 끔, 특수 토큰 포함. `mnli` train 392,702 / eval 19,647(matched 9,815, mismatched 9,832), 14G, 3m01s, 최대 절댓값 6,375/5,052; `sst2` 67,349 / 872, 2.3G, 28s, 7,994/6,463; `imdb_L128` 25,000 / 25,000, 1.7G, 53s, 3,051/3,587; `imdb_L512` 1.7G, 1m44s; `imdb_Lfull` 1.7G, 2m04s(IMDB 최대 절댓값 세 길이 동일). 합계 약 8m10s·약 21GB. 잘린 문서 비율(train/eval): L128 88.8%/88.2%, L512 14.0%/13.1%, Lfull 0%. IMDB `token_lengths`는 자르기 전 길이(최대 train 3,270, eval 2,971). L512 중복 실행: 두 번째 실행이 CPU에서만 10초(17:25:28–17:25:37) 겹친 뒤 `.partial` 가드로 GPU 작업 전에 멈춰 시간 측정에는 영향 없음; `logs/t08b_imdb_L512.log`에 두 실행이 섞여 있어 캐시별 `report.json`이 기준. 디버깅 캐시 `data/cache_debug/mnli_limit200`, `data/cache_debug/imdb_Lfull_limit50` 유지. 실행 로그용 `logs/`를 `.gitignore`에 추가. T-06a 리뷰와 섞지 않으려고 PR #15 병합 후 별도 PR로 기록.
- 2026-11-23 · T-07b 구현(고용민, 브랜치 YongMin, 아직 PR 전이라 🔄): `judgment.py`에 `load_decision_rules(path, *, require_locked=False)`(YAML 값 그대로, `null`은 `None` 유지, 가설별 type/priority는 `DecisionRules.extra["hypotheses"]`로 보존), `compute_paired_mcnemar`(doc_id 짝짓기, 시드별 정확 이항검정, 문서 집합·중복·정답 불일치는 `ValueError`), `seed_mean_std`, `select_main_records`(debug·재현 트랙 제외), `judge_hypotheses`(H1 판정 + 탐색적 가설 H4·H5c는 항상 `EXPLORATORY`) 구현. Seam ② 6개 un-skip, 나머지 3개는 skip 사유를 `TODO(T-07c)`로 바꿈. `tests/test_judgment_seam.py`(합성 레코드) 보강과 신규 `tests/test_judgment_core.py`. 결함 주입 8종 중 7종을 테스트가 잡았고 남은 1종(방향 가드)은 `improvement >= seed_std`가 이미 포함하는 죽은 코드라 제거. verify ALL PASSED(156 passed, 29 skipped). `/code-review`(표준·명세) 반영: McNemar가 양측검정이라 후보가 유의하게 *나빠진* 시드도 "유의"로 세어지던 구멍을 시드별 방향 조건(그 시드에서 후보 오차율이 낮을 때만 p를 인정, 아니면 1.0)으로 막고 테스트 추가(수정 전 코드는 잘못 `지지`가 나옴을 RED로 확인). **SPEC이 정하지 않아 구현에서 고른 해석(T-07a에서 확정·SPEC 반영 필요)**: ① "시드 간 표준편차" = 비교하는 두 설정의 표본표준편차(ddof=1) 중 큰 값 ② McNemar는 정확 이항검정 ③ 여러 시드 결합은 전략 이름으로 고르며 현재 `all_seeds_significant`·`majority_of_seeds`만 지원(규칙값이 `None`이거나 모르는 이름이면 `ValueError`, 결합값은 "k번째로 작은 p"). `mcnemar_alpha`·`h1_target_fusion_rule`이 `None`이면 판정을 거부한다(TBD를 추측하지 않음). 이 PR의 `judgment.py`는 공유 고위험 파일이라 차가원 리뷰 필요. **팀 확인 필요**: (a) §7의 "`hypotheses` type/priority를 계약에 넣을지" 행은 이번에 "로더가 `extra`로 읽음"으로 정리됨(`contracts.py` 변경 없음) — §7은 사람이 고칠 것. (b) H1이 MNLI의 어느 오차율을 쓸지(전체 `eval_error_rate` vs matched만)는 SPEC에 없어 T-07b는 레코드의 `eval_error_rate`와 전체 예측을 사용, matched/mismatched 분리는 T-07c(H4′)에서 정함. (c) 확증 가설 중 H3·H2·H4′·H5a·H5b는 평가자가 아직 없어 `judgments`에서 빠짐(NO_EFFECT로 위장하지 않음, T-07c가 같은 평가자 레지스트리에 추가). (d) T-07c 유의: `mean_diff`는 기준선 − 후보(양수 = 후보가 낫다)이고 SPEC의 "F1 − B0"·"F2 − F1" 표기와 부호가 반대일 수 있어 H5b 분해에서 부호 규칙을 정할 것, 판정표 키는 YAML id(`H4_prime`, SPEC의 H4′)를 따름. (e) T-07a 유의: `judge_hypotheses`는 `rules.status`를 검사하지 않음(`load_decision_rules(require_locked=True)`만 거부), `multiple_comparison_correction`은 H1이 단일 비교라 아직 읽지 않음(여러 결합 방식을 비교하는 T-07c에서 필요), 결합 전략 이름·`max(std)`·정확 McNemar는 T-07a 확정 대상.
- 2026-11-23 · T-07b 교차검증 반영(고용민): 다른 에이전트(Antigravity) 리뷰 3건을 실제 코드·실행으로 대조, 2건 반영. (1) **한 비교 안에 다른 실험 칸 레코드가 섞여도 통과하던 문제**: `_mnli_frozen_full`이 `input_setting`·`epochs`·`lr_candidates` 등을 보지 않아 MNLI B0/F1에 L128·L512·Lfull·`None`이 섞여도 `지지`가 나왔음 → `_judge_comparison`이 기준선·후보 레코드 전체에서 `RunConfig` 중 `seed`·`fusion_method` 외 필드가 모두 같은지 확인하고 다르면 필드 이름과 함께 `ValueError`(조용히 걸러내지 않음; SPEC 모듈 7 "모든 결합 방식에 같은 에폭 수·같은 조기 종료 기준", 입력 설정은 IMDB만). (2) **빈 예측·비유한 오차율**: `predictions=[]`이면 McNemar가 p=1.0 → `효과 없음`으로 조용히 판정되던 것을 `ValueError`로, `eval_error_rate`가 NaN/inf이면 `statistics` 내부 `AttributeError` 대신 `seed_mean_std`가 `ValueError`. 테스트 7개 추가(수정 전 코드에서 7개 모두 실패 확인 후 수정), 리뷰의 혼재 시나리오를 직접 재실행해 `ValueError` 확인. 반영 안 함: (3) `multiple_comparison_correction=None` 미거부 — `h1_target_fusion_rule`이 목록이면 이미 `unsupported ... shape`로 거부되어 단일 비교만 가능하고, 위 (e)에 T-07c 과제로 기록되어 있음. **T-07c 유의**: H5a처럼 `input_setting`이 비교 축인 가설은 `_COMPARISON_FREE_FIELDS`를 비교별로 넘기도록 일반화해야 함. MNLI 레코드에 `input_setting`이 들어오는 것 자체는 실행 쪽에서 막음(PR #15 리뷰 반영으로 `run_experiment`가 IMDB 외 `input_setting`을 거부).
- 2026-11-23 · T-07b PR #17(고용민): PR #15(T-06a)가 main에 먼저 병합되어 `origin/main`을 YongMin에 병합(§8 충돌만 발생, 날짜순으로 양쪽 유지). 병합으로 들어온 `RunConfig.fixed_lr`·`lr_selection_seed`는 설정마다 첫 시드가 학습률을 고르고 나머지 시드가 재사용하는 구조라 **한 칸 안에서도 시드·결합 방식마다 값이 다름** → 위 실험 칸 검사가 정상 레코드를 거부하던 것을 테스트로 확인(RED)한 뒤 두 필드를 비교 허용 필드에 추가, 회귀 테스트 `test_lr_reuse_fields_may_differ_inside_one_cell` 추가. §0 규칙에 따라 PR 안에서 T-07b를 `✅ 완료 (PR #17 병합 시)`로 갱신, T-07a·T-07c가 남아 T-07 묶음 제목은 🔄 유지. `judgment.py`는 공유 고위험 파일이라 차가원 리뷰 필요. 리뷰 중 막히면 👀/⛔로 되돌린다. **PR 번호 주의**: 작성 시점 마지막 번호가 #16(T-08b)이라 #17로 적었으며, 다르면 PR에서 고친다. **미확인**: ModernBERT 가중치가 이 PC에 없어 실제 모델 테스트 12개(§8 2026-11-19 PR #13의 "Windows 재확인" 포함)는 여전히 skip.
- 2026-11-23 · T-05b 구현(고용민, 브랜치 YongMin, 아직 PR 전이라 🔄): `fusion.py`에 `F4Head`(층별 정규화된 23개 층 벡터의 평균 → `Linear(768→23)` 게이트 → softmax 문서별 가중치 → 가중합 → 선형 분류기, 게이트 0 초기화로 시작 시 F4 = F1)와 `B0PlusHead`(마지막 층 벡터 → ReLU MLP, 은닉 3클래스 26 / 2클래스 25, 그 밖의 클래스 수는 SPEC이 TBD라 `ValueError`), `PARAM_MATCH_TOLERANCE = 0.01`, 공개 함수 `extract_document_layer_weights(head, layer_vectors)`(F4만 `[batch][23]`, 나머지 `None`) 구현. `extract_layer_weights`는 F2 전용 유지(F4는 `None`). 파라미터 수: F4 19,994 / 19,225, B0+ 20,075 / 19,277(차이 0.41% / 0.27%, 모두 1% 이내). `tests/test_fusion.py`에 F4·B0+ 테스트 추가(기존 `NotImplementedError` 테스트 제거), `test_execution_seam.py`의 `test_trainable_parameter_counts_and_b0_plus_matched_to_f4` un-skip(SPEC 리터럴 값과 대조, 코드 공식 재계산 아님). 결함 주입 3종(B0+ 은닉 크기 변경, 게이트가 입력 무시, 게이트가 정규화 전 벡터를 봄)을 테스트가 모두 잡는 것 확인 후 복원. `run_experiment`에 `noisy_provider`로 F1·F4·B0+를 돌려 `trainable_params`가 헤드와 일치함을 확인. verify ALL PASSED(288 passed, 16 skipped). **SPEC 결정 제안 3건을 공통 파일에 기록(차가원 PR 승인이 곧 확정, 반려되면 구현도 함께 고침)**: ① F4 형태(위 게이트 구조·입력) ② F4도 softmax ③ B0+와 F4 파라미터 수 허용 차이 상대 1% 이하(+ B0+ 활성화 ReLU는 구현 기본값). **후속(T-05b 범위 밖, 차가원)**: (a) 러너(`training.evaluate`/`run_experiment`)가 F4 평가셋의 문서별 가중치를 `PredictionRecord.layer_weights`에 채우는 연동은 아직 없음 — 현재 F4를 돌리면 학습·예측은 되지만 문서별 가중치 필드가 `None`이며 T-08h(Lfull F4)·T-07c(길이–가중치 순위 상관) 전에 필요. (b) `training.py`의 가중치 감쇠 규칙(2차원 이상에만 0.01)이 F4 `gate.weight`에도 걸려 게이트가 0(= F1) 쪽으로 당겨질 수 있음 — F2 로짓에서 같은 이유로 감쇠를 뺀 선례가 있으니 F4 게이트에도 적용할지 협의 필요. (c) AG News 전환 시 B0+ 은닉 크기는 여전히 TBD.
- 2026-11-23 · T-05b `/code-review`(표준·명세) 반영(고용민): 버그는 없다는 결론. 반영 — 쓰이지 않는 `B0PlusHead.mlp_hidden` 제거, `backward` 없이 항상 통과하던 공허한 테스트(`..._does_not_track_gradients`) 삭제, 세 테스트에 복붙된 층별 a·x+b 변환을 `_rescale_per_layer` 헬퍼로 통합, seam 테스트가 코드 상수만 대조하던 허용 오차를 SPEC 값(0.01)으로 고정, SPEC 모듈 6의 B0+ 문구가 "2클래스에도 쓴다"로 읽혀 ADR 0008("MNLI 본 실험에만")과 어긋나던 것을 정리. 반영하지 않음 — `isinstance` 분기·위임 단계 같은 취향 차원의 리팩터링, 러너 연동·가중치 감쇠(합의한 범위 밖, 위 후속 (a)(b)). **팀 확인 필요**: SPEC에 새로 쓴 "F1 벡터"·"문서별 층 가중치"는 GLOSSARY에 없는 표현이라 공유 파일 변경 때 정의할지 정할 것. verify ALL PASSED(288 passed, 16 skipped).
- 2026-11-23 · T-05b 교차검증(Antigravity, 커밋 `9de68e0`) 반영(고용민): 지적 4건을 실제 코드·실행과 대조해 모두 재현했다. 반영 — (1)[중요] 게이트 입력을 "23개 평균"에서 "마지막 층만"·"임베딩만"으로 바꿔도 83개 테스트가 모두 통과하던 것을 재현(소스 수정 없이 메서드 바꿔치기로 주입). forward 테스트가 기대 가중치를 검증 대상 함수에서 받아오는 동어반복이 원인이었다. 슬롯 23개 각각을 흔들면 문서별 가중치가 바뀌는지 보는 테스트와, SPEC 결정 제안의 식(층별 정규화 → 평균 → 게이트 → softmax)으로 가중치를 따로 계산해 대조하는 테스트를 추가했고, 두 결함이 이제 실패로 잡힌다(평균→합 결함도 정확값 테스트가 잡음). (2)[사소] seam ① 파라미터 수가 `run_experiment`를 거치지 않던 것 — SPEC 리터럴 표를 모듈 상수로 올리고, `test_identical_tuning_budget_across_fusion_methods`를 6개 결합 방식 전체로 넓혀 `ResultRecord.trainable_params`(F4 19,994, B0+ 20,075)를 확인한다(B0+ 은닉 26→27 주입 시 이 테스트 단독으로 실패). T-06a 이후 낡은 "러너가 생기면 다시 보자" 주석도 정리. (3)[사소] `F4Head.document_layer_weights`가 정규화된 입력을 가정하는데 공개돼 있어 층별 정규화를 우회할 수 있던 것 — `_document_layer_weights`로 비공개화, 공개 경로는 `layer_weights_for`·`extract_document_layer_weights`만 남김. (4)[정리] `TESTED_METHODS`를 6개 전체로 넓혀 B0+·F4 아핀 불변·로짓 모양 테스트의 중복을 없앰(F4 개별 테스트는 가중치 불변만 남김). 코드 수정 없이 PR·팀 리뷰로 넘김 — SPEC 결정 제안 3건(F4 형태, F4 softmax, 허용 오차 1%)과 GLOSSARY 미등록 표현 승인(차가원), F4 `gate.weight` 가중치 감쇠 제외 여부(위 후속 (b), 교차검증자도 F1 쪽으로 당겨짐을 실행 확인), 2클래스 B0+(`2: 25`)는 ADR 0008 근거로 유지(본 실험은 MNLI만이라 SPEC 범위 밖 사용은 없음), PR 생성 시 §0 규칙대로 T-05b와 T-05 묶음 제목을 완료로 갱신. verify ALL PASSED(315 passed, 16 skipped).
- 2026-11-24 · T-05b PR #18(고용민): §0 규칙에 따라 PR 안에서 T-05b를 `✅ 완료 (PR #18 병합 시)`로, T-05a가 이미 완료라 T-05 묶음 제목도 ✅로 갱신. 리뷰 중 막히면 👀/⛔로 되돌린다. PR #18은 SPEC 결정 제안 3건(F4 형태, F4 softmax, 허용 차이 1%)을 담은 공통 파일 PR이라 차가원 승인이 곧 결정 승인이다.
- 2026-11-24 · T-05b PR #18 리뷰 반영(고용민, 차가원 리뷰 + Antigravity 교차검증): (1) **가중치 감쇠 역할 기준 분리((a)안 채택)** — `fusion.no_weight_decay_parameters(module)`(`module.modules()` 순회, `FinetuneModel` 래퍼 호환)를 추가해 층 가중치를 만드는 파라미터(F2 `layer_logits`, F4 `gate.weight`·`gate.bias`)를 감쇠 `0.0`으로 제외하고 `training.make_optimizer`·`SPEC.md` §결정 필요 사항(L307)을 갱신, `test_runner.py`에 F2·F4·B0+·래퍼 감쇠 검증 추가. (2) **`origin/main`(PR #16, T-08b) 병합 및 `03-plan.md` 충돌 해결** — T-08b 완료 상태·로그와 양쪽 `마지막 verify`를 날짜순 보존. (3) **용어·문구·안전성·테스트 정리** — `SPEC.md` `"F1 벡터"`를 `"층별 정규화된 23개 층 벡터의 평균"`으로 풀어쓰기, `GLOSSARY.md` **층 가중치** 설명 보완, 차가원 동의 3건 확정 표기, `extract_document_layer_weights`의 `eval()` 전환·복원 및 `_FusionHead` 다형성 위임, `test_fusion.py`의 F4 학습 헬퍼를 공개 `parameters()` 기반으로 교체하고 중복 `B0_PLUS` 테스트 통합, `test_execution_seam.py`에서 `ResultRecord.trainable_params`로 B0+↔F4 1% 허용차 검증 추가. 후속(차가원 T-06d): `PredictionRecord.layer_weights` 기록 및 `FinetuneModel` 연동. verify ALL PASSED(311 passed, 16 skipped).
- 2026-11-24 · T-06b 구현(차가원): 새 모듈 `records.py`(결과 레코드 JSON 저장/로드: enum·tuple·float 키까지 그대로 복원해 다시 읽은 레코드가 원본과 `==`, `record_format` 1, 표준 JSON만 씀 — 발산 후보의 `inf` 튜닝 점수만 문자열 `"inf"`, 그 밖의 NaN·inf는 저장 시 오류, 모르는/빠진 키·파일 이름≠`run_id`는 로드 거부, 임시 파일 + `os.replace`로 원자적 저장, **이미 있는 레코드는 덮어쓰지 않음**(AGENTS §7, `overwrite`는 debug만), `find_lr_selection_record`로 재사용 실행의 선택 실행 레코드 찾기) + `matrix.py`(`experiment_matrix.yaml` → `RunConfig`: 실험 → 입력 설정 → 층 부분집합 → 결합 방식 → 시드 오름차순, 첫 시드 = 학습률 선택 실행, `expected_cache_meta`는 `week1_env.yaml`에서 자동 생성(11/18 PR #8 합의), **TBD(`seeds`·`fusion_methods`·`BEST_FUSION`·학습 예산 null)는 추측하지 않고 무엇이 비었는지 모아 거부**, 시드·예산 덮어쓰기는 debug만, `with_selected_lr`는 T-06c 대기열용) + `scripts/run_experiment.py` CLI(캐시 → 실행 → 레코드 파일, 기본 캐시 경로 `data/cache/<데이터셋>[_<입력 설정>]`, 덮어쓰기는 학습 전에 거부). `make_run_id`에 타임스탬프는 붙이지 않기로 함(설정+시드+실행 종류 = 레코드 파일 하나). 공통 파일(`contracts.py`·`configs/`·SPEC·`judgment.py`) 변경 없음. 테스트 37개 추가(`test_records.py`·`test_matrix.py`·`test_run_experiment_cli.py`). 판정 연동 테스트 1개는 `judgment.select_main_records`(PR #17) 전까지 skip — PR #17 `judgment.py`를 겹쳐 돌려 통과 확인. 셸에서 합성 캐시로 CLI 디버그 실행 → JSON 1개 생성·재로드 확인, main 실행은 학습 예산 TBD로 학습 전에 거부됨 확인. 현재 실제 행렬은 시드·예산이 TBD라 본 실험 전개는 11/30 고정 후 가능.
- 2026-11-24 · T-06b 코드 리뷰 반영(차가원, `/code-review` Standards·Spec 두 축): (1) **Spec: CLI 학습률 재사용 검증** — `--fixed-lr`·`--lr-selection-seed` 실행은 학습 전에 `--output-dir`의 선택 실행 레코드를 찾아 `selected_lr`과 대조하고, 없거나 다르면 거부(`records.find_selection_record_for`, SPEC §결정 필요 사항 "선택 실행 레코드를 찾아 참조"). (2) **Spec: CLI는 debug 전용** — `--run-kind main`은 T-06c 대기열(GPU 잠금, ADR 0009)로 가도록 거부. 본 실험 `expected_cache_meta` 자동 생성은 행렬 전개(`matrix.expand_experiment_matrix`)에 남음. (3) **Standards: 예산 조립 중복 제거** — `matrix.resolve_budget`(main은 덮어쓰기 거부)·`TrainingBudget.run_config_fields`(TBD 거부)를 CLI와 행렬 전개가 함께 씀. **팀 확인 필요(고용민, T-06b PR에서)**: (a) `week1_env.yaml` `training_budget.early_stopping`을 정수 patience(에폭 수)로 읽기로 함 — 다른 형태면 로더 수정 필요; (b) `records.same_setting`은 SPEC 설정 조합 키(데이터셋·데이터 양·결합 방식·층 부분집합·백본 조건·입력 설정)보다 엄격하게 `run_kind`·`track`·학습 예산·`expected_cache_meta`까지 같아야 같은 조합으로 봄(debug 선택 실행을 main 재사용에 쓰지 않도록); (c) `finetune_lr_candidates`는 아직 읽지 않음(T-06d). 남은 리팩터링 후보(병합 막지 않음): `week1_env.yaml` 파서 2개(`cache_build.load_env_config`·`matrix.load_training_budget`) 통합, `same_setting`·`find_*`를 직렬화 모듈 밖으로 분리, 레코드 존재 확인 함수 하나로 모으기. verify 268 passed, 12 skipped.
- 2026-11-24 · T-06b PR #19(차가원): PR #16 병합 후 `origin/main`을 병합(충돌 없음)하고 T-06b만 올림(T-06c·T-06d는 로컬에서 이어서 PR). §0 규칙에 따라 PR 안에서 T-06b를 `✅ 완료 (PR #19 병합 시)`로 갱신, T-06c~e가 남아 T-06 묶음 제목은 🔄 유지. PR #17 병합으로 `judgment.select_main_records`가 생겨 `tests/test_records.py`의 판정 연동 skip 조건을 제거(이제 실제로 실행됨). verify ALL PASSED(307 passed, 5 skipped). **PR 번호 주의**: 작성 시점 마지막 번호가 #18(T-05b)이라 #19로 적었으며, 다르면 PR에서 고친다.
- 2026-11-24 · T-06b PR #19 리뷰 반영(차가원): 레코드 로드 시 필수 enum의 null, 비표준 JSON 상수, 정수가 아닌 record_format, 객체가 아닌 config를 거부. 학습률 재사용 실행이 자기 시드를 참조하는 경우 학습 전에 거부. 회귀 테스트 7건 포함, verify 314 passed, 5 skipped.
- 2026-11-24 · T-06c 구현(차가원): 새 모듈 `run_queue.py` + `scripts/run_queue.py` CLI. (1) **본 실험 잠금**: OS 파일 잠금(POSIX `fcntl.flock`, Windows `msvcrt.locking`, 논블로킹, 새 의존성 없음)을 대기열 전체 동안 잡고, 이미 잡혀 있으면 보유자 pid·호스트·시작 시각과 함께 `MainQueueLockedError`로 거부(ADR 0009, US 47). 프로세스가 죽으면 OS가 풀어 남은 잠금 파일을 지울 일이 없음. 기본 경로 `results/main_queue.lock`(`--lock-file`). **한계**: 같은 잠금 파일을 보는 프로세스끼리만 막으므로 metis 컨테이너들은 같은 저장소 경로를 마운트해야 함. (2) **우선순위** MNLI > SST-2 > IMDB, 데이터셋 안에서는 행렬 순서 유지(US 51). (3) **학습률 재사용**: 설정마다 첫(가장 낮은) 시드가 선택 실행, 나머지는 이번 대기열 또는 디스크의 선택 실행 레코드로 `matrix.with_selected_lr` 적용. 순서가 어긋나거나 시드가 중복이면 실행 전에 거부. (4) **이어하기·즉시 멈춤(사용자 결정)**: 이미 있는 레코드는 건너뛰고 덮어쓰지 않음(AGENTS §7), 첫 실패에서 멈추고 완료 목록과 함께 `QueueRunError` — 원인을 고친 뒤 같은 명령을 다시 실행하면 이어서 진행. (5) **debug 대기열(`--debug`, 사용자 승인)**: 같은 순서·재사용 로직, 잠금은 잡지 않고 `run_kind=debug`로 기록, seeds·학습 예산·`--fusion-methods`(행렬 목록을 거르기만 함, TBD를 채우지 않음) 덮어쓰기는 debug만. SPEC 모듈 9 "디버깅 실행은 대기열 밖"을 "잠금 밖"으로 해석(US 49, 결과표 제외는 `run_kind`로 보장) — T-08a 스모크가 이 모드를 씀. (6) 재현 트랙 설정(T-06e)·main/debug 혼합·이미 `fixed_lr`가 채워진 설정은 거부. `--dry-run`은 순서와 건너뛸 항목만 출력. `runner.run_sequential_queue` 스텁 삭제 — **§1-2 표의 "9 대기열 실행기" 파일 칸(`runner.run_sequential_queue`)은 `run_queue.run_queue`로 사람이 고칠 것**. 공통 파일 변경 없음. 테스트 `tests/test_run_queue.py` 21개(별도 프로세스가 잠금을 잡은 상태에서 main 대기열 거부 = 완료 조건, 실제 `run_experiment` 학습률 재사용, CLI debug 실행·재실행 건너뛰기, dry-run, main TBD 거부 시 잠금 파일 미생성). 결함 주입 4종(잠금 무시·재사용 제거·우선순위 제거·건너뛰기 제거) 모두 테스트가 잡음(우선순위 제거는 처음에 못 잡아 테스트 추가). 셸 확인: 실제 `configs/`로 main 실행 → TBD 목록과 함께 거부, debug dry-run에서 IMDB를 먼저 줘도 MNLI가 먼저, 두 프로세스 동시 잠금 → 두 번째 거부·보유자 종료 후 획득. **알아 둘 것**: debug로도 결합 방식이 TBD인 실험(`mnli_frozen_small`·`sst2_frozen_full`·`mnli_finetune_full`의 BEST_FUSION)은 `--experiments`로 빼야 전개됨, F4·B0+는 T-05b 전까지 실행 시 오류로 대기열이 멈추므로 `--fusion-methods`로 뺄 것.
- 2026-11-24 · T-06d 구현(차가원, 담당 차가원·고용민): 파인튜닝 경로. 캐시 벡터 대신 토큰 ID를 넘기고 ModernBERT + 결합 헤드를 함께 학습. (1) 새 모듈 `finetune.py`: `TokenRowView`(배치마다 패딩, `RowView`와 같은 인터페이스) + `FinetuneModel`(백본 → 23개 은닉 상태를 추출기와 같은 `masked_token_mean`으로 풀링 → 헤드). `training.train_head`·`evaluate`는 그대로 재사용. (2) `providers.BackboneProvider` 스텁 교체: 토큰화한 분할(`TokenBatch`: doc_id·라벨·장르·토큰 길이·시야 비율), 사전학습 가중치의 깨끗한 복사본(`fresh_model`, 학습률 후보·시드마다 새로 시작), `from_env`(고정 transformers 버전·어텐션 구현 확인, debug `--limit` 표본), 메타데이터 dtype `float32`, 표본 공급원은 main 거부. (3) `run_experiment`가 백본 조건에 따라 뷰와 모델만 갈라 같은 레코드 형식을 냄. 파인튜닝은 MNLI·전체 데이터만(ADR 0004), `trainable_params`는 백본 포함 실제 학습 수(보고 범위는 미정, T-09a), F2 층 가중치는 헤드에서. (4) `matrix.TrainingBudget.finetune_lr_candidates`: 파인튜닝 설정은 이 값을 `lr_candidates`로(미정이면 파인튜닝 실험만 거부), 기대 메타데이터 dtype `float32`. `run_experiment.py`에 `--backbone finetune`·`--limit`·`--finetune-lr-candidates`, 대기열 공급원 팩토리가 파인튜닝이면 `BackboneProvider`. **SPEC §결정 필요 사항(공통 파일, 고용민 PR 리뷰 필요)**: 파인튜닝 학습 루프 구현 기본값 = 백본 고정과 같은 규칙(AdamW·감쇠 규칙, 학습률 하나, 스케줄러·warmup 없음, fp32), `trainable_params` 기록 방식. 테스트 `test_finetune.py` 19개(가짜 HF 백본·토크나이저: 백본 고정 레코드와 같은 필드·저장/재로드·`select_main_records` 통과 = 완료 조건의 형식 부분, 백본 가중치 변화·후보마다 사전학습에서 시작, 패딩 제외·특수 토큰 설정, 범위 밖 거부, CLI; 실제 ModernBERT CPU 1에폭 1개) + `test_matrix.py` 2개. 결함 주입 6종 모두 잡힘. 셸 확인: 로컬 MNLI 캐시 + 실제 ModernBERT로 `--limit 64` CPU debug 실행 → 레코드(파라미터 149,016,602, 예측 64, matched/mismatched, F2 가중치 23) 생성·재로드, main + `--limit`은 거부. **남은 완료 조건**: metis GPU 디버깅 실행(사용자 실행) — GPU 메모리·에폭당 시간 실측으로 T-08c·T-08f 시간 추정, bf16·warmup 필요 여부 판단.
- 2026-11-24 · T-06d GPU 디버깅·RNG 재현성 보강(차가원): metis RTX 5090 개인 개발 컨테이너에서 MNLI `--limit 2000`, 1 epoch, batch 32, seed 0, fp32로 B0·F2 파인튜닝 디버그 실행. B0는 lr 2e-5, 평가 오류율 0.4835, 학습 3.626초, 전체 19.976초, 파라미터 149,016,579; F2는 후보 1e-5/2e-5/5e-5 중 5e-5 선택, 평가 오류율 0.5425, 학습 3.274초, 전체 29.004초, 파라미터 149,016,602. 두 실행 모두 최대 GPU 메모리 14,270 MiB, `ResultRecord` 생성·재로드 성공(`run_kind=debug`, 본 판정 제외). 코드 리뷰에서 모델 초기화만 seed 범위에 있고 dropout 학습은 밖에 있는 재현성 결함을 발견. 실행 seam 회귀 테스트로 RED(같은 seed에서 학습 곡선 불일치) → 학습 전체를 격리된 CPU/CUDA RNG 범위에 넣어 GREEN. 관련 67 passed, 1 skipped; 전체 `python scripts/verify.py` 355 passed, 4 skipped, ALL PASSED. **후속**: RNG 수정 후 GPU B0/F2 재실행, 공유 GPU 잠금 경로는 팀 합의 전까지 본 실험 시작 금지.
- 2026-11-24 · T-06d RNG 수정 후 GPU 재실행(차가원, RNG 수정 커밋 기준): metis RTX 5090 개인 개발 컨테이너, 같은 조건(MNLI `--limit 2000`, 1 epoch, batch 32, seed 0, fp32), 출력 `results/debug_runs/rngfix_8ba1fe4/`(debug, git 제외). B0 lr 2e-5를 **같은 seed로 두 번** 실행: 평가 오류율 0.4835 / 0.4845, 학습 손실 1.0790965 / 1.0790999(소수 여섯째 자리부터 다름), 예측 32/2000건 다름, 학습 3.57초. F2 후보 1e-5/2e-5/5e-5 → 1e-5 선택, 평가 오류율 0.6345(수정 전 5e-5·0.5425 — 2,000건·1에폭에서는 학습률 선택이 흔들림). 최대 GPU 메모리 14,268 MiB(수정 전과 같음). **판단**: CPU 회귀 테스트는 학습 곡선이 비트 단위로 같으므로 RNG 범위 수정은 동작하고, GPU 차이는 seed가 아닌 CUDA 커널의 비결정적 부동소수점 합산(역전파 atomic 등)으로 보임. **팀 결정 필요(고용민)**: 본 실험 파인튜닝에서 GPU 비결정성을 허용할지(시드 표준편차에 포함), 아니면 `torch.use_deterministic_algorithms`·`CUBLAS_WORKSPACE_CONFIG`로 고정하고 속도 비용을 잴지.
- 2026-11-25 · T-06e 구현(차가원, 로컬 브랜치 `Gawoun-t06e`, T-06d `9d36df5` 위): 재현 트랙 실행기. 논문 값은 arXiv 2412.13663v2 부록 E.1·Table 6 PDF 원문에서 확인(ModernBERT-base: MNLI 학습률 5e-5·감쇠 5e-6·1 에폭, SST-2 8e-5·1e-5·2 에폭; Table 5 dev MNLI 89.1·SST-2 96.0). **계약 변경(공통 파일, 고용민 리뷰 필요)**: SPEC §데이터 형태 → `contracts.py` `FusionMethod.PAPER_HEAD`("paper_head", 재현 트랙 전용, B0와 구별)·`ReproductionSettings`(가중치 감쇠·warmup 비율·스케줄러)·`RunConfig.reproduction`·`MAIN_FUSION_METHODS`(본 실험 결합 방식 B0~B0+) → `test_contracts.py`. `list(FusionMethod)`로 결합 헤드를 돌던 `tests/test_fusion.py`(고용민 모듈 테스트)·`test_execution_seam.py`는 `MAIN_FUSION_METHODS`로 바꿈(주석의 원래 의도 B0~B0+ 그대로). `records.py`는 `reproduction`을 저장·복원하고 이 키가 없는 예전 레코드는 `None`으로 읽음. **SPEC §결정 필요 사항**: 논문 값과 출처 기록, 미정 값(배치·warmup·스케줄러·시드), 결정 제안(승인 대기) — 튜닝 분할 없이 학습셋 전체로 학습, Table 6 에폭 고정 후 **마지막 모델** 평가(논문 조기 종료 기준 미기재, dev로 에폭을 고르면 평가셋으로 모델 선택), AdamW·감쇠는 bias·LayerNorm 제외, 본 실험 대기열과 같은 GPU 잠금. **`configs/experiment_matrix.yaml`** `reproduction_track`에 논문 값과 미정 값(null) 추가. 새 모듈 `reproduction.py`(`expand_reproduction`: TBD 모아 거부, 덮어쓰기는 debug만 / `run_reproduction`: 고정 에폭·마지막 모델·`transformers.get_scheduler`·에폭별 dev 오차율은 학습 곡선에만·`train_seconds`는 학습 단계만), `finetune.PaperHeadModel`(HF 분류 모델을 토큰 뷰 인터페이스로), `BackboneProvider(model=None)`·`from_env(load_backbone=False)`(재현 트랙은 데이터만), `runner.prediction_records`·`genre_error_rates` 헬퍼로 레코드 조립 공유, `runner.run_reproduction_track` 스텁 삭제, `run_experiment`는 `paper_head` 거부, `run_queue.matching_record` 공개. CLI `scripts/run_reproduction.py`(main은 `main_queue_lock`, 결과는 `results/repro`, 같은 설정 레코드만 건너뜀, 첫 실패에서 멈춤, debug 전용 `--seeds`·`--batch-size`·`--warmup-ratio`·`--lr-scheduler`·`--limit`). 테스트 `test_reproduction.py` 24개(판정 제외·저장 왕복 = 완료 조건, 마지막 모델 보고, 학습셋 전체 사용, 같은 시드 같은 결과, 감쇠 그룹, warmup 후 감소, 실제 config TBD 거부, debug 전개 논문 값, 범위 밖 거부, 예전 레코드 읽기, CLI, 실제 ModernBERT 분류 모델 CPU 1개). 결함 주입 4종(최적 에폭 보고·10% 떼기·감쇠 전부·트랙 검사 제거) 모두 잡힘(최적 에폭은 처음에 못 잡아 "마지막이 최선이 아닌" 설정으로 테스트 보강). 셸 확인: 실제 configs main → TBD 목록과 함께 거부·잠금 파일 없음, 로컬 MNLI `--limit 64` + 실제 ModernBERT CPU debug → 레코드(재현 트랙·paper_head·5e-5·5e-6·1 에폭·파라미터 149,607,171·matched/mismatched) 생성, `select_main_records` 결과 0개. **남은 일**: metis GPU 짧은 실행(완료 조건), 미정 값 팀 결정(11/30 전, T-08c 선행), §1-2 표 "8 재현 트랙 실행기" 파일 칸(`reproduction.run_reproduction`, `scripts/run_reproduction.py`)은 사람이 고칠 것. push는 T-06d PR 병합 후.
- 2026-11-25 · T-06e `/code-review` 반영(차가원, Standards·Spec 두 축): (1) **SPEC**: "분류 헤드 = transformers 기본 헤드"는 논문에 명시가 없는 해석이라 결정 제안 (0)으로 옮김, **MNLI 공개 수치와 비교할 오차율(matched만 / matched·mismatched 합친 값)을 미정으로 추가**(논문은 MNLI 숫자 하나만 적음, 결과 보기 전에 정할 것). (2) `lr_scheduler`를 설정 검사 단계에서 transformers 스케줄러 이름과 대조 — 오타가 모델 생성·GPU 잠금 뒤에야 드러나던 것. (3) CLI가 결과를 논문 오차율 옆에 출력(MNLI는 matched도 함께, T-08c "공개 수치와 나란히" 준비; 레코드 계약은 그대로). (4) debug `--limit N` 결과는 `results/repro/limitN`에 저장 — 표본 크기가 `RunConfig`에 없어 다른 크기의 표본 레코드를 "같은 설정"으로 건너뛰던 것. (5) 정리: 미정 값 목록을 `ReproductionOverrides` 필드 하나로(CLI는 main에서 `--limit`만 따로 거부하고 나머지는 전개가 거부), "MNLI·SST-2만" 검사 하나로, `records.py` 상수 사용 통일, 본 실험 CLI(`run_queue.py --fusion-methods`·`run_experiment.py --fusion`) 선택지에서 `paper_head` 제외. 새 테스트 3개(스케줄러 오타 거부, 표본 크기별 폴더, 본 실험 CLI의 `paper_head` 거부) + CLI 출력 확인, 결함 주입 2종 잡힘. **사람 확인 필요**: (a) `GLOSSARY.md` "결합 방식" 정의에 `paper_head`(재현 트랙 전용 논문 분류 헤드)가 없어 SPEC과 어긋남 — GLOSSARY는 공통 파일이라 팀이 고칠 것. (b) 고용민 모듈 테스트 `tests/test_fusion.py`의 `TESTED_METHODS`를 `MAIN_FUSION_METHODS`로 바꿈(PR에서 알림). 보류: 다른 모듈의 비공개 함수(`_check_seeds`·`_check_labels`·`_resolve_device`·`_rng_devices`·`_sync`) 재사용을 공개 API로 정리하는 일과 학습 루프 공통화는 T-06d 병합 뒤 별도로.
- 2026-11-25 · T-06e GPU 디버깅(차가원, `Gawoun-t06e` `552ecc2` + 아래 수정): metis RTX 5090 개인 개발 컨테이너, `run_reproduction.py --debug --seeds 0 --limit 2000 --batch-size 32 --warmup-ratio 0.06 --lr-scheduler linear --device cuda`(배치·warmup·스케줄러는 논문에 없어 점검용으로 고른 값, 본 실험 값 아님). MNLI(학습률 5e-5·감쇠 5e-6·1 에폭): 평가 오차율 0.5635(matched 0.5531 / mismatched 0.5739), 학습 3.9초, 전체 19.4초, 최대 GPU 메모리 13,442 MiB, 학습 파라미터 149,607,171. SST-2(8e-5·1e-5·2 에폭, 평가 872건): 오차율 에폭 1 0.1135 → 에폭 2 0.1009(마지막 모델 평가), 학습 4.4초, 전체 13.5초, 최대 GPU 메모리 5,148 MiB. 두 레코드 모두 `track=reproduction`·`run_kind=debug`·`paper_head`·재현 설정이 기록되고 CLI가 논문 오차율(10.9% / 4.0%) 옆에 출력. 2,000건 표본이라 수치는 논문과 비교 대상이 아님. 체크포인트 로드 보고의 `classifier` MISSING(새 분류층)·`decoder.bias` UNEXPECTED(MLM 출력층)는 예상대로. **수정 2건**: (1) `records.save_result_record`가 `mkstemp` 기본 권한 0600을 그대로 남겨 컨테이너(root)에서 쓴 레코드를 호스트 계정이 못 읽던 문제(T-06b부터) — 저장 전 umask를 따르는 권한(보통 0644)으로 바꿈, 회귀 테스트 `test_saved_record_follows_the_umask`(RED 확인 후 GREEN), SST-2 레코드가 0644로 생성됨을 실제 확인. (2) 재현 CLI 기본 출력이 `.gitignore`에 없는 `results/repro`였음 → 이미 무시되는 `results/reproduction`으로 바꾸고, `results/debug_runs/`도 `.gitignore`에 추가. 앞서 `results/repro/limit2000`에 생긴 MNLI 디버그 레코드는 `results/debug_runs/repro/limit2000`으로 옮김(삭제 없음). **주의**: 컨테이너에서 root로 `git switch`하면 바뀐 파일이 root 소유가 되어 호스트 계정이 편집하지 못함 — 브랜치 전환은 호스트에서 하거나 전환 후 소유권을 되돌릴 것.
- 2026-11-25 · T-06e PR #22 리뷰 반영(차가원, 고용민 리뷰): (1) `expand_reproduction` main 전개에서 시드 수가 `num_runs`(3)와 다르면 거부(ADR 0004·US 45; debug는 그대로 허용), (2) `check_reproduction_config`가 재현 트랙에서 쓰지 않는 본 실험 필드(`layer_subset≠all`, `early_stopping_patience`, `lr_selection_seed`)를 거부해 레코드에 쓰지 않은 설정이 기록되지 않게 함, (3) `records.py` 권한 조정을 POSIX에서만 실행(Windows에는 umask가 없고 읽을 때 프로세스 전역 상태를 바꿈). 새 테스트 5개(RED 확인 후 GREEN: 시드 2·4개 거부, 3개 통과, 세 필드 거부). 후속(이번 PR 아님): `default_model_factory`의 모델 로딩을 `providers.py`로 옮길지는 비공개 헬퍼 정리 때 함께 검토.
