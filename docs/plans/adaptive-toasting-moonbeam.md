# T-01c 계약 검토·합의 — 실행 계획

## Context

`docs/03-plan.md` T-01c: `contracts.py`의 `RunConfig`·`ResultRecord`·`CacheMetadata`·`DecisionRules`가
`docs/SPEC.md` §데이터 형태(계약)의 필드를 모두 담는지 대조하고 합의한다. 마감 11/17(화)이 지났고,
T-03a·T-05a·T-07b·T-09a가 모두 T-01c에 막혀 있다. 브랜치는 사용자 지시대로 현재 `YongMin`을 쓴다
(새 브랜치 없음, main 직접 커밋 아님).

## 대조 결과 (SPEC §데이터 형태 vs `src/layer_fusion/contracts.py`)

| 계약 | SPEC 필드 | 결과 |
| --- | --- | --- |
| `RunConfig` | 데이터셋·데이터 양·결합 방식·층 부분집합·백본 조건·입력 설정·시드·학습률 후보·에폭·조기 종료·실행 종류·트랙 | **모두 있음** (+ `batch_size`, `expected_cache_meta`는 SPEC 밖 추가분) |
| `ResultRecord` | 실행 설정·학습률 선택·오차율(MNLI matched/mismatched)·문서별 예측/정답·파라미터 수·학습 시간·학습 곡선·층 가중치(F2, F4)·소량 표본 ID·transformers 버전·어텐션·컴파일 | **모두 있음** (F2는 레코드 단위 `layer_weights`, F4는 `PredictionRecord.layer_weights`) |
| `CacheMetadata` | 모델·버전·길이(입력 설정, 상한)·자르기 방식 | **모두 있음** (+ 어텐션·컴파일·특수 토큰 포함·dtype 추가분) |
| `DecisionRules` | 판정 기준·최소 표본·시드 일관성 기준값 | **5개 규칙 필드 누락** → 아래 변경 대상 |

누락: `decision_rules.yaml`에는 있으나 `DecisionRules`에 대응 필드가 없는 항목
`hypotheses.H1.target_fusion_rule`, `hypotheses.H5a.target_fusion_rule`,
`criteria.core_comparisons_for_5_seeds`, `criteria.finetune_num_seeds`, `criteria.best_fusion_selection_rule`.
현재는 `extra` dict로만 들어가 타입이 없다 (`03-plan.md` §7에 이미 위험으로 기록됨). 사용자 결정: **T-01c에서 추가**.

## 변경 사항

TDD 순서(SPEC → contracts → 테스트 → 의존 모듈)를 따른다. 테스트를 먼저 쓰고 실패를 확인한 뒤 계약을 고친다.

1. **`tests/test_contracts.py` 신규 (먼저 작성, 실패 확인)**
   - SPEC 필드 목록을 테스트에 적고 `dataclasses.fields()`로 4개 계약이 모두 담는지 검사 (필드 이름 하나 = 테스트 하나가 아니라 계약당 1개 파라미터화 테스트).
   - `configs/decision_rules.yaml`의 `criteria`·`layer_weight_interpretation` 키와 가설별 `target_fusion_rule`이
     `DecisionRules` 필드로 대응되는지 검사 → YAML과 계약이 어긋나면 잡힌다.
   - `DecisionRules`에 새 필드를 넣어도 기존 호출(`extra` 기본값 등)이 깨지지 않는지 기본값 검사.
2. **`src/layer_fusion/contracts.py` (공통·고위험 파일)**: `DecisionRules`에 `extra` 앞으로 기본값이 있는 필드 추가.
   값은 지어내지 않고 전부 `None` 기본값 (AGENTS.md §2).
   - `h1_target_fusion_rule: str | None = None`
   - `h5a_target_fusion_rule: str | None = None`
   - `core_comparisons_for_5_seeds: list[str] | None = None`
   - `finetune_num_seeds: int | None = None`
   - `best_fusion_selection_rule: str | None = None`
   - 타입은 형태가 정해지기 전이라 가장 느슨하게 둔다. 11/30에 값이 정해질 때 T-07a/T-07b에서 좁힌다.
3. **`docs/progress.md`**: T-01 행 메모와 세션 로그에 T-01c 결과 기록 (상태·acceptance는 건드리지 않음).

수정하지 않는 것: `docs/SPEC.md`(§데이터 형태의 "판정 기준"이 이미 이 필드들을 포괄), `docs/03-plan.md`(승인 후 표는 사람만 수정),
`configs/decision_rules.yaml`(TBD 값 그대로), `judgment.py`(로더는 T-07b).

## 검토 후 팀에 넘길 메모 (코드 변경 없음)

- `ResultRecord.lr_tuning_scores: dict[float, float]` — JSON 저장 시 키가 문자열이 된다. T-06b(왕복 직렬화)에서 처리 필요.
- `RunConfig.early_stopping_patience: int | None` — SPEC에서 "조기 종료 기준"의 형태는 미정(`week1_env.yaml early_stopping: null`). int patience를 가정한 상태라 확정 시 재검토.
- `validate_metadata_match`는 `CacheMetadata` 전체 동등 비교라 SPEC의 4항목(모델·버전·길이·자르기)보다 엄격하다(어텐션·dtype·특수 토큰 포함 여부도 비교). ADR 0005 의도와 맞는지 차가원 확인 필요.

## 검증

- `python -m pytest tests/test_contracts.py` — 신규 테스트: 변경 전 실패, 변경 후 통과.
- `python scripts/verify.py` 전체 — 기대: 기존 5 passed + 신규 통과, skip 26개 그대로, ruff 통과.
- 계약 변경이므로 `/code-review`(skill `code-review`)로 Standards·Spec 리뷰 1회.
- `YongMin`에 커밋(Co-Authored-By 포함). `contracts.py`는 공통 파일이라 **병합 전 차가원 리뷰 필요** (CODEOWNERS).
