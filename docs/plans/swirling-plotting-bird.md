# T-07c 계획 — 가설별 판정·분석 (judgment seam 나머지)

## Context
T-07b가 판정 코어(규칙 로더, 시드 평균·표준편차, 짝지은 McNemar, H1, 탐색적 행)를 끝냈다. T-07c는 나머지 확증 가설(H3·H2·H4′·H5b·H5a)의 판정 평가기와
층 가중치 분석(무게중심·전역 비율·시드 일관성), 길이 구간 분석, F4 길이–가중치 상관, H4/H5c 탐색 행의 근거 자료를 채운다.
`docs/03-plan.md` T-07c의 완료 조건: 손 계산 예(22층에 가중치 전부 → 무게중심 22, 비율 1)와 일치, Seam ② 나머지 3개 un-skip + 추가 단위 테스트.
규칙 값(알파, 집계 방식, 임베딩 처리, 일관성 기준 등)은 11/30(T-07a)까지 TBD이므로 **None이면 ValueError("... is TBD")** — T-07b 관례 그대로, 값을 추정하지 않는다.

## 결정 (사용자 확인 대기 — 추천안으로 작성)
1. **H3/H2 검정**: 조건별 효과 = B0 오차율 − 대표 방식 오차율. 효과 차이 ≥ 두 설정 중 큰 시드 표준편차 AND 문서별 이득(+1/0/−1)의 조건 간 차이에 대한 정확 부호 검정(양측 이항, 시드별 후 `multi_seed_mcnemar_aggregation`으로 집계)이 유의하면 지지. 방향성: 효과가 줄어든(H3) / 더 큰(H2) 시드만 센다. SPEC에 "T-07c 임시 선택, T-07a에서 확정"으로 기록.
2. **대표 방식**: H2·H4′는 `rules.h1_target_fusion_rule` 재사용(새 규칙 필드 없음 → `contracts.py` 변경 없음). H3 후보는 파인튜닝 레코드 중 B0가 아닌 방식 하나(둘 이상이면 ValueError).
3. **길이 구간**: H5a/H5b는 IMDB 평가셋 전체로 판정 하나. 구간별 오차율·문서 수는 `report.length_bucket_analysis`, `below_min_samples`는 `min_bucket_samples`가 None이면 None. ≤512 구간은 판정 없이 점검(두 설정의 문서 ID·토큰 길이 동일 여부, 오차율 차이)만; 문서 집합이 다르면 ValueError.

## 구현 (src/layer_fusion/judgment.py — 공유 파일, 팀원 리뷰 필요)
- `compute_center_of_gravity(weights, include_embedding, *, layer_subset=ALL)`, `compute_global_layer_weight_ratio(weights, include_embedding_in_denom, *, layer_subset=ALL)`: `layers.slots_for_subset`으로 슬롯↔층 번호 매핑. 임베딩 제외 시 층 1~22 가중치를 재정규화. 포함 시 슬롯 0을 위치 0으로(규칙값이 정한 처리, 문서화). 길이/합 검증.
- 시드 일관성: F2 셀별로 시드 간 스피어만 순위 상관(쌍별 평균, scipy `spearmanr`; nan은 미달로 처리)과 무게중심 시드 표준편차를 `seed_rank_corr_min`·`cog_seed_std_max`와 비교 → 미달 시 `layer_weight_analysis[셀]["interpretation"] = Verdict.HELD`.
- `_judge_comparison` 일반화: 비교 중 달라질 수 있는 필드(`_COMPARISON_FREE_FIELDS`)를 인자로 받게 해 H5a(입력 설정 L512 vs Lfull)에 재사용. 하위 집합 평가(mismatched, 길이 구간)는 `dataclasses.replace`로 예측·오차율을 걸러낸 레코드 뷰를 만들어 넘긴다.
- 신규 평가기를 `_CONFIRMATORY_EVALUATORS`에 등록(키는 YAML 그대로 `H1`, `H3`, `H2`, `H4_prime`, `H5b`, `H5a`): 필요한 레코드가 없으면 None(판정표에서 빠짐).
- H5b: F2 판정(F1 보조). 긴 입력 이득의 차이 = [err(F2,L512) − err(F2,Lfull)] − [err(B0,L512) − err(B0,Lfull)], 같은 방식의 문서별 이득 짝지은 검정. `details`에 섞기 효과(F1−B0)·가중치 학습 효과(F2−F1)를 설정별 오차율 차이와 이득 차이 두 방식으로 분해(합이 F2−B0와 일치).
- H4/H5c(탐색): 판정은 EXPLORATORY 유지, 최상위 `mean_diff`·`mcnemar_p_value`는 None. `details`에 MNLI/SST-2 무게중심, SST-2 오차율 McNemar, 입력 설정별 무게중심·전역 비율, 전역/지역 ablation 표를 담는다. 의존하는 층 가중치가 시드 일관성 미달이면 `details["layer_weight_interpretation"] = HELD`(행 verdict는 EXPLORATORY 유지).
- F4 상관: Lfull F4의 `PredictionRecord.layer_weights`(23슬롯)로 문서별 전역 비율 → 토큰 길이와의 스피어만 상관, 전체·라벨별, 시드별 후 평균, `layer_weight_analysis`에 저장.
- 시드 일관성·상관 계산은 순수 함수로 분리해 합성 레코드로 CPU 테스트.

## 문서·상태
- `docs/SPEC.md` "구현 단계에서 정할 것"에 H3/H2 검정·임베딩 처리·≤512 점검·H4/H5c 보류 표기를 **결정 제안(T-07c, 승인 대기)** 로 추가 (공유 파일).
- `docs/03-plan.md`: T-07c 상태 컬럼과 §8 진행 기록만 갱신.

## 테스트 (tests/test_judgment_seam.py, 새 tests/test_judgment_analysis.py)
- un-skip 3개: 시드 일관성 HELD, 무게중심·전역 비율 손 계산(22층 전부 → 22.0, 1.0; 임베딩 포함/제외 각각), H5b 분해.
- 추가: 각 가설 지지/효과 없음/레코드 없음(판정표에서 제외), TBD 규칙 거부, debug·재현 제외 유지, 부분 집합(전역 8/지역 14) 무게중심, F4 라벨별 상관, ≤512 점검 오류, 탐색 행 최상위 None 유지, 기존 H1 테스트 회귀 없음.

## 검증
1. TDD: 테스트 먼저(red) → 구현(green), `pytest tests/test_judgment_seam.py tests/test_judgment_analysis.py tests/test_judgment_core.py tests/test_reporting.py -q`
2. `python scripts/verify.py` 전체 실행
3. 합성 레코드로 `judge_hypotheses` → `generate_report_artifacts` 왕복을 직접 돌려 보고 모듈이 새 details·분석 dict를 깨뜨리지 않는지 확인
4. 구현 후 `code-review` 스킬, `YongMin` 브랜치에 `feat(T-07c): ...` 커밋 (push/PR은 요청 시)

## 남는 위험
다중 비교 보정·시드 집계·임베딩 처리·일관성 기준값은 T-07a 전까지 TBD(코드는 ValueError). 공유 파일(judgment.py, SPEC.md) 변경이라 차가원 리뷰 필요.
