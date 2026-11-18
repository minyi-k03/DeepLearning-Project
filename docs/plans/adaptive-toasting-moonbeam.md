# progress.md를 03-plan.md로 합치기 — 실행 계획

## Context

지금은 Task 정의(`docs/03-plan.md`)와 진행 상태(`docs/progress.md`)가 두 파일에 나뉘어 있고, 하위 Task(T-01a…T-09c) 상태는
어디에도 없다(`progress.md`는 상위 T-01~T-09만 추적). 사용자가 **각 Task의 진행·완료 상태를 `03-plan.md` 한 곳에서만** 확인하길 원한다.

사용자 결정:
- 상태는 **각 Task 표에 "상태" 열을 추가**해 보여 준다.
- `progress.md`는 **삭제**하고 참조 링크를 정리한다.

이 변경은 작업 규칙을 바꾸므로 공통 파일(AGENTS.md, CLAUDE.md)을 건드린다 → 병합 전 차가원 리뷰 필요(AGENTS.md §4).

## 변경 사항

### 1. `docs/03-plan.md` (주 변경)
- **§4 모든 Task 표에 "상태" 열 추가** (ID 바로 뒤). 값: `대기` / `진행` / `리뷰 대기` / `완료` / `막힘`.
  변환은 스크립트로 일괄 처리: 헤더·구분선·각 행에 셀 하나 삽입하고, 변환 전 모든 행이 11셀인지 확인한 뒤 12셀인지 재확인.
- **초기 상태값** (구현 시 `git fetch`로 원격을 다시 확인한 뒤 기입):
  - T-01a `진행` (main 병합됨, 고용민 PC 확인 대기) · T-01b `진행` (CODEOWNERS 완료, main 보호·테스트 PR 대기)
  - T-01c `리뷰 대기` (YongMin 26bcd09, 미푸시, 차가원 리뷰 필요) · T-01d `대기`
  - T-02b `리뷰 대기` (origin/feat/T-02b-imdb-length: 최대 3,270 토큰, Lfull 잘림 0건) · T-02a·T-02c `대기`
  - 나머지 전부 `대기`
- **상위 Task 표제에 요약 상태 병기** (예: `### T-01 공통 기반 — 진행`): 옛 progress.md의 T-01~T-09 단위 상태를 대체.
- **§0 읽는 법**에 상태 값 정의 한 줄 추가 (`완료` = 완료 조건 충족 + 병합).
- **§8 진행 기록 신설**(문서 끝): `progress.md`의 세션 로그 3건을 원문 그대로 옮기고, "마지막 verify" 한 줄 추가
  (2026-11-18 통과, 12 passed·26 skipped).
- **규칙 문구 수정** (4행·11행·16행): "표는 사람만 고친다, 진행 상태는 progress.md에서만" →
  "상태 열과 §8 진행 기록만 에이전트가 갱신한다. 완료 조건·마감·담당 등 나머지는 사람만 고친다. Task 삭제 금지."

### 2. 참조 정리
| 파일 | 위치 | 변경 |
| --- | --- | --- |
| `AGENTS.md` §3 | 14·15·16행 | 세션 시작: `docs/03-plan.md`와 git log 읽기(중복 제거). 인계 전 갱신 대상 → 03-plan.md 상태 열·§8. "Task 삭제·완료 조건 변경 금지" 규칙의 대상 파일을 03-plan.md로 |
| `CLAUDE.md` | 7행 | `/clear` 전 갱신 대상 → `docs/03-plan.md` |
| `README.md` | 9·20행 | `progress.md` 링크·설명 제거, 03-plan.md가 계획과 진행을 함께 담는다고 수정 |
| `docs/progress.md` | — | `git rm` |

수정하지 않는 것: `docs/03-capstone-project.md`(가이드·템플릿), `docs/plans/spec-md-spec-md-composed-volcano.md`(과거 계획 기록),
`scripts/verify.py`·`tests/`(progress.md를 참조하지 않음을 grep으로 확인함).

### 3. 팀에 알릴 것
- `origin/feat/T-02b-imdb-length`(차가원)는 옛 `progress.md`를 수정해 두었다 → 병합 시 modify/delete 충돌.
  해결: 충돌 시 삭제를 받아들이고, T-02b 상태·로그가 03-plan.md에 이미 들어 있는지 확인.
- 작업 규칙(AGENTS.md §3)이 바뀐다는 점.
- 이 변경은 YongMin에 커밋하며 푸시·병합은 하지 않는다.

## 검증
- 표 검사: 모든 Task 행이 12셀이고 `상태` 값이 허용 목록 안인지 스크립트로 확인.
- `grep -rn "progress\.md"`로 남은 참조가 위 "수정하지 않는 것"의 과거 문서뿐인지 확인.
- `python scripts/verify.py` 전체 실행 (기대: 12 passed, 26 skipped, ALL PASSED — 문서 변경이라 결과 불변).
- `git status`로 progress.md 삭제와 수정 파일 4개(03-plan.md, AGENTS.md, CLAUDE.md, README.md) 확인, `git diff`로 03-plan.md의 Task 정의 열(완료 조건 등)이 바뀌지 않았는지 확인.
