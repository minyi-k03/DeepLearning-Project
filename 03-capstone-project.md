# 캡스톤·일반 프로젝트 워크플로우 & 하네스 설정 가이드

> 해커톤은 `01-hackathon-24-48h.md`(24~48시간), `02-onsite-hackathon-6h.md`(6시간), 모델 학습·데이터 분석 중심 프로젝트는 `04-ml-data-project.md`를 씁니다.

---

## 0. 한눈에 보기

중장기 프로젝트·캡스톤은 **요구사항 검증, 아키텍처 설계, 테스트, 산출물 문서화**에 집중합니다.

```text
[1~3단계: 데스크톱 앱]  브레인스토밍 → 구체화·실현 가능성 → 요구사항 명세·SPEC.md
        ↓ 인계 (기획 문서 커밋 → 프로젝트 스킬 선정·설치)
[하네스 구축 Step 1~8]  SPEC → AGENTS/CLAUDE → 진단 → 실행 환경 → verify → 규칙·보안 → CI·Evals → 실기능 검증
        ↓
[4~8단계: AI 에이전트]  아키텍처·계획 → 기능별 개발(PR) → 리뷰·통합 테스트 → 배포·운영 → 문서화·발표
        ↓
[유지보수]  마일스톤마다 최신성 점검·구조 개선, 분기마다 하네스 덜어내기
```

**처음 적용할 때의 최소 구성:** 전체를 한 번에 만들지 않습니다.

1. `SPEC.md`에 핵심 기능 2~3개와 완료 조건을 씁니다.
2. 사람별 기능 담당과 공통 파일 담당을 정합니다.
3. 모든 팀원 컴퓨터에서 설치·실행·`verify`가 되는지 확인합니다.
4. 각자 작은 기능 하나를 기능 브랜치에서 구현하고 바로 합칩니다.
5. 반복해서 생긴 문제만 규칙·테스트·스킬로 보완합니다.

처음 필요한 문서는 `SPEC.md`, 짧은 `AGENTS.md`(+ 한 줄 `CLAUDE.md`), `README.md`, `docs/progress.md`뿐입니다.

---

## 1. 스킬

### 1-0. 설치 위치

| 위치 | 언제 | 무엇을 |
| --- | --- | --- |
| ① Claude·Gemini **데스크톱 앱** | 1회 등록 | 기획 1~3단계 스킬 |
| ② **개인 전역** `~/.claude/skills/` | 1회 | `find-skills` **하나만** |
| ③ **프로젝트** `.agents/skills/`(원본) → `.claude/skills/`(sync 사본) | `SPEC.md` 확정 후 | 4단계 이후 실행 스킬 |

- 전역에 `find-skills`만 두는 이유: Claude Code는 같은 이름이면 **개인 스킬을 우선**하므로, 전역의 옛 버전이 프로젝트 버전을 덮어씁니다.
- 데스크톱 앱에서 켠 스킬은 Claude Code의 전역 경로(`~/.claude/skills/synced/`)로 자동으로 내려옵니다. 팀원 각자 개인 설정(`~/.claude/settings.json`)에 `{ "syncClaudeAiSkills": false }`를 넣어 끕니다.

### 1-1. ① 데스크톱 앱 기획 스킬

| 단계 | 스킬 | 출처 | 비고 |
| --- | --- | --- | --- |
| 1 | `brainstorming` | obra/superpowers | 후보 도출만. 설계·구현 계획 금지 (§4-2) |
| 2·3 | `grill-with-docs` (+ `grilling`, `domain-modeling`) | mattpocock/skills | 구체화 + 용어집·ADR. 괄호 안 두 스킬을 내부에서 부르므로 함께 등록 |
| 2 | `research` | mattpocock/skills | 기술·API 실현 가능성 1차 자료 조사 |
| 3 | `to-spec` | mattpocock/skills | 이슈 트래커 대신 `SPEC.md` 형식으로 출력 |

```bash
# 빈 임시 폴더에서 스킬 폴더 받기 → ZIP
mkdir planning-skills && cd planning-skills
npx skills add mattpocock/skills -s grill-with-docs -s grilling -s domain-modeling -s research -s to-spec -a codex --copy -y
npx skills add obra/superpowers -s brainstorming -a codex --copy -y
cd .agents/skills && for d in */; do zip -r "${d%/}.zip" "$d"; done
```

- **Claude 데스크톱:** 설정 → Capabilities → "Create and edit files" 켜기 → Customize → Skills → `+` → Create skill → Upload a skill → ZIP 업로드 → 토글 켜기. (ZIP 안 폴더 이름 = 스킬 이름) Claude 데스크톱에서는 `/스킬명`으로 부르지 않고 "grill-with-docs 스킬로 이 아이디어를 검증해줘"처럼 스킬 이름을 넣어 요청합니다(설명에 맞는 요청이면 자동으로 쓰임).
- **Gemini 앱:** Settings → Skills → 폴더/ZIP 업로드. 쓸 때는 `@`를 입력해 스킬을 고르거나, 관련 요청이면 자동으로 쓰입니다. 웹·Mac 앱에서 등록하고, 학교(Workspace) 계정에서 안 보이면 개인 계정을 씁니다.
- 이 문서의 `/brainstorming` 같은 표기는 스킬 이름입니다. 데스크톱 앱에서는 위 방식(Gemini `@스킬명`, Claude는 이름을 넣어 요청)으로 부릅니다.

### 1-2. ② 개인 전역

```bash
node -v   # Node.js 22.20 이상
npx skills add vercel-labs/skills -s find-skills -g -a claude-code
```

### 1-3. ③ 프로젝트 스킬

**조건부** = `SPEC.md` 기술 스택이 맞을 때만.

| 단계 | 스킬 | 출처 | 하는 일 | 구분 |
| --- | --- | --- | --- | --- |
| 4 전 | `setup-matt-pocock-skills` | mattpocock/skills | 이슈 트래커·문서 위치 설정 (레포당 1회) | **필수** |
| 2 | `prototype` | mattpocock/skills | 위험한 핵심 로직을 본구현 전에 시험 | 선택 (코드 실행 → 에이전트에서) |
| 4 | Plan 모드 (`/plan`) | Claude Code 내장 (Antigravity는 `/plan`) | 수직 분할 Task를 `docs/03-plan.md`에 | 필수 · 설치 불필요 |
| 5 | `implement` | mattpocock/skills | Task 구현. 중간에 `tdd`, 마지막에 `code-review`를 부르고 커밋 (**커밋 순서는 §4-2**) | 필수 |
| 5 | `tdd` | mattpocock/skills | 테스트 우선 규칙. 내부에서 `codebase-design`을 부름 | 필수 |
| 5·6 | `verification-before-completion` | obra/superpowers | 완료 주장 전 검증 명령 실제 실행 | 필수 |
| 6 | `code-review` | mattpocock/skills | diff를 코딩 표준 + spec 두 축으로 검토 | 필수 |
| 6 | `improve-codebase-architecture` | mattpocock/skills | 얕은 모듈·강결합 찾기 (코드가 쌓인 뒤) | 필수 |
| 5·6 (의존) | `codebase-design`, `grilling`, `domain-modeling` | mattpocock/skills | 위 스킬들이 내부에서 부르는 기반 스킬 | 필수 |
| 6 | `diagnosing-bugs` | mattpocock/skills | 실패 재현부터 버그 원인 진단 | 선택 |
| 6 | `agent-browser` | vercel-labs/agent-browser | 실제 브라우저로 화면·흐름 확인 | 필수 |
| 7 | `finishing-a-development-branch` | obra/superpowers | 브랜치 마무리 (2번 "push + PR"만 사용) | 필수 |
| 8 | `pptx` | anthropics/skills | 발표 자료 | 필수 |
| 5·6·회고 | `pr`, `retro`, `writing-for-agents` | mattpocock/skills | PR 본문 형식 / 세션 회고 / `AGENTS.md` 작성 지침 | 선택 |
| Step 5 | `setup-pre-commit` | mattpocock/skills | Husky + lint-staged + typecheck·test 훅 | 조건부: Node |
| 5 | `vercel-react-best-practices` | vercel-labs/agent-skills | React·Next.js 성능 규칙 | 조건부 |
| 5 | `supabase-postgres-best-practices` | supabase/agent-skills | Postgres 스키마·인덱스·RLS | 조건부 |
| 5 | `design-taste-frontend` | Leonxlnx/taste-skill (폴더 `taste-skill`) | 회사·제품 소개 사이트 디자인 (앱 화면 자체 설계용 아님). 30~45분 소요, Stop hook 포함 | 조건부 |
| 7 | `deploy-to-vercel` | vercel-labs/agent-skills | Vercel 배포·프리뷰 | 조건부 |

**설치 절차**

1. **탐색:** **Claude Code에서** 역할별로 후보를 요청합니다(`find-skills`는 Claude Code 전역에만 설치). 표의 필수 스킬은 그대로 쓰고, 조건부·도메인 스킬만 새로 찾습니다.

```text
/find-skills SPEC.md를 읽고, 아래 역할마다 스킬 후보를 표로 정리해. 설치는 하지 마.
역할: 계획 검증 / 구현 / 도메인 도구 / 동작 확인 / 완료 확인 / 리뷰 / 버그 진단 / 발표
- 위 표에 이미 있는 역할은 표의 스킬을 적고, 표에 없는 도메인 도구(프레임워크·DB·배포 등)만 새로 찾아.
- 후보마다 출처 저장소, 설치 수, 고른 이유를 적어.
- 각 후보의 SKILL.md에서 다른 스킬을 부르는 부분을 찾아 하위 스킬도 같이 적어.
```
2. **검토:** 후보의 `SKILL.md`와 동봉 파일을 사람이 읽습니다(§3-8 보안 검토).
3. **설치:** 프로젝트 루트에서 `.agents/skills/`에 복사본으로만 설치합니다. **`-g`(전역)는 쓰지 않습니다.**

```bash
# 필수
npx skills add mattpocock/skills -s setup-matt-pocock-skills -s implement -s tdd \
  -s code-review -s improve-codebase-architecture \
  -s codebase-design -s grilling -s domain-modeling -a codex --copy -y
npx skills add obra/superpowers -s verification-before-completion -s finishing-a-development-branch -a codex --copy -y
npx skills add vercel-labs/agent-browser -s agent-browser -a codex --copy -y
npx skills add anthropics/skills -s pptx -a codex --copy -y
# 선택
npx skills add mattpocock/skills -s diagnosing-bugs -s prototype -a codex --copy -y
npx skills add mattpocock/skills -s pr -s retro -s writing-for-agents -a codex --copy -y
# 조건부
npx skills add vercel-labs/agent-skills -s vercel-react-best-practices -s deploy-to-vercel -a codex --copy -y
npx skills add supabase/agent-skills -s supabase-postgres-best-practices -a codex --copy -y
npx skills add Leonxlnx/taste-skill -s design-taste-frontend -a codex --copy -y
npx skills add mattpocock/skills -s setup-pre-commit -a codex --copy -y   # Node

ls .agents/skills                    # 실제 위치 확인
python scripts/check_skill_deps.py   # 빠진 하위 스킬이 없을 때까지 반복
python scripts/sync_skills.py        # .claude/skills/ 생성

# 팀원 각자: agent-browser CLI (최초 1회)
npm install -g agent-browser && agent-browser install
```

4. **기록:** `README.md` Skills 표에 역할 → 고른 스킬, 원본 저장소와 이유를, 설치 결정은 `docs/adr/`에 남기고 `skills-lock.json`을 커밋합니다. 에이전트별 인식을 확인합니다(§3-4).

**알아 둘 것**

- `-a codex`: Codex·Antigravity가 읽는 `.agents/skills/`에만 설치합니다(이 폴더를 원본으로 씀). `--copy`: 심볼릭 링크 대신 실제 파일로 복사합니다(Windows 대응). Claude Code용 `.claude/skills/`는 `sync_skills.py`로 만들어, 두 폴더가 같은지 `--check`로 검사할 수 있게 합니다.
- `setup-pre-commit`은 공식 README 목록에는 없고 저장소 `skills/misc/`에 있지만, `-s setup-pre-commit`으로 그대로 설치됩니다.
- `prototype`은 코드를 실행하므로 데스크톱 앱이 아니라 에이전트에서 씁니다.
- mattpocock 스킬을 Claude Code·Codex **플러그인**으로도 설치하면 같은 스킬이 두 번 보이므로, 플러그인은 설치하지 않습니다.
- `design-taste-frontend`는 한 번 실행에 30~45분과 수십만 토큰이 들고(브라우저로 화면을 검사하며 최대 3회 고침), 응답이 끝날 때마다 `node` 스크립트를 실행하는 Stop hook이 있습니다. 시간 여유가 있을 때만 씁니다.
- 스킬 업데이트 확인은 `npx skills@latest check`, 반영은 `npx skills@latest update`. 최신 CLI는 `skills-lock.json`의 `installMode: "copy"`를 보고 복사본으로 유지합니다(옛 CLI는 심볼릭 링크로 바꾸는 문제가 있었음). lock에 `installMode`가 없으면 처음 설치 명령을 한 번 다시 실행합니다. 반영 후 `python scripts/sync_skills.py`를 실행하고 `git diff`로 검토합니다.
- `handoff`(임시 폴더에 저장)와 `implement-spec`(티켓 파일 전제)은 이 가이드 방식과 맞지 않아 쓰지 않습니다.
- **하위 스킬:** 아래 스킬은 다른 스킬을 내부에서 부릅니다. 빠지면 그 단계가 조용히 생략됩니다.

| 스킬 | 함께 있어야 하는 스킬 |
| --- | --- |
| `grill-with-docs` | `grilling`, `domain-modeling` |
| `implement` | `tdd`(→ `codebase-design`), `code-review` |
| `improve-codebase-architecture` | `codebase-design`, `grilling`, `domain-modeling` |
| `retro` | `writing-for-agents` |
| `code-review`, `to-spec` | `/setup-matt-pocock-skills`로 만든 설정 |

- **Claude Code 내장 스킬과 이름이 겹칩니다.**
  - `/code-review`: 프로젝트의 mattpocock 스킬이 실행됩니다. 내장 리뷰는 `/review`로 씁니다.
  - `/verify`·`/run`: 앱을 실행해 확인하는 내장 스킬로, `agent-browser`와 역할이 같습니다. 팀에서 하나로 정합니다(기본 `agent-browser`). 셸 명령 `npm run verify`와는 다릅니다.

---

## 2. 디렉터리 구조

```text
project/
├── SPEC.md               # 요구사항의 유일한 기준 (완료 조건·API 계약)
├── GLOSSARY.md           # 도메인 용어집 (mattpocock 스킬이 루트에서 읽음)
├── AGENTS.md             # 세 에이전트 공통 작업 규칙
├── CLAUDE.md             # @AGENTS.md import + Claude Code 전용 지침
├── README.md             # 설치·실행·verify, 사용 에이전트, Skills 표
├── CHANGELOG.md
├── .env.example  .gitignore
├── .mcp.json             # Claude Code 팀 공유 MCP (비밀값은 환경 변수)
├── docs/
│   ├── 00-brainstorm.md
│   ├── 01-concept.md  01-feasibility.md
│   ├── 02-requirements.md
│   ├── 03-architecture.md  03-plan.md  progress.md
│   ├── 06-test-report.md
│   ├── 07-deploy.md  07-runbook.md
│   ├── adr/  meetings/  reviews/
│   └── agents/           # setup-matt-pocock-skills 설정
├── frontend/  backend/  database/  tests/   (또는 src/)
├── api/openapi.yaml      # API 계약 (공통 기반 담당)
├── scripts/
│   ├── verify.*          # 진입점은 OS 무관 명령 하나
│   ├── sync_skills.py
│   └── check_skill_deps.py   # 빠진 하위 스킬 검사
├── .agents/
│   ├── rules/            # Antigravity 전용 (frontmatter 필수)
│   └── skills/           # Skills 원본
├── .claude/
│   ├── settings.json     # 팀 공유 권한·Hooks·Plan 설정 (§3-8)
│   ├── hooks/guard.py    # 차단 hook
│   └── skills/           # Skills 사본 (직접 수정 금지)
├── .codex/config.toml    # Codex 팀 공유 (프로젝트 신뢰 시 적용)
└── .github/
    ├── CODEOWNERS
    └── workflows/ci.yml
```

---

## 3. 하네스 구축 (8단계)

템플릿 `harness-template-capstone`으로 시작하면 Step 2의 파일(`AGENTS.md`·`CLAUDE.md`·`sync_skills.py`·`check_skill_deps.py`)과 Step 6~7의 `settings.json`·`guard.py`·`CODEOWNERS`·`ci.yml`·`dependabot.yml`이 이미 들어 있습니다. 해당 Step은 **확인·채우기만** 합니다.

| Step | 작업 | 완료 기준 |
| --- | --- | --- |
| 1 | `SPEC.md` 작성·검수 (데스크톱 앱 → 프로젝트로 인계) | 모호한 부분·미확정 사항이 구분됨 |
| 2 | 담당 지정, `AGENTS.md` `<...>` 채우기, 역할별 스킬 설치·하위 스킬 검사, `/setup-matt-pocock-skills` | 담당 표 기록, 세 에이전트 로딩 확인 |
| 3 | 현재 하네스 진단 | 누락 사항·우선순위 확인 |
| 4 | 실행 환경 안정화 | 새로 clone → README대로 설치 → 같은 `verify` 통과 |
| 5 | 자동 검증 (`verify`, Node는 pre-commit) | `verify`가 실패 단계를 정확히 보고 |
| 6 | 규칙·보안 (자동 검사, `CODEOWNERS`·브랜치 보호, 권한, MCP) | 고위험 작업이 통제됨 |
| 7 | CI/CD, Evals | 반복 실행 가능한 검증 파이프라인 |
| 8 | 실제 기능으로 검증 | 계획 → 구현 → 검증 → 병합 → 수정 루프 작동 |

### 3-1. Step 1 — `SPEC.md`

필수 6요소: ① 개요·목표 ② 범위·기능 요구사항(Must/Should/Could, Non-goals, Mock 대상) ③ 완료 조건·예외 ④ 기술 스택·설계 ⑤ 비기능·운영 제약 ⑥ 미확정·위험.

요구사항과 완료 조건은 분리해서 **테스트 가능하게** 씁니다.

| ID | 요구사항 | 완료 조건 |
| --- | --- | --- |
| FR-AUTH-01 | 정상 인증 정보로 로그인할 수 있다 | 인증 성공, 로그인 상태 설정 |
| FR-AUTH-02 | 잘못된 인증 정보를 거부한다 | 인증 실패, 적절한 오류 응답 |
| FR-AUTH-03 | 보호된 API 접근을 통제한다 | 비인증 요청 거부 |
| FR-AUTH-04 | 오류를 사용자에게 안내한다 | UI에 오류 상태 표시 |

```markdown
# SPEC.md

## 1. Project Overview
- 프로젝트 목적: / 해결하려는 문제: / 대상 사용자: / 핵심 가치:

## 2. Goals, Non-goals, Mock
### Goals
### Non-goals
### Mock 대상 (해당 시)

## 3. User Scenarios
- 사용자 유형: / 주요 이용 흐름: / 예상되는 예외 상황:

## 4. Functional Requirements
### FR-001: [기능명]
- Description:
- Priority: Must / Should / Could
- Owner: (기능 담당자)
- Input: / Expected output: / Failure cases:
- Acceptance criteria:
- Verification method: (자동 테스트 / 화면·API 직접 확인)

## 5. Technology and Architecture
- Frontend: / Backend: / Database: / External APIs: / Authentication: / Deployment: / Confirmed constraints:

## 6. Data and API Contracts (변경은 공통 기반 담당자가, 코드보다 먼저)
- 주요 데이터 모델: / API 엔드포인트 및 입출력: / 오류 응답 규격: / 데이터 검증 규칙:

## 7. Non-functional Requirements
- Security: / Performance: / Reliability: / Accessibility: / Cost constraints:

## 8. Testing and Acceptance
- 핵심 테스트 시나리오: / 필수 완료 조건: / 성능 및 보안 검증: / 최종 검수 기준:

## 9. Risks and Dependencies
## 10. Open Questions
```

- `SPEC.md`가 생긴 뒤에는 **`SPEC.md`가 유일한 기준**이고 `docs/`의 원천 문서는 이력입니다. 요구사항 변경은 `SPEC.md`부터 합니다.

<details>
<summary>SPEC.md 작성·검수 프롬프트 (세 에이전트 공통)</summary>

```markdown
# SPEC.md 작성 및 요구사항 검수
현재 프로젝트 아이디어와 제공된 자료(docs/ 폴더의 기획·요구사항 문서 포함)를 분석하여 SPEC.md를 작성하라.
## 조사 및 작성 항목
1. 해결하려는 문제와 대상 사용자 2. 목표, Non-goals, Mock 대상 3. 사용자 시나리오
4. 기능 요구사항 및 우선순위 5. 기능별 성공 조건과 예외 6. 기술 스택과 시스템 제약
7. 데이터 모델과 API 계약 8. 보안·성능·신뢰성 9. 비용·외부 서비스·배포 제약 10. 미확정 사항과 위험
## 작성 원칙
- 확인된 사실과 추정을 분리한다. 확인되지 않은 기술 스택이나 API 동작을 지어내지 않는다.
- 중요한 기능마다 ID를 부여하고 완료 조건을 테스트 가능한 형태로 쓴다.
- 필수/선택, 구현 제외/Mock을 구분한다. 모순·불명확한 요구사항은 명시한다.
- docs/ 원천 문서와 다르면 차이를 목록으로 보고한다.
먼저 누락과 충돌을 검토하고 미확정 사항을 분리한 뒤 초안을 제시하라. 확인되지 않은 사항을 임의로 확정하지 마라.
```
</details>

### 3-2. Step 2 — 담당 지정 · `AGENTS.md` · `CLAUDE.md`

**설정 순서:**

1. `docs/03-plan.md`에 담당 표 (기능 담당 / 공통 기반 담당 / 병합 담당)
2. 프로젝트 스킬 설치 → `scripts/check_skill_deps.py` → `scripts/sync_skills.py` (§1-3, §3-5)
3. `AGENTS.md` 작성 (템플릿이면 `<...>` 자리 채우기)
4. `/setup-matt-pocock-skills` 실행 → **블록은 `AGENTS.md`에**, 이슈 트래커 GitHub Issues(권장) 또는 **Other**("Task는 `docs/03-plan.md`, spec은 `SPEC.md`". 로컬 마크다운은 `.scratch/`에 따로 파일을 만듦), 용어집은 기본값(루트 `GLOSSARY.md`), ADR `docs/adr/`
5. `CLAUDE.md` 확인 (첫 줄 `@AGENTS.md`, 스킬 블록이 들어가지 않음)
6. 사람이 `AGENTS.md`를 "이 줄이 없으면 에이전트가 틀리는가?" 기준으로 줄인 뒤 커밋
7. 로딩 확인 (§3-4)

- `setup-matt-pocock-skills`는 `CLAUDE.md`가 있으면 그쪽에 블록을 쓰려 합니다. 템플릿에는 `CLAUDE.md`가 이미 있으므로, 초안을 보여줄 때 "`AGENTS.md`에 써"라고 지정합니다.
- 용어집은 루트 `GLOSSARY.md`를 씁니다. `domain-modeling`·`tdd` 등 mattpocock 스킬이 이 이름을 루트에서 바로 읽으므로, 다른 위치로 바꾸면 스킬이 용어집을 찾지 못합니다.

**담당 표 예시 (2인):**

| 담당 | 책임 기능 | 작업 범위 |
| --- | --- | --- |
| 팀원 A | FR-001 검색·추천 결과 보기 | 검색 화면 → 추천 API → 결과 표시 |
| 팀원 B | FR-002 결과 저장·다시 보기 | 저장 버튼 → 저장 API·DB → 목록 화면 |
| 팀원 A | 공통 기반 | API 계약, 공통 타입, DB 마이그레이션, 의존성·배포 설정 |
| 팀원 B | 병합 | 충돌 해결 (일반 프로젝트는 PR 작성자가 승인 후 병합) |

**세 에이전트의 규칙 파일**

| 에이전트 | 읽는 파일 | 크기 제한 |
| --- | --- | --- |
| Antigravity | `AGENTS.md`, `GEMINI.md`, `.agents/rules/*.md` | 파일당 24KB |
| Claude Code | `CLAUDE.md` (`@AGENTS.md`로 import) | 200줄 미만 권장 |
| Codex | `AGENTS.override.md` → `AGENTS.md` | 합산 32KiB |

→ `AGENTS.md`는 약 100줄. `GEMINI.md`·`AGENTS.override.md`는 만들지 않습니다.

**`AGENTS.md` 템플릿** (`<...>`는 실제 값, 없으면 줄째 삭제)

```markdown
# AGENTS.md

## 1. Project Overview
- Read SPEC.md before implementing features.
- <One-line project purpose and constraints not discoverable from the repo>

## 2. Source of Truth
- SPEC.md defines functional requirements and acceptance criteria.
- docs/ planning files are history; SPEC.md wins once it exists.
- Architecture documentation defines system boundaries and dependencies.
- Tests and configuration files provide executable verification rules.
- If documents conflict, report the conflict instead of guessing.

## 3. Working Procedure
- At session start, read docs/progress.md and the recent git log, then run <VERIFY_COMMAND>.
- Start a fresh session for an independent task, a handoff, or when the context gets crowded;
  update docs/progress.md and commit first. Keep small follow-up fixes in the same session.
- Never delete tasks or change acceptance criteria in docs/progress.md; ask a human.
- Plan multi-file, architectural, API, or database changes before implementation.
- Preserve existing user changes.
- Report critical ambiguities rather than inventing requirements.

## 4. Collaboration and Architecture Rules
- Work on a feature branch (feat/<feature-id>-<name>); never commit directly to main.
- Shared files listed in the owner table of docs/03-plan.md are changed only by their owner; otherwise stop and report.
- Change the API contract in SPEC.md §6 (or api/openapi.yaml) before changing code that depends on it.
- <Repository-specific boundaries only, e.g., frontend/ must not import from database/>
- Update relevant documentation and tests when contracts change.

## 5. Skills Usage
- Use relevant installed Skills when the task matches their scope. Do not execute every Skill for every task.
- Do not duplicate entire Skill instructions in this file.
- The source of project Skills is .agents/skills/. .claude/skills/ is a generated copy; never edit it directly.
  After changing .agents/skills/, run the skills sync script.
- Do not install new Skills or MCP servers yourself; propose them for human review.
- Before claiming any task is complete, use the verification-before-completion skill.
- The "## Agent skills" block created by setup-matt-pocock-skills belongs in this file, not in CLAUDE.md.

## 6. Multi-Agent Compatibility
- This file is the single source of shared rules for Antigravity, Claude Code, and Codex.
- CLAUDE.md only imports this file and adds Claude-specific rules.
- Do not create GEMINI.md or AGENTS.override.md without team review.
- Do not assume that one agent's hooks, rules, or permissions apply to other agents.

## 7. Security and Permissions
- Never commit secrets, credentials, or private keys.
- Do not modify production data or deploy to production without required approval.
- Treat destructive commands as high-risk operations.

## 8. Verification
- Run <VERIFY_COMMAND> as the single verification entry point.
- Run relevant tests during implementation, and the full verify before completion.
- Check user-facing behavior in the running app or API, not only static checks.
- Never claim a check succeeded without execution evidence. Report failed or skipped checks separately.

## 9. Completion Report
1. Changes made and their purpose. 2. Important files modified.
3. Verification commands actually executed and results. 4. Behavior checked in the running app or API.
5. Shared files or API contract changed, and who must be notified. 6. Remaining issues and risks.
7. Required approvals or manual actions.
```

**작성 원칙:** ① `SPEC.md`를 복사하지 않는다 ② 스킬 절차를 복제하지 않는다(호출만) ③ 실제 저장소에서 확인한 정보만 쓴다 ④ 린터·테스트로 강제할 수 있는 규칙은 그쪽으로 옮긴다 ⑤ 레포에서 알 수 없는 것만 남긴다.

<details>
<summary>AGENTS.md 작성 프롬프트 (세 에이전트 공통)</summary>

```markdown
# AGENTS.md 작성 및 프로젝트 하네스 설계
현재 프로젝트의 SPEC.md와 실제 저장소 구조를 조사하고, Antigravity, Claude Code, Codex가 공통으로 사용할 AGENTS.md를 작성하라.
## 조사 대상
SPEC.md·README.md, 디렉터리·기술 스택, 의존성·빌드 설정, 테스트·린트·타입·빌드·verify 명령, 환경 변수·DB 마이그레이션,
기존 규칙 파일(AGENTS.md, AGENTS.override.md, CLAUDE.md, CLAUDE.local.md, GEMINI.md, .agents/rules/),
에이전트 설정(.claude/settings.json, .codex/config.toml, .mcp.json), docs/progress.md, docs/03-plan.md 담당 표,
설치된 Skills(.agents/skills/, .claude/skills/) 일치 여부, 레거시 .agents/workflows/, CI/CD와 Git 상태
## 엄격한 작성 규칙
- SPEC.md 요구사항과 Skills 상세 지침을 복사하지 않는다. 존재하지 않는 파일·명령·스택을 만들지 않는다.
- 미확인 사항은 미확정으로 표시한다. 약 100줄, 32KiB 이내.
- 에이전트 기본 행동과 레포에서 바로 보이는 정보, 린터·테스트로 강제되는 규칙은 적지 않는다.
- 초안의 각 규칙 옆에 "이 줄이 없으면 에이전트가 틀릴 수 있는 이유"를 한 줄로 표시한다(검토용).
- 다른 규칙 파일과 충돌하는 내용은 목록으로 보고한다.
먼저 조사 결과와 초안(필요 시 CLAUDE.md 포함)을 제시하라. 검토 전에는 기존 파일을 덮어쓰거나 수정하지 마라.
```
</details>

**`CLAUDE.md`** (파일에는 코드 블록 없이 저장)

- Claude Code는 `CLAUDE.md`가 없으면 `AGENTS.md`를 직접 읽지만, 팀원이 `CLAUDE.local.md`를 만들면 그 기능이 꺼지므로 import 방식을 씁니다.

```markdown
@AGENTS.md

## Claude Code
- Project Skills are copied into .claude/skills/ by the sync script; do not edit them here.
- Use Plan mode for multi-file or architectural changes and wait for approval.
- Personal preferences belong in CLAUDE.local.md (git-ignored), not in this file.
- Use /clear for an independent task or a handoff after updating docs/progress.md; keep small follow-up fixes in the same session.
- The bundled /verify skill does not replace the verify command in AGENTS.md; run that command.
```

**팀 공유 vs 개인 설정** (충돌하면 팀 규칙 우선)

| 에이전트 | 팀 공유 (커밋) | 개인 (커밋 금지) |
| --- | --- | --- |
| Antigravity | `AGENTS.md`, `.agents/rules/*.md`, `.agents/skills/`, `.agents/mcp_config.json`(비밀값이 없을 때) | `~/.gemini/` 아래 전역 규칙·Skills·MCP |
| Claude Code | `CLAUDE.md`, `.claude/settings.json`, `.claude/skills/`, `.mcp.json` | `CLAUDE.local.md`, `.claude/settings.local.json`, `~/.claude/` |
| Codex | `AGENTS.md`, `.codex/config.toml`, `.agents/skills/` | `~/.codex/`, `$HOME/.agents/skills/` |

### 3-3. Step 3 — 하네스 진단

진단 영역: 프로젝트 구조 / 실행 환경 / 환경 변수 / 테스트(`verify`) / DB / 규칙 자동화 / Skills / CI / 보안(권한·MCP) / 관측성(로그·`progress.md`) / 다중 에이전트 정합성 / 협업 구조.

<details>
<summary>하네스 진단 프롬프트 (세 에이전트 공통)</summary>

```markdown
# 현재 하네스 환경 진단
추가 Skills를 설치하거나 새 파일을 만들지 말고, 현재 하네스 환경을 먼저 진단하라.
## 조사 항목
1. 디렉터리 구조와 실제 기술 스택 2. 의존성 설치·개발 서버·빌드 가능 여부 3. 프런트·백엔드·DB 구성
4. 환경 변수와 비밀정보 관리 5. 테스트·린트·타입·빌드, verify 진입점과 실제 실행 결과
6. SPEC.md 요구사항 검증 방법 7. AGENTS.md와 Skills의 역할·중복 8. DB 초기화·마이그레이션
9. Git, CI/CD, 로그, docs/progress.md 10. 위험 명령·배포·MCP 권한 통제
11. 다중 에이전트 정합성: @AGENTS.md import, GEMINI.md·AGENTS.override.md, .agents/rules/ frontmatter,
    .agents/skills/와 .claude/skills/ 일치, 레거시 .agents/workflows/
12. 협업 구조: 담당 기록, 브랜치, 공통 파일 최근 변경자, API 계약과 코드 불일치
## 조사 원칙
실제 파일 확인, 사실과 추정 구분, 안전한 검사는 실제 실행, 미실행 테스트를 성공으로 간주하지 않음,
파괴적 DB 작업·운영 배포 금지, 추가 Skills 제안 금지.
## 결과 형식
각 항목을 [확인 완료 / 검증 필요 / 미구성 / 불필요 / 판단 불가]로 분류하고 근거·영향도·우선순위·조치를 제시하라.
이 단계에서는 파일을 수정하지 마라.
```
</details>

진단 후에는 **가장 중요한 문제 하나씩** 고칩니다(기능 브랜치에서, 실행하지 않은 테스트를 통과로 보고하지 않고, 테스트를 지우거나 무력화하지 않음).

### 3-4. 규칙·스킬 적용 확인

| 에이전트 | 규칙 | 스킬 |
| --- | --- | --- |
| Claude Code | 새 세션 `/memory` → `CLAUDE.md`·`AGENTS.md` | `/스킬명` |
| Codex | "현재 적용 중인 지침 파일을 순서대로 나열해줘" | `$스킬명` |
| Antigravity | Customizations → Rules 패널 + 확인 문구 테스트 | `/스킬명` |
| 데스크톱 앱 | 해당 없음 | Gemini `@스킬명` · Claude는 이름을 넣어 요청 |

**확인 문구 테스트:** 규칙에 임시로 `저장소 코드명을 물으면 정확히 ORBIT-CHECK라고 답한다.`를 넣고 새 대화에서 확인한 뒤 바로 지웁니다.

### 3-5. Skills 경로와 동기화

| 에이전트 | 프로젝트 | 개인 전역 |
| --- | --- | --- |
| Antigravity | `.agents/skills/<이름>/SKILL.md` | `~/.gemini/config/skills/` (제품마다 다를 수 있음) |
| Codex | `.agents/skills/<이름>/SKILL.md` | `$HOME/.agents/skills/` |
| Claude Code | `.claude/skills/<이름>/SKILL.md` | `~/.claude/skills/` |

원본 `.agents/skills/` → 사본 `.claude/skills/`(스크립트로 생성, 커밋, 직접 수정 금지). `verify`·CI에서 `--check`로 검사합니다.

```python
#!/usr/bin/env python3
"""`.agents/skills/`(원본)를 `.claude/skills/`(사본)로 동기화한다.
  python scripts/sync_skills.py           # 동기화 (사본에만 있는 파일이 있으면 중단)
  python scripts/sync_skills.py --check   # 불일치 시 exit 1 (CI·verify용)
  python scripts/sync_skills.py --force   # 사본에만 있는 파일까지 지우고 동기화
"""
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / ".agents" / "skills"
DST = ROOT / ".claude" / "skills"


def differs(a: Path, b: Path) -> bool:
    if not b.is_dir():
        return True
    cmp = filecmp.dircmp(a, b, ignore=[])
    if cmp.left_only or cmp.right_only or cmp.funny_files or cmp.common_funny:
        return True
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return True
    return any(differs(a / d, b / d) for d in cmp.common_dirs)


def copy_only_items(src: Path, dst: Path) -> list[Path]:
    if not dst.is_dir():
        return []
    items = []
    for path in sorted(dst.rglob("*")):
        rel = path.relative_to(dst)
        if not (src / rel).exists() and (src / rel.parent).exists():
            items.append(rel)
    return items


def main() -> None:
    if not SRC.is_dir():
        sys.exit(f"원본 경로가 없습니다: {SRC}")
    if "--check" in sys.argv:
        if differs(SRC, DST):
            sys.exit("Skills 불일치: python scripts/sync_skills.py 실행 후 커밋하세요.")
        print("Skills 동기화 상태 정상")
        return
    extras = copy_only_items(SRC, DST)
    if extras and "--force" not in sys.argv:
        print("사본(.claude/skills/)에만 있는 항목이 있어 동기화를 중단합니다:")
        for rel in extras[:30]:
            print(f"  - {rel}")
        if len(extras) > 30:
            print(f"  ... 외 {len(extras) - 30}개")
        sys.exit("필요한 스킬은 .agents/skills/로 옮긴 뒤 다시 실행하세요. 지워도 되면 --force로 실행하세요.")
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"{SRC.relative_to(ROOT)} -> {DST.relative_to(ROOT)} 동기화 완료")


if __name__ == "__main__":
    main()
```

**Antigravity `.agents/rules/`** (Antigravity 전용 규칙만):

- YAML frontmatter + `trigger`(`always_on` / `model_decision` / `glob` / `manual`)가 없거나 틀리면 **경고 없이 무시**됩니다.
- `model_decision`은 `description`, `glob`은 `globs`가 필수. 폴더 바로 아래 파일만 읽습니다.
- 예전 `.agents/workflows/`는 지원 중단(deprecated) 상태입니다. 지금은 동작하지만 새로 만들지 않고, Antigravity App 2.12.0 이상에서 `/migrate-workflows`로 스킬로 바꾼 뒤 sync합니다(CLI·IDE에는 이 명령이 없어 직접 옮김).

```markdown
---
trigger: glob
globs: "*.sql, migrations/**"
description: "DB 마이그레이션 작성·검토 시 적용하는 규칙"
---
# Database Migration Rules
1. 한 번의 배포에서 컬럼을 삭제하거나 이름을 바꾸지 않는다.
```

**`scripts/check_skill_deps.py`** (템플릿에 있음): 설치된 스킬의 `SKILL.md`에서 다른 스킬을 부르는 부분을 찾아 빠진 하위 스킬을 출력하고, 하나라도 빠지면 exit 1로 끝납니다. 스킬을 설치·삭제한 뒤 sync 전에 실행합니다.

### 3-6. Step 4 — 재현 가능한 실행 환경

목표는 **같은 도구가 아니라 같은 실행 결과**입니다.

| 맞춰야 하는 것 | 달라도 되는 것 |
| --- | --- |
| 런타임 버전, 패키지 관리자, 잠금 파일 | 편집기, OS |
| 환경 변수 이름과 의미 | 개인 API 키 |
| DB 스키마·마이그레이션·샘플 데이터 | 개인 개발 DB 주소 |
| 실행·검증 명령, API 계약 | 사용하는 AI 에이전트·요금제 |

- 런타임 고정(`.nvmrc`, `.python-version`, `engines`), 잠금 파일 커밋
- `README.md`에 설치·실행·`verify`, 사용 에이전트, Skills 표
- `.env.example`(변수 이름만), seed·초기화 절차
- Docker·Dev Container는 환경 차이가 반복될 때만 도입

```gitignore
.env
.env.*
!.env.example
CLAUDE.local.md
.claude/settings.local.json
.superpowers/
```

### 3-7. Step 5 — 단일 `verify`

세 에이전트와 CI가 **같은 명령 하나**를 실행합니다.

```text
verify
├── lint
├── typecheck
├── unit tests
├── integration tests
├── build
└── harness checks (sync_skills --check, AGENTS.md 크기 ≤ 32KiB, .agents/rules frontmatter)
```

| 스택 | 진입점 |
| --- | --- |
| Node.js | `npm run verify` |
| Python | `python scripts/verify.py` |
| 혼합 | `just verify` / `make verify` |

- 실패하면 0이 아닌 종료 코드와 실패 단계를 보여 주고, 대화형 입력이 없어야 합니다.
- `SPEC.md` 요구사항이 실제 테스트로 판별되어야 합니다.
- Node 스택은 pre-commit 훅(Husky + lint-staged)에서 `verify` 또는 그 일부를 호출합니다.

### 3-8. Step 6 — 규칙 자동화 · 보안 · MCP

**문서 규칙 → 자동 강제**

| 규칙 | 자동화 |
| --- | --- |
| 잘못된 데이터 형식 금지 | 타입 검사, Zod·Pydantic 스키마 |
| 프런트가 DB에 직접 접근 금지 | 모듈 경계 린터, 코드 리뷰 |
| API 계약 유지 | `openapi.yaml` 스키마 검증, 통합 테스트 |
| 공통 파일은 담당자만 | `CODEOWNERS` + 브랜치 보호 |
| `main` 직접 커밋 금지 | 브랜치 보호(PR 필수, CI 통과 필수) |
| 비밀정보 금지 | `.gitignore`, 시크릿 스캐닝 |
| `.claude/skills/` 직접 수정 금지 | `sync_skills.py --check` |

```text
# .github/CODEOWNERS
/SPEC.md              @팀원A-깃허브아이디
/api/openapi.yaml     @팀원A-깃허브아이디
/database/migrations/ @팀원A-깃허브아이디
package-lock.json     @팀원A-깃허브아이디
/.github/             @팀원A-깃허브아이디
```

**에이전트별 권한**

| 에이전트 | 팀 공유 | 개인 | 주요 항목 |
| --- | --- | --- | --- |
| Antigravity | 워크스페이스 설정 | 에이전트 설정 화면 | 터미널 실행·리뷰 정책, Permissions (권장값을 README에) |
| Claude Code | `.claude/settings.json` | `.claude/settings.local.json` | permissions 허용·거부, Hooks |
| Codex | `.codex/config.toml` | `~/.codex/config.toml` | `sandbox_mode`, `approval_policy` (프로젝트 신뢰 필요) |

```toml
# .codex/config.toml
sandbox_mode = "workspace-write"
approval_policy = "on-request"
# danger-full-access + never 조합 금지
```

| 작업 | 정책 |
| --- | --- |
| 코드 읽기·수정, 테스트·빌드 | 필요한 범위 허용 |
| `.env` 등 비밀정보 | 노출 방지 |
| 개발 DB 초기화 | 데이터 손실 확인 후 |
| 운영 DB 변경, 운영 배포, 자원 삭제 | 명시적 승인 없이는 금지 |
| 외부 유료 서비스, MCP 쓰기 작업 | 사전 승인 기준 |

**차단 장치 (Claude Code)** — 어기면 피해가 큰 규칙만 실제로 막습니다.

| 막을 것 | 장치 |
| --- | --- |
| `.env` 읽기·수정 (`.env.example`은 허용) | `deny` 규칙 |
| `.claude/skills/` 직접 수정 | `deny` 규칙 |
| main에서 commit·push·merge·rebase, 한 명령 안에서 main 전환 후 같은 작업, main으로 push, 강제 push, 셸로 `.env` 접근 | `PreToolUse` hook `guard.py` |

**`.claude/settings.json`** (엄격한 JSON: 주석·끝 쉼표 금지)

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "deny": [
      "Read(.env)",
      "Read(.env.*)",
      "Read(*.env)",
      "Read(!.env.example)",
      "Edit(/.claude/skills/**)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|PowerShell",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -c '' >/dev/null 2>&1 && P=python3 || P=python; \"$P\" \"${CLAUDE_PROJECT_DIR}/.claude/hooks/guard.py\""
          }
        ]
      }
    ]
  },
  "plansDirectory": "./docs/plans",
  "showClearContextOnPlanAccept": true
}
```

- `plansDirectory`: Plan 모드 계획을 `docs/plans/`에 남깁니다(리뷰·보고서 근거).
- `showClearContextOnPlanAccept`: Plan 승인 때 계획 컨텍스트를 비우는 선택지를 보여 줍니다.

**`.claude/hooks/guard.py`**

```python
#!/usr/bin/env python3
"""Claude Code PreToolUse(Bash|PowerShell) 가드.

막는 것: main/master에서 commit·push·merge·rebase·cherry-pick, 한 명령 안에서 main/master로 전환한 뒤의 같은 작업,
main/master로 push(HEAD:main, refs/heads/main, +main 포함), main/master 강제 이동(branch -f, checkout -B, switch -C),
강제 push(--force-with-lease 포함 --force*, -f 묶음, +refspec),
.env 계열 파일 접근(.env.example 제외).
실수 방지용입니다. 문자열 쪼개기·스크립트 안 파일 열기 같은 우회는 못 막으므로,
main 보호는 GitHub 브랜치 보호가, 파일 보호는 샌드박스가 맡습니다.
차단 시 이유를 stderr에 쓰고 exit 2 → Claude Code가 명령을 막고 이유를 Claude에게 보여 준다.
"""
import json
import re
import subprocess
import sys

try:
    data = json.load(sys.stdin)
except ValueError:
    sys.exit(0)  # 검사할 명령이 없음
cmd = (data.get("tool_input") or {}).get("command", "") or ""

# git 하위 명령 위치: git [-C dir] [-c k=v] [--옵션] <하위 명령>
GIT = r"\bgit(?:\s+-[cC]\s+\S+|\s+--?[\w-]+(?:=\S+)?)*\s+"
BRANCH = r"['\"]?(main|master)(?=$|[\s;&|'\")`])"  # 따옴표로 감싼 "main"도 포함
ARGS = r"[^;&|\n]*"  # 같은 명령 안의 인자 (다음 명령·줄로 넘어가지 않음)


def block(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(2)


def current_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=data.get("cwd") or None, timeout=5, check=False,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


if re.search(GIT + r"push\b" + ARGS + r"(\s--force\S*|\s-[a-zA-Z]*f[a-zA-Z]*\b|\s\+\S)", cmd):
    block("강제 push는 금지입니다. 필요하면 사람에게 요청하세요.")

if re.search(GIT + r"push\b" + ARGS + r"(?:[\s:+]|refs/heads/)" + BRANCH, cmd):
    block("main으로 직접 push 금지: 기능 브랜치를 push하고 PR(또는 병합 담당자)로 병합하세요.")

# main을 강제로 옮기는 명령: branch -f main, checkout -B main, switch -C main (main이 대상일 때만)
FORCE_MOVE = (r"(?:branch(?=" + ARGS + r"\s(?:-[a-zA-Z]*f[a-zA-Z]*\b|--force\b))(?:\s+-\S+)*"
              r"|checkout(?:\s+-\S+)*?\s+-B|switch(?:\s+-\S+)*?\s+-C)\s+")
if re.search(GIT + FORCE_MOVE + BRANCH, cmd):
    block("main 브랜치를 강제로 옮기는 명령은 금지입니다.")

WRITE = GIT + r"(commit|push|merge|rebase|cherry-pick)\b"
if re.search(WRITE, cmd) and current_branch() in ("main", "master"):
    block("main 브랜치에서 commit/push/merge 금지: feat/<기능ID>-<이름> 브랜치로 전환하세요.")

# 새 브랜치 생성(-b/-c 등)의 시작점이나 파일 복원(checkout main -- 파일)은 main 전환이 아니므로 제외
NOT_SWITCH = r"(?!" + ARGS + r"\s(?:-[bBcC]\b|--create\b|--force-create\b|--orphan\b|--(?=\s|$)))"
to_main = re.search(GIT + r"(checkout|switch)\b" + NOT_SWITCH + ARGS + r"\s" + BRANCH, cmd)
if to_main and re.search(WRITE, cmd[to_main.end():]):
    block("main으로 전환한 뒤 commit/push/merge하는 명령은 금지입니다. 기능 브랜치에서 작업하세요.")

# 커밋 메시지(-m "...")에 적힌 .env는 파일 접근이 아니므로 빼고 검사
scan = re.sub(r"""\s-m\s*(?:"[^"]*"|'[^']*')""", " ", cmd)
if re.search(r"(?<![\w.-])(?:[\w-]+\.env|\.env(?:\.(?!example(?=$|[\s'\";|&>)},`]))[\w.-]+)?)(?=$|[\s'\";|&>)},`])", scan) \
        or re.search(r"(?<![\w.-])\.env?[*?\[]", scan):
    block(".env 파일 접근 금지: 변수 이름은 .env.example을 참고하세요.")

sys.exit(0)
```

**동작 확인 (팀원 모두 1회)**

```bash
# macOS·Linux·Git Bash
echo '{"tool_input":{"command":"cat .env"}}' | python3 .claude/hooks/guard.py; echo $?                  # 2
echo '{"tool_input":{"command":"cat .env.example"}}' | python3 .claude/hooks/guard.py; echo $?          # 0
echo '{"tool_input":{"command":"git push origin HEAD:main"}}' | python3 .claude/hooks/guard.py; echo $? # 2
```

```powershell
# Windows PowerShell
'{"tool_input":{"command":"cat .env"}}' | python .claude/hooks/guard.py; $LASTEXITCODE                  # 2
'{"tool_input":{"command":"cat .env.example"}}' | python .claude/hooks/guard.py; $LASTEXITCODE          # 0
'{"tool_input":{"command":"git push origin HEAD:main"}}' | python .claude/hooks/guard.py; $LASTEXITCODE # 2
```

- 확인 후 Claude Code 새 세션에서 `/status`로 설정 로딩을 보고, 프로젝트 폴더를 신뢰 처리합니다.
- `Read(.env)` 같은 규칙과 hook은 Claude Code를 실행한 폴더를 기준으로 동작합니다. **항상 프로젝트 루트에서 Claude Code를 시작합니다.**
- **hook 명령:** `python3`가 있으면 그것을, 없으면 `python`을 씁니다(macOS·Linux·Windows 공통). matcher `Bash|PowerShell`로 Windows의 PowerShell 도구도 검사합니다.
- **Windows:** Git Bash를 설치합니다. Git Bash가 없으면 hook이 PowerShell로 실행되어 위 명령이 실패하고, PreToolUse hook 실패는 명령을 막을 수 있습니다.
- guard는 **실수 방지용**입니다. 문자열을 쪼갠 명령이나 스크립트 안에서 파일을 여는 것은 막지 못하므로, main은 GitHub 브랜치 보호로 지킵니다.
- 이 장치는 Claude Code에만 적용됩니다. `main` 병합은 PR로만 합니다(브랜치 보호).
- 위 확인 명령을 `scripts/`에 넣어 `verify`의 harness checks에 포함해도 좋습니다.

**외부 Skills 보안 검토**

1. 출처: 공식 조직·유지보수 이력이 있는 저장소만
2. 본문 검토: `SKILL.md`와 `scripts/`·`references/` 전부. 위험 신호 = 외부 URL 전송, `.env`·`~/.ssh` 읽기, 권한 변경, 추가 설치 지시, 난독화 문자열, `SKILL.md` 맨 위 frontmatter의 `hooks:`(스킬이 쓰일 때 명령을 자동 실행. 예: `design-taste-frontend`는 응답이 끝날 때마다 `node`로 `gate.mjs`를 실행하는 Stop hook을 가짐)
3. 버전 고정: `.agents/skills/`와 `skills-lock.json`을 커밋, 업데이트는 diff 검토 후
4. 권한 최소화: 처음 실행은 승인 모드(Codex `on-request`, Claude Code 기본 권한 확인)

**MCP 서버**

- 필요한 서버만 연결합니다. 브라우저 검증은 하나로 통일합니다(Antigravity 내장 브라우저 / Claude Code·Codex는 Playwright MCP 또는 `agent-browser` 중 하나).
- 토큰은 환경 변수로 참조합니다. DB MCP는 개발 DB + 읽기 전용 계정으로 연결합니다.

| 에이전트 | 설정 위치 | 팀 공유 |
| --- | --- | --- |
| Claude Code | 루트 `.mcp.json` | 커밋 (비밀값은 환경 변수) |
| Codex | `.codex/config.toml`의 `[mcp_servers.<이름>]` | 커밋 |
| Antigravity | 워크스페이스 `.agents/mcp_config.json` 또는 사용자 전역 `~/.gemini/config/mcp_config.json` | 워크스페이스 파일은 비밀값이 없을 때만 커밋, 토큰이 들어가면 README에 예시만 |

### 3-9. Step 7 — CI/CD · Evals

**CI:** 로컬 `verify`가 안정된 뒤 같은 명령을 호출합니다. CI 전용 검사 단계는 따로 두지 않습니다.

```yaml
# .github/workflows/ci.yml (Node 예시 — 액션 버전은 사용 전 최신 확인)
name: ci
on: [push, pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-node@v6
        with: { node-version-file: .nvmrc, cache: npm }
      - uses: actions/setup-python@v6
        with: { python-version: "3.12" }   # sync_skills.py 등 스크립트용
      - run: npm ci
      - run: npm run verify
```

브랜치 보호: `main` 병합 전 CI 통과 + 리뷰 필수.

**관측성:** 에이전트가 실행 명령·종료 코드, 오류 전문, 앱 로그, 브라우저 콘솔·스크린샷, git diff, CI 실패 로그를 볼 수 있게 합니다.

**Evals (초기 3~5개):** API 입력 검증 추가 / 기존 페이지 기능 추가 / DB 스키마 변경 / 버그 수정 / 코드 리뷰. 같은 과제를 에이전트·모델별로 돌려 비교하고, 모델명·판정 기준·결과·사람이 고친 횟수·시간을 기록합니다.

### 3-10. Step 8 — 실기능 검증과 피드백 루프

팀원 각자 작은 기능을 브랜치에서 구현·리뷰·병합한 뒤, 실패 원인별로 개선합니다.

| 원인 | 개선 위치 |
| --- | --- |
| 코드 오류 | 코드·테스트 (재현 테스트 먼저) |
| 지침 모호 | `AGENTS.md` (Claude 전용이면 `CLAUDE.md`) |
| 반복 절차 부족 | 해당 Skill 보완 → sync |
| 위험 명령 실행 | 해당 에이전트 권한 설정 |
| 환경 재현 실패 | 실행 스크립트·환경 설정 |
| 병합 충돌·계약 불일치 | 담당·계약 절차 점검, `CODEOWNERS`·스키마 검증 |
| 특정 에이전트만 규칙 무시 | 그 에이전트의 로딩 방식 확인 (§3-4) |

---

## 4. 워크플로우 (8단계)

### 4-0. 시작 전 결정

- **사람 → 기능**을 먼저 정합니다(기능 단위 수직 분할: 화면 → API → DB). 에이전트는 담당자가 고릅니다.
- **전체 기한은 1단계에서** 정하고, 4단계 `docs/03-plan.md`에서 기한부터 거꾸로 마일스톤·기능 동결일·버퍼를 정합니다.
- 참여 인원·요금제·사용 가능 모델은 `docs/00-brainstorm.md` 상단에 적습니다.

### 4-1. 단계표

| 단계 (스킬) | 업무 | 담당 모델 · 환경 | 추론 | 교차 검증 | 결과물 |
| --- | --- | --- | --- | --- | --- |
| 1 `/brainstorming` | 아이디어 브레인스토밍 | Gemini T3 · Gemini 앱 | High | 선택 (0~1회) | `docs/00-brainstorm.md` |
| 2 `/grill-with-docs`, `/research`, 선택 `/prototype` | 구체화, 실현 가능성 검증 | Gemini T1 · Gemini 앱 (불가 시 Claude T1) | High | **필수**, Claude T1 (1~2회) | `docs/01-concept.md`, `docs/01-feasibility.md`, `docs/meetings/` |
| 3 `/to-spec`, `/grill-with-docs` | 요구사항 명세·검토, `SPEC.md` 확정 | 초안 Gemini T1 / 검토 Claude T1 | Medium → High | **필수 2회** (최대 3회) | `docs/02-requirements.md`, `GLOSSARY.md`, `SPEC.md` |
| — 인계 | 기획 문서 커밋 → 스킬 설치 → 하네스 구축 (§1-3, §3) | | | | |
| 4 `/plan` | 아키텍처 설계, 담당 지정, Task 분할 (§4-3) | Claude T1 · Claude Code 또는 Antigravity `/plan` | High | **필수**, Gemini T1 (2회, 최대 3회) + 사용자 승인 | `docs/03-architecture.md`, `docs/adr/`, `api/openapi.yaml`, `docs/03-plan.md`, `docs/progress.md` |
| 5 `/implement`, `/tdd`, `verification-before-completion` | 기능별 개발, PR 단위로 작게 병합 | 일반 Claude T2 / 단순 Gemini T3 / 복잡 Claude T1 / 독립 모듈 Codex T2 | Medium / Low / High / Medium | 변경 위험 기준: 일반 PR은 상대 팀원, 복잡한 모듈은 Codex T1 추가 | 소스, 단위 테스트, PR 설명, `CHANGELOG.md` |
| 6 `/code-review`, `agent-browser`, `improve-codebase-architecture`, 선택 `/diagnosing-bugs` | 코드 리뷰, 통합 테스트 | 테스트 Claude T2 / 리뷰 팀원 + Codex T1 / 중요 영역 Codex T1 + Gemini T1 | Medium / Medium / High | **필수**: 일반 PR 1회, 인증·결제·공통 파일·API 계약 PR은 2회 + 사람 리뷰 | 통합·E2E 테스트, `docs/06-test-report.md`, `docs/reviews/` |
| 7 `finishing-a-development-branch`, 조건부 `/deploy-to-vercel` | 배포·운영 | Claude T2 (장애 시 T1) | Medium / 고위험 High | 일반 0회, DB 마이그레이션·인프라 변경 1~2회 (Codex T1) | CI/CD, `docs/07-deploy.md`, `docs/07-runbook.md` |
| 8 `/pptx` | 문서화·발표 | 초안 Gemini T1 / 검토 Claude T1 | Medium | 보고서 1회 권장 | 최종 `README.md`, 보고서, 발표 자료, 사용자 매뉴얼 |

- Claude가 작성한 코드는 Codex(OpenAI)가, Gemini·Codex가 작성한 코드는 다른 계열(예: Claude T1)이 리뷰합니다.
- 5단계 `/implement`는 마지막에 `code-review`를 자동으로 돌리므로, 6단계에서는 **PR 전체 diff**를 대상으로 팀원·다른 계열 모델 리뷰를 합니다.

### 4-2. 스킬 사용 요령

- **`/brainstorming`:** "아이디어 후보를 넓게 나열하고 비교까지만 정리해. **설계·구현 계획은 만들지 마.**" 범위를 정하지 않으면 설계 문서까지 만들고 앱에 없는 `writing-plans`를 부르려 합니다. → `docs/00-brainstorm.md`
- **`/grill-with-docs`:** `/brainstorming`과 같은 단계에서 함께 쓰지 않습니다. 용어집·ADR 결과는 프로젝트로 옮깁니다.
- **`/research`:** 앱에서 백그라운드 실행이 안 되면 대화로 진행합니다.
- **`/to-spec`:** "이슈 트래커에 게시하지 말고 `SPEC.md` 템플릿 항목대로 출력해."
- **`/implement`:** "docs/03-plan.md의 T-02를 구현해"처럼 Task 위치를 함께 줍니다. 기능 브랜치에서 시작합니다. `/implement`는 마지막에 `code-review`를 부른 뒤 현재 브랜치에 **바로 커밋**하는데, `code-review`는 커밋된 변경(`git diff <기준>...HEAD`)만 봅니다. 그래서 커밋 전에는 리뷰할 diff가 비어 있습니다. 요청할 때 순서를 정해 줍니다: "code-review 단계 전에 verify를 실행해 통과시키고, 통과하면 먼저 커밋한 뒤 main 대비로 code-review를 실행해라. 리뷰 지적은 별도 커밋으로 반영해라."
- **`/code-review`:** "main 대비 diff를 `SPEC.md` 기준으로 검토해"처럼 기준과 경로를 함께 줍니다. 결과는 `docs/reviews/`에 옮깁니다. Codex에서 쓸 때는 표준·spec 두 축이 따로 나오는지 처음 한 번 확인합니다.
- **`improve-codebase-architecture`:** 마일스톤 사이, 코드가 쌓인 뒤. 후보를 이슈로 등록해 하나씩 고칩니다.
- **`finishing-a-development-branch`:** 2번(push + PR)만 고르고, 테스트는 `verify`로 돌리라고 지시합니다. 1번은 로컬 `main`에 병합하므로 쓰지 않습니다.

### 4-3. 4단계 작업 분해: Plan 모드

Task는 Claude Code 내장 **Plan 모드**로 나누고 **`docs/03-plan.md` 하나**에 모읍니다(`/to-tickets`는 Task마다 파일을 만들어 쓰지 않음). Antigravity는 `/plan`에 같은 프롬프트를 씁니다.

- **들어가기:** `Shift+Tab`, 또는 프롬프트 앞에 `/plan`.
- **승인:** 계획이 나오면 `Ctrl+G`로 직접 수정 → Gemini T1 교차 검증(최대 3회) → 승인.
- **모델 분리:** 계획은 Opus, 구현은 Sonnet으로 하려면 `/model opusplan`.

```text
/plan AGENTS.md, SPEC.md, docs/03-architecture.md, api/openapi.yaml, docs/02-requirements.md를 먼저 읽고 구현 계획을 세워라.
1. 기능 ID(FR-xxx)별로 화면 → API → DB를 관통하는 작은 Task로 나눠라(수직 분할, 각 Task는 PR 하나로 병합·검증 가능).
2. 각 Task에 ID(T-NN), 제목, 기능 ID, 담당(사람 · 에이전트), 선행 Task(Blocked by), 완료 조건(SPEC.md 기준),
   확인 방법(단위·통합 테스트, verify, 화면·API), 공통 파일(API 계약·공통 타입·마이그레이션·의존성) 수정 여부,
   변경 위험(낮음·보통·높음)을 적어라.
3. 공통 기반 Task를 맨 앞에 두고, 병렬로 진행 가능한 Task를 표시하라.
4. 전체 기한에서 거꾸로 계산해 마일스톤·기능 동결일·버퍼를 제안하고 Task를 마일스톤에 배정하라.
5. 기능 담당·공통 기반 담당·병합 담당 표를 제안하라.
6. SPEC.md에 없는 기능을 추가하지 말고, 불명확한 점은 질문하라.
승인 후 첫 작업으로 docs/03-plan.md(담당 표 + 마일스톤 + Task 목록 표)와 docs/progress.md(전 Task 상태: 대기)를 작성하고 커밋하라.
Task마다 별도 파일을 만들지 마라.
```

```markdown
## Task 목록
| ID | 제목 | 기능 ID | 마일스톤 | 담당 | Blocked by | 완료 조건 | 확인 방법 | 공통 파일 | 위험 | 이슈·PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-01 | 공통 타입·API 계약 | — | M1 | 팀원 A · Claude Code | — | openapi.yaml 스키마 검증 통과 | verify | 예 | 높음 | |
| T-02 | 검색 → 추천 API → 결과 표시 | FR-001 | M1 | 팀원 A · Claude Code | T-01 | 검색 시 추천 3개 표시 | 통합 테스트 + 화면 | 아니오 | 보통 | |
```

- 상태 변경은 `docs/progress.md`에서만 하고, `docs/03-plan.md` Task 표는 승인 이후 사람만 고칩니다.
- GitHub Issues를 함께 쓰면 Task 하나에 이슈 하나를 만들고 "이슈·PR" 열에 번호만 적습니다.

---

## 5. 공통 운영 규칙

### 5-1. 완료 판정 (4조건) · 리뷰 강도

1. `SPEC.md` 완료 조건 충족 2. `verify` 실제 실행·통과 3. 실제 화면·API 동작 확인 4. 변경 위험에 맞는 리뷰

| 변경 위험 | 예시 | 리뷰 |
| --- | --- | --- |
| 낮음 | 문구·스타일, 담당 기능 내부 작은 수정 | `verify` + 직접 확인 |
| 보통 | 일반 기능 추가·수정 | 담당자 외 1명 1회 |
| 높음 | 인증·결제·개인정보, DB 마이그레이션, 공통 파일, API 계약, 배포 설정 | 팀원 + 다른 계열 모델 |

- 1라운드 = 검증 → 수정 → 재검증. 치명·중요 0건이면 바로 끝냅니다. "치명·중요"에는 근거가 필요합니다.
- 최대 횟수에도 의견이 갈리면 사람이 결정하고 이유를 `docs/adr/`에 남깁니다.
- 2회차부터 이전 리뷰 파일을 함께 전달합니다.

### 5-2. 리뷰 파일 양식

```markdown
# 06-review-r1
- 대상: (파일 또는 PR 번호) / 기능 담당: (팀원)
- 작성: Claude T2 (실제 모델명) · 검증: Codex T1 (실제 모델명) / 상대 팀원
- 변경 위험: 낮음 · 보통 · 높음 / 라운드: 1 / 최대 2

| 지적 사항 | 근거 (재현·로그·코드 위치) | 심각도 (치명·중요·사소) | 반영 여부 | 반영하지 않은 이유 |
| --- | --- | --- | --- | --- |
| 토큰 만료 시 예외 처리 누락 | `auth/session.ts:42`, 만료 토큰 요청 시 500 재현 | 중요 | 반영 완료 | - |
```

캡스톤 보고서의 "AI 교차 검증 과정" 근거로 그대로 쓸 수 있습니다.

### 5-3. 브랜치·병합·공통 파일

- `feat/<기능ID>-<짧은이름>`, `main` 직접 커밋 금지, `main`은 항상 `verify` 통과.
- 한 컴퓨터에서 여러 에이전트를 쓰면 `git worktree add ../<프로젝트>-<기능ID> feat/<기능ID>`.
- 공통 파일은 담당자만(에이전트는 멈추고 보고). API 변경은 계약(`openapi.yaml`/`SPEC.md` §6) → 공유 → 코드.
- DB 마이그레이션은 공통 기반 담당이, 잠금 파일 충돌은 재설치로 해결.
- **최소 하루 1회 또는 PR 단위로 병합**, 병합 전 최신 `main` 반영 후 `verify`.

### 5-4. 진행 기록 (`docs/progress.md`)

```markdown
# progress.md
| ID | Task | 담당 | 브랜치 | 상태 | 마지막 verify | 메모 |
| --- | --- | --- | --- | --- | --- | --- |
| T-01 | 로그인 API | 팀원 A · Claude Code | feat/fr-auth-01-login | 통과 | 11-02 통과 | |
| T-02 | 로그인 화면 | 팀원 B · Antigravity | feat/fr-auth-04-login-ui | 진행 | 11-02 typecheck 실패 | 오류 응답 형식 미정 |

## 세션 로그
- 11-02 · 팀원 A · Claude Code · T-01 완료, main 병합. 다음: §6 오류 응답 형식 확정
```

- 세션 시작: progress + git log → `verify` → Task 선택. 종료: 상태·로그 갱신 → 커밋.
- 에이전트는 Task 삭제·완료 조건 변경을 하지 않습니다.

### 5-5. 세션·추론 수준

- 새 세션: 독립 Task, 인수인계, 컨텍스트가 혼잡할 때. 같은 Task의 후속 수정은 같은 세션에서. **압축보다 리셋.**
- 넓은 탐색·조사·로그 분석은 서브에이전트나 `/research`에 맡기고, 메인 세션에는 요약만 가져옵니다.
- 추론 수준은 세션 시작 때 정하고 바꾸지 않습니다. 부족하면 T1으로 바꿉니다.

---

## 6. 하네스 유지보수

| 주기 | 점검 | 방법 |
| --- | --- | --- |
| 마일스톤마다 | 패턴 이탈·중복·강결합 | `improve-codebase-architecture` → 이슈로 하나씩 |
| 마일스톤마다 | 문서 최신성 | 아래 프롬프트로 문서 ↔ 레포 대조 |
| 같은 리뷰 지적 2회 반복 | 규칙 자동화 | 린터·테스트로 옮기고 `AGENTS.md`에서 삭제 |
| 모델 교체·분기마다 | 하네스 덜어내기 | 안 쓰는 규칙·Skill·MCP·Hook 제거. Claude Code `/skill-doctor`(스킬별 사용 빈도), `/doctor prompt-audit`(지침 파일 점검) 활용 |
| 마일스톤·프로젝트 회고 | 개선 후보 | (선택) `retro`가 세션을 읽고 개선 후보를 제안, 반영은 사람이 결정 |

```markdown
# 하네스 최신성 점검
AGENTS.md, CLAUDE.md, SPEC.md, docs/, .agents/skills/, MCP 설정을 실제 저장소와 대조하라.
1. 문서가 가리키는 파일·디렉터리·명령이 존재하고 동작하는가
2. 문서와 실제 코드·설정이 다른 곳 (특히 SPEC.md API 계약 vs 실제 API)
3. 린터·테스트로 이미 강제되어 중복된 AGENTS.md 규칙
4. 최근 쓰이지 않은 Skills·MCP·Hooks
5. docs/progress.md 상태와 실제 verify 결과 일치 여부
6. docs/03-plan.md 담당 표와 실제 공통 파일 변경 이력 일치 여부
파일은 수정하지 말고 목록·근거·영향도·우선순위만 보고하라.
```

---

## 7. 최종 체크리스트

**스킬**
- [ ] 데스크톱 앱에 기획 스킬 등록
- [ ] 전역 `~/.claude/skills/`에 `find-skills`만, 개인 설정 `"syncClaudeAiSkills": false`
- [ ] 프로젝트 스킬은 `.agents/skills/`에만, 하위 스킬까지 설치, `sync_skills.py --check` 통과
- [ ] `## Agent skills` 블록이 `AGENTS.md`에만

**하네스**
- [ ] `SPEC.md`가 테스트 가능한 수준이고 유일한 기준
- [ ] 담당 표 기록, 기능 브랜치 작게 자주 병합
- [ ] API 계약 변경이 코드보다 먼저 반영·공유
- [ ] `AGENTS.md`는 레포에서 알 수 없는 정보만, `CLAUDE.md`는 import + 전용 지침만
- [ ] 세 에이전트 모두 규칙·스킬 로딩 확인
- [ ] 차단 장치가 모든 팀원 컴퓨터에서 "막힘/통과" 테스트 통과
- [ ] 단일 `verify`가 모든 OS와 CI에서 동작, clone 후 README대로 통과
- [ ] 완료 판정이 실행 결과 기반
- [ ] 반복 제약은 린터·스키마·테스트로 자동 검사, `CODEOWNERS`·브랜치 보호
- [ ] 외부 Skills·MCP는 출처 검토·버전 고정, 비밀값은 설정 파일에 없음
- [ ] `docs/progress.md`와 git log만으로 작업을 이어받을 수 있음

---

## 8. 템플릿으로 시작하고 에이전트에게 맡기기 (VS Code)

**고정 파일은 템플릿에, 프로젝트마다 바뀌는 것만 에이전트에게** 맡깁니다.

| 템플릿에 이미 있음 (확인만) | 프로젝트마다 정함 |
| --- | --- |
| `.gitignore`·`.env.example`, `CLAUDE.md`, `settings.json`·`guard.py`, `sync_skills.py`·`check_skill_deps.py`, `docs/` 양식, CI·`CODEOWNERS`·Dependabot | 스킬 선정, `verify` 명령, `AGENTS.md`의 `<...>` 자리 |

**순서**

1. GitHub에서 `harness-template-capstone` 템플릿 → **Use this template** → clone → VS Code. `SPEC.md`(와 기획 문서)를 넣고 커밋합니다.
2. Claude Code에게 3가지를 한 번에 요청합니다.

```text
SPEC.md와 AGENTS.md를 읽고, 파일은 아직 커밋하지 마라.
1) /find-skills로 계획 검증 · 구현 · 도메인 도구 · 동작 확인 · 완료 확인 · 리뷰 · 버그 진단 · 발표
   역할마다 스킬 후보를 표로 보여줘. 설치는 하지 말고, 각 후보가 내부에서 부르는 하위 스킬도 적어.
2) 실제 기술 스택(package.json 등)에 맞는 verify 명령을 구성해줘. 없는 명령은 만들지 마.
3) AGENTS.md의 <...> 자리만 채운 초안을 보여줘. 가이드 내용을 복사하지 마.
```

3. 사람이 고르고 마무리합니다: 후보 `SKILL.md` 검토 → 설치 → `check_skill_deps.py` → `sync_skills.py` → `/setup-matt-pocock-skills`(블록은 `AGENTS.md`에) → `AGENTS.md` 초안을 줄여서 커밋 → `README.md` Skills 표 작성.
4. 마지막 확인(로딩 확인, guard 테스트, `verify`)은 사람이 터미널에서 직접 합니다.

| 에이전트가 할 일 | 사람이 할 일 |
| --- | --- |
| 역할별 스킬 후보·하위 스킬 표 | 외부 스킬 `SKILL.md` 검토, 설치 결정 |
| `verify` 구성, `AGENTS.md` **초안** | 초안을 줄여서 커밋 |
| 로딩 확인·guard 테스트·`verify` 실행 | Plan 승인, `/setup-matt-pocock-skills` 답변, 신뢰 처리, `.env` 값, 브랜치 보호, `CODEOWNERS` 실제 아이디 |

- 이 가이드는 `docs/harness-guide-capstone.md` 같은 **참고 문서**로만 둡니다. `AGENTS.md`·`CLAUDE.md`에 복사하지 않습니다.
- 템플릿 없이 시작한다면 §3 하네스 구축 Step 1~8를 한 단계씩(계획 → 승인 → 실행 → 보고) 에이전트에게 맡깁니다.

---

## 부록. 모델 등급

| 등급 | Gemini (Antigravity) | Claude (Claude Code) | OpenAI (Codex) |
| --- | --- | --- | --- |
| T1 · 상위 추론 | Gemini 4 Argon | Claude Opus 5.5 | GPT-6 Astra |
| T2 · 표준 | Gemini 4 Hydrogen (High) | Claude Sonnet 5.5 | GPT-6.1 Sol |
| T3 · 경량 | Gemini 4 Hydrogen (Low~Medium) | Claude Haiku 5.5 (없으면 Sonnet 5.5 `low`) | GPT-6 Luna |

| 공통 수준 | Gemini | Claude Code `/effort` | Codex |
| --- | --- | --- | --- |
| Low / Medium / High | Low / Medium / High | `low` / `medium` / `high` | Light / Medium / High |
| 예외 상향 | `Extra` (Hydrogen만, CLI 값 `xhigh`) | `xhigh` (이 가이드는 Opus 5.5에만 사용) | Extra High |

- Antigravity에서 Hydrogen은 모든 요금제에 있고 최고 단계는 `Extra`(CLI `xhigh`)입니다. Argon은 유료 Pro(또는 적격 기기의 Pro 체험) 이상, Claude Sonnet·Opus 5.5는 Pro(체험 제외) 이상에서만 쓸 수 있고 Haiku는 없습니다. 쓸 수 없으면 T1 작업은 Claude Code의 Opus 5.5 등 다른 에이전트로 옮기고 실제 모델명을 기록합니다.
- Antigravity의 Claude 5.5는 Pro(체험 제외) 이상에서만 쓸 수 있습니다.

**대체 규칙:** Gemini T1 불가 → Claude T1 / Codex T1 불가 → Codex T2 High → 다른 계열 T1 / Claude Code 불가 → Antigravity의 Claude(불가하면 Codex). 작성·검증 모델이 같은 계열이 되면 남은 다른 계열이 검증합니다.

---

## 출처

- [vercel-labs/skills](https://cdn.jsdelivr.net/gh/vercel-labs/skills@main/README.md) · [find-skills](https://raw.githubusercontent.com/vercel-labs/skills/main/skills/find-skills/SKILL.md)
- [mattpocock/skills](https://raw.githubusercontent.com/mattpocock/skills/main/README.md) · [AI Hero — Skills](https://www.aihero.dev/skills)
- [obra/superpowers](https://github.com/obra/superpowers) · [agent-browser](https://agent-browser.dev/installation)
- [Claude Code Docs — Skills](https://code.claude.com/docs/en/skills) · [Permissions](https://code.claude.com/docs/en/permissions) · [Hooks](https://code.claude.com/docs/en/hooks) · [Model config](https://code.claude.com/docs/en/model-config)
- [Claude Help Center — Use Skills](https://support.claude.com/en/articles/12512180-use-skills-in-claude)
- [Codex — Config reference](https://learn.chatgpt.com/docs/config-file/config-reference)
- [Antigravity — Choose a model](https://antigravity.google/docs/models/)
- [OpenAI — Harness engineering](https://openai.com/index/harness-engineering/)
