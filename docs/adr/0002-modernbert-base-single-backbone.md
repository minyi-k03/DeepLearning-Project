# 백본은 ModernBERT-base 하나만 쓴다

---
status: accepted
---

가설의 근거가 된 선행연구(Devlin, Tenney, Liu, Sun)는 모두 BERT 기준이지만, 백본은 **ModernBERT-base 하나**로 정했다. ModernBERT 논문 부록에 GLUE 과제별 수치와 학습 설정(MNLI 89.1, SST-2 96.0)이 공개되어 있어 메인 백본으로 재현이 가능하고, 전역/지역 어텐션이 교대하는 구조가 BERT와 달라 "길이에 따라 어느 층에 기대는가"라는 분석 축이 생기기 때문이다.

## Considered Options

- **BERT-base 단독**: 선행연구와 바로 대조할 수 있지만, 다른 팀과 겹칠 가능성이 높고 구조적으로 새로 분석할 거리가 적다.
- **ModernBERT 메인 + BERT-base 기준점**: 처음에는 재현 단계(AG News, Sun et al.)에 BERT-base가 필요해서 넣었다. 데이터가 GLUE로 바뀌면서 그 역할이 사라졌고, 선행연구와의 비교는 논문 인용으로 대신하기로 했다.
- **소형 디코더 LLM 추가**: 차별화 효과는 크지만, 파인튜닝에 LoRA가 필요해 변인이 하나 늘고 저장 용량도 커서 제외했다.

## Consequences

- 선행연구와의 비교는 같은 실험을 다시 돌리는 게 아니라 **보고된 경향과의 정성 비교**다. 보고서 한계 절에 이를 적는다.
- ModernBERT는 층에 들어가기 전에 정규화하는 구조(pre-norm)라서 층 출력의 크기가 층마다 다르다. 이 차이 때문에 ADR 0005가 필요하다.
