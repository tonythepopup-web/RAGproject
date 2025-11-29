# HACCP 법률 특화 RAG 시스템

HACCP 관련 법률 및 매뉴얼에 특화된 검색증강생성(RAG) 시스템입니다.

## 주요 기능

### Boot Once 아키텍처
프로그램 시작 시 모든 리소스(LLM, 인덱스)를 1회 로딩하여 질의 처리 속도를 최적화합니다.
- **초기화**: LLM 모델, 법률/매뉴얼 인덱스(총 28개 카테고리) 메모리 로딩 (1회)
- **질의 처리**: 카테고리 자동 분류 → 하이브리드 검색(의미+키워드) → LLM 답변 생성 → 결과 라벨링

### 기술 스택
- **임베딩 모델**: dragonkue/BGE-m3-ko
- **LLM**: NCSOFT/VARCO-VISION-2.0-14B
- **벡터 DB**: FAISS

---

## 프로젝트 구조
`00_data/output/finetuning/` 폴더만 `.gitignore`로 GitHub에서 제외되며 (4GB+), 나머지는 Git으로 관리합니다.
매뉴얼 원본 파일(hwp)은 현재 따로 저장해놓지 않으며, 이를 변환한 html 데이터만 저장 후 사용합니다.

```
.
├── 00_data/                     # 데이터
│   ├── input/                   # 원본 데이터 및 인덱스 (읽기 전용)
│   │   ├── raw_law/
│   │   │   └── 법률파일원본(pdf)/   # 원본 PDF
│   │   ├── raw_manual/
│   │   │   ├── 매뉴얼_1차_전처리(html_to_blocks)/  # 원본 HTML/HWP
│   │   │   └── 매뉴얼_1차_전처리결과물(json파일모음)/  # 1차 전처리 결과
│   │   └── indexes/             # 인덱스 (날짜별 버전 관리)
│   │       ├── law/
│   │       │   └── YYYY-MM-DD/  # 법률 인덱스 (날짜별)
│   │       │       ├── idx_식품위생법/
│   │       │       ├── idx_축산물 위생관리법/
│   │       │       └── ... (11개 카테고리)
│   │       └── manual/
│   │           └── YYYY-MM-DD/  # 매뉴얼 인덱스 (날짜별)
│   │               ├── idx_1. 효율적인.../
│   │               ├── idx_11. HACCP.../
│   │               └── ... (17개 카테고리)
│   └── output/                  # 생성 데이터 (쓰기)
│       ├── logs/                # RAG 실행 시 질문/답변/검색결과 자동 저장 (result.txt)
│       ├── training_data/       # 라벨링 반영 시 triplet 학습 데이터 누적 저장 (triplets_group_bgem3.jsonl)
│       ├── benchmark_result/    # 벤치마크 테스트 실행 시 성능 평가 결과 (Excel)
│       └── finetuning/          # 임베딩 모델 파인튜닝 시 생성 (4GB+, Git 제외)
│
├── 01_preprocess/               # 데이터 전처리 스크립트
│   ├── law/
│   │   ├── preprocess_law_final.py  # PDF → 인덱스 (날짜별 저장)
│   │   └── preprocess_law_final_OOM완화.py  # OOM 완화 버전
│   └── manual/
│       ├── 매뉴얼_1차_전처리코드(json파일 생성)/  # HTML → JSON
│       └── preprocess_manual_final.py  # JSON → 인덱스 (날짜별 저장)
│
├── 02_rag/                      # RAG 실행 및 평가
│   ├── haccp_rag.py             # 메인 라우터
│   ├── run_law_final.py         # 법률 RAG 엔진
│   ├── run_manual_final.py      # 매뉴얼 RAG 엔진
│   ├── benchmark/               # 성능 평가 도구
│   └── finetuning/              # 임베딩 모델 파인튜닝 (미사용)
│
├── config.py                    # 경로 설정 파일
└── README.md
```

---

## 실행 방법

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install torch torchvision sentence-transformers faiss-cpu keybert
```

### 2. 데이터 설정

**중요**: `00_data/output/finetuning/`만 `.gitignore`로 제외됩니다 (4GB+).
나머지 데이터는 Git으로 관리되므로 GitHub에서 클론 시 함께 다운로드됩니다.

#### 경로 설정 방법

**옵션 1: 프로젝트 내부 (기본값)**
```bash
# 프로젝트 루트에 00_data/ 폴더 생성
cslee_outsourcing/
├── 00_data/          # <- 여기에 데이터 배치
├── 01_preprocess/
└── 02_rag/
```

**옵션 2: 외부 경로 사용**
```bash
# 환경변수 설정
export HACCP_DATA_ROOT=/path/to/your/data  # Linux/Mac
set HACCP_DATA_ROOT=C:\path\to\data       # Windows

# 또는 config.py 수정
DATA_ROOT = "/your/custom/path"
```

#### 경로 확인
```bash
python config.py  # 모든 경로 확인 및 검증
```

### 3. RAG 시스템 실행

```bash
cd 02_rag
python haccp_rag.py
```

실행 후 법률 RAG 또는 매뉴얼 RAG 모드를 선택하고 질문을 입력합니다.

---

## 데이터 처리 파이프라인

```
[원본 문서]
    ↓
[전처리 스크립트] (PDF→JSON, HTML→JSON)
    ↓
[사용자 질의 입력 및 대형별 모듈]
    ↓
[임베딩 및 인덱싱] (BGE-m3-ko, FAISS)
    ↓
[하이브리드 검색 모듈] (코사인 + IDF)
    ↓
[질의 정규화 및 리랭킹 모듈]
    ↓
[하이브리드 검색 모듈] (최종 상위 n개 추출)
    ↓
[생성형 AI 모델] (VARCO-VISION-2.0-14B)
    ↓
[답변 출력]
```

---

## 카테고리 구성

### 법률 카테고리 (19개)
1. 식품위생법
2. 축산물 위생관리법
3. 건강기능식품에 관한 법률
4. 먹는물관리법
5. 가축전염병 예방법
6. 사료관리법
7. 농약관리법
8. 수입식품안전관리
9. 식품·의약품분야 시험·검사
10. 한국식품안전관리인증원의 설립 및 운영에 관한 법률
11. 전체 (통합 검색)
12. 개인정보 보호법: 현대 사회 핵심 이슈, 디지털 윤리와 직결
13. 근로기준법: 노동권, 최저임금, 근로시간
14. 주택임대차보호법: 전월세 분쟁 (대학생/사회초년생 필수)
15. 도로교통법: 운전, 교통사고, 음주운전, 보행자 보호
16. 소비자기본법: 환불, 피해 보상 등
17. 성폭력범죄의 처벌 등에 관한 특례법
18. 장애인차별금지 및 권리구제 등에 관한 법률
19. 양성평등기본법

### 매뉴얼 카테고리 (17개)
- 방충·방서 가이드라인
- 온라인 유통식품 위생안전관리 가이드라인
- HACCP 인증업체 위탁생산 바로알기
- 소독·헹굼 공정 관리 가이드라인
- 훈제연어 제조 가이드라인
- 김치 제조 가이드라인
- 소독·헹굼(CCP-BC) 변경 시 따라하기
- HACCP 인증 따라하기
- 심사 업무 매뉴얼 (2024, 2025)
- 스마트 HACCP 업무 매뉴얼
- 주요수출국 규제정보지
- 수출국정부 현지실사 대응 매뉴얼 (대만, 캐나다)
- 중국 수출식품 생산업체 등록 매뉴얼
- OEM 식품 위생평가 업무 매뉴얼
- 제외국 제도 조사분석 보고서
- 전체 (통합 검색)

---

## 주요 기능 설명

### 1. 카테고리 자동 추천
질의 텍스트를 임베딩하여 가장 유사한 카테고리를 자동으로 추천합니다.

### 2. 하이브리드 검색
- **의미 기반 검색**: BGE-m3-ko로 생성한 벡터 임베딩의 코사인 유사도
- **키워드 검색**: TF-IDF 기반 키워드 매칭
- **통합 점수**: alpha * 코사인_유사도 + (1-alpha) * IDF_점수

### 3. 답변 생성
검색된 상위 문서를 컨텍스트로 제공하여 LLM이 질문에 대한 답변을 생성합니다.

### 4. 검색 결과 라벨링
사용자가 검색 결과의 관련성을 평가하여 피드백을 제공할 수 있습니다.

---

## 데이터 전처리

### 법률 데이터
- **입력**: `00_data/input/raw_law/법률파일원본(pdf)/`
- **출력**: `00_data/input/indexes/law/YYYY-MM-DD/idx_법률명/`
- **스크립트**: `01_preprocess/law/preprocess_law_final.py`
- **OOM 완화 버전**: `01_preprocess/law/preprocess_law_final_OOM완화.py`

### 매뉴얼 데이터 (2단계 전처리)

**1단계: HTML → JSON 변환**
- **입력**: `00_data/input/raw_manual/매뉴얼_1차_전처리(html_to_blocks)/`
- **출력**: `00_data/input/raw_manual/매뉴얼_1차_전처리결과물(json파일모음)/`
- **스크립트**: `01_preprocess/manual/매뉴얼_1차_전처리코드(json파일 생성)/`

**2단계: JSON → 인덱스 생성**
- **입력**: `00_data/input/raw_manual/매뉴얼_1차_전처리결과물(json파일모음)/`
- **출력**: `00_data/input/indexes/manual/YYYY-MM-DD/idx_매뉴얼명/`
- **스크립트**: `01_preprocess/manual/preprocess_manual_final.py`

---

## 벤치마크

성능 평가를 위한 벤치마크 도구가 포함되어 있습니다.

- **스크립트**: `02_rag/benchmark/Benchmark_TEST.py`
- **질문 데이터**: `02_rag/benchmark/Benchmark_qna.txt`
- **결과 저장**: `00_data/output/benchmark_result/` (Excel 형식)

다양한 검색 방식 및 리랭커를 비교 평가한 결과가 포함되어 있습니다.

---

## 파일 형식

### 인덱스 파일 구조
```
idx_[카테고리명]/
├── index.faiss      # FAISS 벡터 인덱스
├── vectors.npy      # 벡터 임베딩 (numpy 배열)
├── chunks.json      # 청크/블록 텍스트 및 메타데이터
└── docs.txt         # 문서 ID 목록
```

---

## 참고사항

### 미사용 구성 요소
- **`00_data/output/finetuning/`**: 임베딩 모델 파인튜닝 실험 (4.28GB)
  - 현재 시스템은 원본 BGE-m3-ko 모델 사용
  - Git에서 제외됨 (필요시 삭제 가능)

### 데이터 생성 위치
- **`logs/`**: `haccp_rag.py` 실행 시 질문/답변/검색결과가 `result.txt`로 저장
- **`training_data/`**: 라벨링 입력 시 triplet 데이터가 `triplets_group_bgem3.jsonl`에 누적
- **`benchmark_result/`**: `Benchmark_TEST.py` 실행 시 성능 평가 결과 (Excel)
- 경로는 `config.py`에서 변경 가능

---
