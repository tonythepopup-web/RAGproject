"""
HACCP RAG 프로젝트 경로 설정

사용 방법:
1. 환경변수 HACCP_DATA_ROOT 설정 (선택)
   export HACCP_DATA_ROOT=/path/to/00_data

2. 또는 이 파일에서 직접 DATA_ROOT 수정

경로 구조:
00_data/
  ├── input/                       ← 원본 데이터 및 인덱스 (읽기 전용)
  │   ├── raw_law/법률파일원본(pdf)/
  │   ├── raw_manual/
  │   └── indexes/                 ← 인덱스 (날짜별 버전 관리)
  │       ├── law/
  │       │   └── YYYY-MM-DD/      ← 법률 인덱스 (날짜별)
  │       │       ├── idx_식품위생법/
  │       │       ├── idx_축산물 위생관리법/
  │       │       └── ...
  │       └── manual/
  │           └── YYYY-MM-DD/      ← 매뉴얼 인덱스 (날짜별)
  │               ├── idx_1. 효율적인.../
  │               ├── idx_11. HACCP.../
  │               └── ...
  └── output/                      ← 생성 데이터 (쓰기)
      ├── logs/                    ← result.txt, chat_log.txt
      ├── training_data/           ← triplets_group_bgem3.jsonl
      ├── benchmark/               ← benchmark_result/
      └── finetuning/              ← 파인튜닝 모델 (Git에서 제외)
"""

import os
from pathlib import Path

# ===== 데이터 루트 경로 설정 =====
# 우선순위: 환경변수 > 수동 설정 > 기본값(프로젝트 루트)
DATA_ROOT = os.getenv(
    "HACCP_DATA_ROOT",
    str(Path(__file__).resolve().parent / "00_data")  # 기본: 프로젝트 내 00_data/
)

DATA_ROOT = Path(DATA_ROOT)

# ===== INPUT: 원본 데이터 및 인덱스 (읽기 전용) =====
INPUT_DIR = DATA_ROOT / "input"

# 원본 데이터 (01_preprocess와 매핑)
RAW_LAW_DIR = INPUT_DIR / "raw_law" / "법률파일원본(pdf)"
RAW_MANUAL_HTML_DIR = INPUT_DIR / "raw_manual" / "매뉴얼_1차_전처리(html_to_blocks)"
RAW_MANUAL_JSON_DIR = INPUT_DIR / "raw_manual" / "매뉴얼_1차_전처리결과물(json파일모음)"

# 인덱스 (02_rag에서 참조)
# 날짜별 인덱스 구조: indexes/law/YYYY-MM-DD/, indexes/manual/YYYY-MM-DD/
# 실제 사용 시 run_law_final.py, run_manual_final.py에서 날짜를 지정하여 로드
IDX_LAW_DIR = INPUT_DIR / "indexes" / "law"
IDX_MANUAL_DIR = INPUT_DIR / "indexes" / "manual"

# ===== OUTPUT: 생성 데이터 (쓰기) =====
OUTPUT_DIR = DATA_ROOT / "output"

# 로그 파일
LOG_DIR = OUTPUT_DIR / "logs"
RESULT_FILE = LOG_DIR / "result.txt"
CHAT_LOG_FILE = LOG_DIR / "chat_log.txt"

# 학습 데이터
TRAINING_DATA_DIR = OUTPUT_DIR / "training_data"
TRIPLET_FILE = TRAINING_DATA_DIR / "triplets_group_bgem3.jsonl"

# 벤치마크
BENCHMARK_RESULT_DIR = OUTPUT_DIR / "benchmark_result"

# 파인튜닝 (Git에서 제외)
FINETUNING_DIR = OUTPUT_DIR / "finetuning"
FINETUNED_MODEL_DIR = FINETUNING_DIR / "finetuned_embedding_model"
CROSS_VALIDATION_DIR = FINETUNING_DIR / "cross_validation_결과"

# ===== vLLM Client 설정 =====
# VARCO 모델용 클라이언트 (지연 로딩)
VLLM_VARCO_BASE_URL = os.getenv("VLLM_VARCO_BASE_URL", "http://localhost:8400/v1")

# ===== 임베딩 서버 설정 =====
# bge-m3 임베딩 모델 원격 서버 사용 여부
# Docker 권한 문제로 로컬 모드를 기본값으로 설정
USE_REMOTE_EMBEDDING = os.getenv("USE_REMOTE_EMBEDDING", "false").lower() == "true"  # 기본값 false (로컬 모드)
# 임베딩 서버 URL (TEI 또는 커스텀 서버)
EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8401")

# OpenAI client 초기화는 필요 시점에 수행 (지연 로딩)
_vllm_varco_client = None
_vllm_embed_client = None

def get_vllm_varco_client():
    """VARCO 모델용 vLLM 클라이언트 (지연 로딩)"""
    global _vllm_varco_client
    if _vllm_varco_client is None:
        try:
            from openai import OpenAI
            _vllm_varco_client = OpenAI(
                base_url=VLLM_VARCO_BASE_URL,
                api_key="EMPTY"
            )
        except ImportError:
            raise ImportError("openai 패키지가 설치되어 있지 않습니다. pip install openai")
    return _vllm_varco_client

def get_embedding_client():
    """임베딩 서버 클라이언트 (지연 로딩)"""
    global _vllm_embed_client
    if _vllm_embed_client is None:
        try:
            import requests
            _vllm_embed_client = requests.Session()
        except ImportError:
            raise ImportError("requests 패키지가 설치되어 있지 않습니다. pip install requests")
    return _vllm_embed_client

def remote_embed(texts, normalize=True):
    """
    원격 임베딩 서버에서 임베딩 생성
    
    Args:
        texts: 문자열 또는 문자열 리스트
        normalize: 정규화 여부 (기본 True)
    
    Returns:
        numpy.ndarray: 임베딩 벡터 (단일 텍스트) 또는 벡터 배열 (리스트)
    """
    import numpy as np
    
    client = get_embedding_client()
    
    # 단일 텍스트를 리스트로 변환
    is_single = isinstance(texts, str)
    if is_single:
        texts = [texts]
    
    try:
        # TEI 호환 API 호출
        response = client.post(
            f"{EMBEDDING_SERVER_URL}/embed",
            json={"inputs": texts, "normalize": normalize},
            timeout=30
        )
        response.raise_for_status()
        embeddings = np.array(response.json())
        
        # 단일 텍스트면 첫 번째 벡터만 반환
        return embeddings[0] if is_single else embeddings
    
    except Exception as e:
        print(f"❌ 원격 임베딩 서버 요청 실패: {e}")
        print(f"   서버 URL: {EMBEDDING_SERVER_URL}")
        raise


class EmbeddingModelWrapper:
    """
    로컬/원격 임베딩 모델을 투명하게 처리하는 래퍼
    
    SentenceTransformer와 동일한 인터페이스 제공
    """
    def __init__(self, local_model=None, use_remote=False):
        self.local_model = local_model
        self.use_remote = use_remote
    
    def encode(self, texts, normalize_embeddings=True, convert_to_tensor=False, **kwargs):
        """
        SentenceTransformer.encode() 호환 인터페이스
        """
        import numpy as np
        
        if self.use_remote:
            # 원격 임베딩 서버 사용
            embeddings = remote_embed(texts, normalize=normalize_embeddings)
        else:
            # 로컬 모델 사용
            if self.local_model is None:
                raise RuntimeError("로컬 임베딩 모델이 로드되지 않았습니다.")
            embeddings = self.local_model.encode(
                texts, 
                normalize_embeddings=normalize_embeddings,
                convert_to_tensor=convert_to_tensor,
                **kwargs
            )
        
        if convert_to_tensor:
            import torch
            return torch.from_numpy(embeddings) if isinstance(embeddings, np.ndarray) else embeddings
        
        return embeddings

# ===== 기타 =====
PROJECT_ROOT = Path(__file__).resolve().parent

# ===== 경로 존재 확인 함수 =====
def check_paths():
    """필수 경로 존재 여부 확인"""
    required_paths = {
        "원본 법률 데이터": RAW_LAW_DIR,
        "원본 매뉴얼 JSON": RAW_MANUAL_JSON_DIR,
        "법률 인덱스 루트": IDX_LAW_DIR,
        "매뉴얼 인덱스 루트": IDX_MANUAL_DIR,
    }
    
    missing = []
    for name, path in required_paths.items():
        if not path.exists():
            missing.append(f"  - {name}: {path}")
    
    if missing:
        print("⚠️  다음 경로가 존재하지 않습니다:")
        print("\n".join(missing))
        print(f"\n💡 DATA_ROOT 설정: {DATA_ROOT}")
        print("   환경변수 HACCP_DATA_ROOT를 설정하거나 config.py를 수정하세요.")
        return False
    
    print(f"✅ 모든 필수 경로 확인 완료 (DATA_ROOT: {DATA_ROOT})")
    return True


def ensure_output_dirs():
    """Output 폴더 자동 생성"""
    dirs_to_create = [
        LOG_DIR,
        TRAINING_DATA_DIR,
        BENCHMARK_RESULT_DIR,
        FINETUNING_DIR,
    ]
    
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Output 디렉터리 준비 완료")


if __name__ == "__main__":
    # 테스트용: python config.py 실행 시 경로 확인
    print("=" * 70)
    print("HACCP RAG 경로 설정")
    print("=" * 70)
    print(f"DATA_ROOT: {DATA_ROOT}\n")
    
    print("📥 INPUT (원본 데이터, 읽기 전용)")
    print(f"  - 법률 원본: {RAW_LAW_DIR}")
    print(f"  - 매뉴얼 HTML: {RAW_MANUAL_HTML_DIR}")
    print(f"  - 매뉴얼 JSON: {RAW_MANUAL_JSON_DIR}")
    print(f"  - 법률 인덱스: {IDX_LAW_DIR}")
    print(f"  - 매뉴얼 인덱스: {IDX_MANUAL_DIR}\n")
    
    print("📤 OUTPUT (생성 데이터, 쓰기)")
    print(f"  - 로그: {LOG_DIR}")
    print(f"  - 학습 데이터: {TRAINING_DATA_DIR}")
    print(f"  - 벤치마크: {BENCHMARK_DIR}")
    print(f"  - 파인튜닝: {FINETUNING_DIR}\n")
    
    print("=" * 70)
    check_paths()
    print()
    ensure_output_dirs()
