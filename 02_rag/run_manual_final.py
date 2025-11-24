# -------------------------------------------------------------
# 목적:
#   - "이미 생성된" 문서별 인덱스(idx_singlelevel/idx_<카테고리>/)만 읽어서
#     검색 → (선택)LLM 답변 → 라벨링까지 단일 스크립트로 수행.
# 특징:
#   - 전처리/인덱스 생성 단계는 포함하지 않음. (오직 로딩+검색)
#   - 검색 점수는 의미론(임베딩) + 키워드(IDF) 가중 합의 하이브리드 방식.
#   - LLM은 VARCO-VISION을 텍스트만으로 호출(이미지 입력 없음).
#   - 코드/이름/로직은 원본과 동일하며, 주석만 자세히 보강.
# 디렉터리 기대 구조(예):
#   idx_singlelevel/
#     ├─ idx_전체/
#     │    ├─ chunks.json      # 청크 본문/메타/키워드(enriched_text 포함)
#     │    ├─ docs.txt         # (참고) 원문 요약/경로 등 텍스트
#     │    ├─ vectors.npy      # 문서 임베딩(문서/청크 기준)
#     │    └─ index.faiss      # FAISS 인덱스
#     ├─ idx_식품위생법/
#     └─ ...
# 사용 흐름:
#   1) 스크립트 실행 → LLM 로드(실패 시 검색/라벨링만) → 인덱스 로드
#   2) 사용자가 질의 입력 → 카테고리 추천/선택 → 검색 Top-k 산출
#   3) (선택) LLM에 상위 3개 청크로 프롬프트 구성하여 답변 생성
#   4) CLI에서 good/bad 복수 선택 → triplets JSONL로 라벨 저장
# =============================================================
import os, re, json, time, sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import AutoProcessor

# config.py에서 vLLM client 가져오기
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_vllm_varco_client

SCRIPT_DIR = Path(__file__).resolve().parent
EMBED_MODEL_NAME = "dragonkue/BGE-m3-ko"  # 임베딩 모델 이름
LLM_MODEL_ID     = "NCSOFT/VARCO-VISION-2.0-14B"  # LLM 모델 ID

# 날짜별 인덱스 로드 (indexes/manual/YYYY-MM-DD 구조)
# 고정: 실제 인덱스가 존재하는 날짜로 설정
INDEX_DATE = "2025-11-11"  # 인덱스 생성 날짜 (고정)
PARENT_DIR       = str(SCRIPT_DIR.parent / "00_data" / "input" / "indexes" / "manual" / INDEX_DATE)  # 인덱스 루트
TRIPLET_JSONL    = str(SCRIPT_DIR.parent / "00_data" / "output" / "training_data" / "triplets_group_bgem3.jsonl")  # 라벨 누적 저장 경로(JSONL)
SAVE_LOG         = str(SCRIPT_DIR.parent / "00_data" / "output" / "logs" / "result.txt")
# -------------------- 전역 모델 --------------------
# GPU 메모리 캐시 비우기
torch.cuda.empty_cache()

# KeyBERT, bge-m3: None으로 초기화 (GlobalBoot에서 주입되거나, 단독 실행 시 로드)
# GlobalBoot 사용 시 이 변수들은 GlobalBoot.__init__()에서 덮어씌워짐
kw_model = None
embed_model = None

# 단독 실행 확인 함수 (GlobalBoot에서 호출되지 않을 때만 로드)
def _init_models_if_needed():
    """단독 실행 모드일 때만 모델 로드"""
    global kw_model, embed_model
    
    if kw_model is None:
        print("[INFO] KeyBERT 로드 중 (단독 실행 모드)...")
        kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")
    
    if embed_model is None:
        print("[INFO] BGE-m3 임베딩 모델 로드 중 (단독 실행 모드)...")
        from config import USE_REMOTE_EMBEDDING, EmbeddingModelWrapper
        
        if USE_REMOTE_EMBEDDING:
            print("   원격 임베딩 서버 사용")
            embed_model = EmbeddingModelWrapper(local_model=None, use_remote=True)
        else:
            print("   로컬 임베딩 모델 로드")
            local_model = SentenceTransformer(EMBED_MODEL_NAME)
            embed_model = EmbeddingModelWrapper(local_model=local_model, use_remote=False)

# -------------------- 유틸  --------------------
# 텍스트 전처리/키워드 추출/소스 정리/간단 로깅 유틸리티.

def remove_josa(word: str) -> str:
    # 목적: 한국어 조사 제거(어절 말단) → IDF 매칭/키워드 교집합 품질 향상
    # 입력: word(단일 토큰)
    # 반환: 말단 조사 제거된 문자열(없으면 원문)
    for j in ['은','는','이','가','을','를','에','의','와','과','도','로','으로','에서','에게','한테','부터','까지','만','보다','처럼','조차','마저']:
        if word.endswith(j):
            return word[:-len(j)]
    return word

def get_keywords(text: str, ratio: float = 1, max_k: int = 30, min_k: int = 5) -> List[str]:
    # 목적: 질의 기반 키워드 집합 생성(top_n 자동 산정)
    # 파라미터:
    #  - ratio: 본문 길이에 따른 top_k 스케일 파라미터(경험적), max/min으로 클리핑
    #  - max_k/min_k: 상/하한
    # 반환: 조사 제거된 유니크 키워드 리스트(순서 비보장)
    # 예외: 추출 실패 시 빈 리스트
    est_k = int(len(text) * ratio / 10)
    top_k = max(min_k, min(est_k, max_k))
    try:
        kws = kw_model.extract_keywords(text, top_n=top_k)
    except Exception:
        return []
    return list({remove_josa(k[0]) for k in kws if isinstance(k, (list, tuple)) and k and isinstance(k[0], str)})

def preprocess_query(q: str) -> str:
    # 설계: 추가 정규화/토큰화 등은 하지 않음(원문 질의로 임베딩)
    return q.strip()

# -------------------- 카테고리/매핑 --------------------
# desired_categories: 사용자 선택/추천 리스트에 쓰일 카테고리 이름 목록(디렉터리명 기반)
# mapping           : "번호 문자열" → 카테고리명 맵핑
# category_embeddings: 카테고리명 자체를 문장 임베딩하여 질의와 유사도 계산할 때 사용
# 초기화는 init_rag_from_saved에서 수행.

desired_categories: List[str] = []
mapping: Dict[str, str] = {}
category_embeddings: Optional[np.ndarray] = None

# 매뉴얼 RAG용: parent_dir(=아카이브 루트)에서 카테고리 목록 추출
#  - 디렉터리명: idx_<카테고리표시제목>
#  - "전체"는 idx_all 사용
def _scan_categories(parent_dir: str) -> List[str]:
    # 목적: 루트에서 idx_<name> 패턴 디렉터리들을 찾아 카테고리명 리스트 생성
    cats: List[str] = []
    if not os.path.isdir(parent_dir):
        return cats
    for d in sorted(os.listdir(parent_dir)):
        if d.startswith("idx_") and d != "idx_all":
            cats.append(d[4:])
    return cats

# -------------------- IDF  --------------------
def compute_idf(chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    # 목적: 청크의 keywords 필드를 이용해 간단 IDF 계산
    # 수식: idf = log((N+1)/(df+1)) + 1  (스무딩)
    import math
    N = len(chunks)
    df: Dict[str, int] = {}
    for ch in chunks:
        for kw in set(ch.get('keywords', [])):
            df[kw] = df.get(kw, 0) + 1
    return {kw: math.log((N + 1) / (cnt + 1)) + 1 for kw, cnt in df.items()}

# -------------------- 저장된 인덱스 로더  --------------------
def load_saved_category(cat: str,
                       parent_dir: str = PARENT_DIR) -> Dict[str, Any]:
    # 입력: cat("전체" 또는 카테고리명), parent_dir(인덱스 루트)
    # 동작: FAISS 인덱스/청크 로드 + embedding_text 배열(docs) 구성 + IDF 사전 계산
    # 반환: 검색에 필요한 구성요소 dict(model/index/chunks/docs/IDF/idx_dir)
    if cat == "전체":
        save_dir = os.path.join(parent_dir, "idx_all")
    else:
        save_dir = os.path.join(parent_dir, f"idx_{cat}")
    ip = os.path.join(save_dir, "index.faiss")
    jp = os.path.join(save_dir, "chunks.json")
    if not (os.path.isfile(ip) and os.path.isfile(jp)):
        raise FileNotFoundError(f"[{cat}] 인덱스 파일이 없습니다: {save_dir}")
    index = faiss.read_index(ip)
    with open(jp, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    docs = [c.get("embedding_text", "") for c in chunks]
    cfg = {
        "model": embed_model,
        "index": index,
        "chunks": chunks,
        "docs": docs,
        "IDF": compute_idf(chunks),
        "idx_dir": save_dir,  # 본문 로드를 위한 디렉터리 보관
    }
    return cfg

# 카테고리 초기화
def init_rag_from_saved(parent_dir: str) -> Dict[str, Dict[str, Any]]:
    # 목적: 디렉터리 스캔으로 카테고리 구성 + 번호 매핑 + 카테고리 임베딩 사전계산 + 인덱스 로드
    # 반환: 카테고리명 → 인덱스 구성요소 dict
    global desired_categories, mapping, category_embeddings
    cat_indices: Dict[str, Dict[str, Any]] = {}

    # 디렉터리 스캔으로 카테고리 채우기
    desired_categories = _scan_categories(parent_dir)
    mapping = {str(i): cat for i, cat in enumerate(["전체"] + desired_categories)}
    if desired_categories:
        category_embeddings = embed_model.encode(desired_categories, normalize_embeddings=True, convert_to_tensor=False)
    else:
        category_embeddings = np.zeros((0, 1), dtype=np.float32)

    # 개별 카테고리 로드(있을 때만)
    for cat in desired_categories:
        try:
            cat_indices[cat] = load_saved_category(cat, parent_dir)
        except FileNotFoundError:
            pass
    # 전체(idx_all) 로드 시도
    try:
        cat_indices["전체"] = load_saved_category("전체", parent_dir)
    except FileNotFoundError:
        if desired_categories:
            # 전체가 없으면 첫 카테고리를 대체로 사용
            cat_indices["전체"] = cat_indices.get(desired_categories[0])
    return cat_indices

# -------------------- 카테고리 추천  --------------------
def parse_category_input(inp: str) -> Optional[str]:
    # 목적: 사용자가 번호/이름 혼합 입력 시 유연 매핑
    # 입력 예: "3", "3식품", "HACCP", "전체"
    inp = inp.strip()
    if inp in mapping:
        return mapping[inp]
    m = re.match(r"(\d+)(.+)", inp)
    if m and m.group(1) in mapping:
        return mapping[m.group(1)]
    if inp in mapping.values():
        return inp
    return None
#수정
def parse_multi_category_input(inp: str) -> List[str]:
    """
    쉼표/공백으로 여러 개 입력 지원.
    - '0' 또는 '전체'가 포함되면 ['전체']만 반환.
    - 중복 제거, 입력 순서 유지.
    """
    if not inp:
        return []
    tokens = [t for t in re.split(r'[,\s]+', inp.strip()) if t]
    if any(t == '0' or t == '전체' for t in tokens):
        return ['전체']

    seen, out = set(), []
    for t in tokens:
        cat = parse_category_input(t)  # 기존 단일 파서 활용
        if cat and cat not in seen:
            seen.add(cat)
            out.append(cat)
    return out

# def classify_category(query: str, sem_threshold: float = 0.4) -> List[str]:
#     # 목적: 질의와 카테고리명 임베딩 간 내적 유사도 기준으로 상위 후보 추천
#     # 파라미터: sem_threshold — 유사도 임계(0~1). 낮출수록 후보 다수 추천.
#     # 반환: 추천 카테고리명 리스트(최대 2개)
#     q_vec = embed_model.encode([query], normalize_embeddings=True)[0]
#     sims = np.dot(category_embeddings, q_vec)
#     idx = np.where(sims >= sem_threshold)[0]
#     if idx.size == 0:
#         return []
#     idx = idx[np.argsort(sims[idx])[::-1]][:2]
#     return [desired_categories[i] for i in idx]

def classify_category(query: str, sem_threshold: float = 0.4) -> List[str]:
    """
    목적: 임베딩 유사도 대신 'RAG 검색 점수'로 상위 후보 카테고리(최대 2개) 추천
    - main()에서 init_rag_from_saved(...)가 만든 cat_indices를 globals()로 참조
    - '전체' 인덱스는 라우팅 대상에서 제외
    - 각 카테고리에서 retrieve_docs(..., top_k=3)의 최고 score를 대표 점수로 사용
    - score>0만 후보로 채택, 점수 내림차순(동점 시 이름 오름차순) 정렬
    """
    indices = globals().get("cat_indices")
    if not isinstance(indices, dict):
        return []

    uq = preprocess_query(query)
    scored: list[tuple[str, float]] = []

    for cat, cfg in indices.items():
        if cat == "전체":
            continue
        try:
            results = retrieve_docs(
                uq,
                cfg["model"], cfg["index"], cfg["docs"], cfg["chunks"], cfg["IDF"],
                top_k=3
            )
            best_score = max((r.get("score", 0.0) for r in results), default=0.0)
        except Exception:
            best_score = 0.0

        if best_score > 0.0:
            scored.append((cat, best_score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [c for c, _ in scored[:2]]
#
# -------------------- 본문 로더 --------------------
# idx_dir 가리키는 폴더의 상위 폴더에 <제목>_chunks.json(풀 텍스트)가 있다고 가정.
# 현재 구현에서는 idx/chunks.json 안에 text가 이미 포함되어 있어 직접 사용.

def _title_from_idx_dir(idx_dir: str) -> str:
    # 목적: idx_<제목> 디렉터리명에서 제목만 추출
    return os.path.basename(idx_dir)[4:]

def resolve_text_for_chunk(candidate: Dict[str, Any], idx_dir: str) -> Optional[str]:
    # 목적: 프롬프트용 본문 텍스트 확보 우선순위
    # 1) candidate['text']가 list/dict면 JSON 직렬화하여 사용(원본 보존)
    # 2) 문자열이면 그대로 사용
    # 3) 모두 없으면 embedding_text로 대체(최후 수단)
    tx = candidate.get("text")
    if isinstance(tx, (list, dict)):
        return json.dumps(tx, ensure_ascii=False)
    if isinstance(tx, str) and tx.strip():
        return tx
    # 안전장치: 비어있으면 embedding_text로 대체
    return candidate.get("embedding_text", "")

# -------------------- 검색: 하이브리드(semantic + IDF) --------------------
def retrieve_docs(query: str, model: SentenceTransformer, index, docs, chunks, IDF,
                  alpha: float = 0.9, top_k: int = 5, idx_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    # 목적: 임베딩 유사도(sem)와 키워드 IDF 점수(ks)를 결합해 최종 상위 top_k 청크 선택
    # 스코어: score = alpha * sem + (1 - alpha) * ks, 두 점수는 [0,1]로 정규화 가정
    # 파라미터:
    #   - docs : embedding_text 리스트(인덱스 순서와 1:1)
    #   - chunks: 청크 원본 리스트(검색 후 결과 매핑/본문 주입에 사용)
    #   - IDF : 키워드별 idf 가중치 사전(청크 keywords로 계산)
    #   - idx_dir: 선택 시 본문(text) 주입을 위해 디렉터리 힌트 전달
    # 반환: 상위 결과 청크 리스트(본문 text 주입 완료)
    qv = model.encode([query], normalize_embeddings=True)[0]
    dists, I = index.search(np.array([qv]), len(docs))
    dists, I = dists[0], I[0]
    sem = np.clip(dists, 0, 1)
    qk = set(get_keywords(query))
    ks = np.array([
        sum(IDF.get(kw, 1.0) for kw in (qk & set(chunks[i].get('keywords', [])))) for i in I
    ], dtype=np.float32)
    if ks.max() > 0:
        ks /= (ks.max() + 1e-6)
    scores = alpha * sem + (1 - alpha) * ks
    top_indices = I[np.argsort(scores)[::-1][:top_k]]
    
    # 점수를 청크에 추가
    selected = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk['score'] = round(float(scores[np.where(I == idx)[0][0]]), 2)  # 해당 청크의 점수 추가
        selected.append(chunk)
   
    # 매뉴얼은 모든 경우에 embedding_text 사용 (평탄화된 마크다운)
    for ch in selected:
        if 'embedding_text' in ch:
            ch['text'] = ch['embedding_text']
    
    return selected

# -------------------- 프롬프트 (매뉴얼 지침) --------------------
def build_chatml_prompt(question: str, results: List[Dict[str, Any]], max_blocks: int = 2, wrap_width: int = 80) -> str:
    # 목적:  매뉴얼 문맥/지시사항에 맞춘 프롬프트 생성
    # 파라미터:
    #   - max_blocks: 프롬프트에 실을 상위 블록 개수(기본 2)
    #   - wrap_width: (미사용) 줄바꿈 폭 자리
    # 구성:
    #   - system: 답변 규칙(핵심 요약, 제공 블록 근거 명시)
    #   - user  : 질문 + 선택 블록(context)
    blocks = []
    for i, ch in enumerate(results[:max_blocks], start=1):
        title = ch.get("source", "")
        body = ch.get("text", "")
        blocks.append(f"{i}. {title}\n{body}")
    context = "\n\n".join(blocks)
    prompt = f"""<|im_start|>system
당신은 HACCP/식품안전 매뉴얼에 정통한 전문가입니다. 아래에는 사용자 질문과 매뉴얼 발췌 블록이 주어집니다.
규칙:
1) 질문과 직접 관련된 핵심만 요약하세요.
2) 오직 제공된 블록에 근거해 명확히 답변하세요. 

[출력 형식 예시]
1. 요약:
(여기에 핵심 요약)
2. 답변:
(여기에 답변 내용)
<|im_end|>
<|im_start|>user
[질문]
{question}

[블록]
{context}
<|im_end|>
<|im_start|>assistant
""".strip()
    return prompt

# -------------------- LLM (vLLM 서버 사용) --------------------
def load_llm(model_id: str = LLM_MODEL_ID):
    """
    vLLM serving framework 활용을 위해 수정됨
    
    기존과 달리 모델을 직접 로드하지 않고 프로세서만 로드
    실제 모델은 vLLM 서버에서 서빙됨
    """
    processor = AutoProcessor.from_pretrained(model_id)
    return None, processor

def generate_llm_response(model, processor, conversation, max_new_tokens: int = 1024):
    """
    vLLM serving framework 활용을 위해 수정됨
    
    모델에 직접 입력하지 않고 vLLM client로 localhost:8400/v1에 요청
    """
    # ChatML 템플릿 적용 (토크나이즈는 서버에서 처리)
    rendered_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    # 토큰 길이 계산 (max_tokens 설정용)
    input_len = len(processor.tokenizer(rendered_prompt)["input_ids"])
    
    # max_tokens 안전하게 계산
    MAX_CONTEXT_LENGTH = 4096
    RESERVED_OUTPUT_TOKENS = 1024
    MIN_OUTPUT_TOKENS = 100
    
    available_tokens = MAX_CONTEXT_LENGTH - input_len
    max_tokens = min(available_tokens, RESERVED_OUTPUT_TOKENS)
    
    if max_tokens < MIN_OUTPUT_TOKENS:
        # 입력이 너무 길어서 출력 공간이 부족한 경우
        print(f"⚠️ 경고: 입력이 너무 깁니다 ({input_len} tokens). 최소 출력 토큰 보장 불가.")
        max_tokens = max(1, available_tokens)  # 최소한 1 토큰은 보장

    # vLLM 서버에 요청
    client = get_vllm_varco_client()
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=conversation,
            max_tokens=max_tokens
        )
        elapsed = time.time() - start
        output_text = response.choices[0].message.content
        return {"rendered_prompt": rendered_prompt, "output": output_text.strip(), "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ vLLM 서버 요청 실패: {e}")
        return {"rendered_prompt": rendered_prompt, "output": f"[오류] vLLM 서버 응답 실패: {str(e)}", "elapsed": elapsed}

# -------------------- Triplet 라벨링  --------------------
def save_group_jsonl(query: str,
                     positives: List[str],
                     negatives: List[str],
                     pos_sources: Optional[List[str]] = None,
                     neg_sources: Optional[List[str]] = None,
                     extra_meta: Optional[dict] = None,
                     out_path: str = TRIPLET_JSONL):
    # 목적: 한 번의 인터랙션에서 선택된 good/bad 본문들을 하나의 JSONL 레코드로 저장
    # 필드:
    #  - query/positives/negatives
    #  - meta.timestamp / pos_sources / neg_sources / extra_meta(선택)
    def _clean(s: str) -> str:
        s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        return re.sub(r"\s+", " ", s).strip()
    rec = {
        "query": _clean(query),
        "positives": [_clean(p) for p in positives if p and p.strip()],
        "negatives": [_clean(n) for n in negatives if n and n.strip()],
        "meta": {"timestamp": datetime.now().isoformat(timespec="seconds")}
    }
    if pos_sources: rec["meta"]["pos_sources"] = [_clean(x) for x in pos_sources]
    if neg_sources: rec["meta"]["neg_sources"] = [_clean(x) for x in neg_sources]
    if extra_meta:  rec["meta"].update(extra_meta)
    # 폴더 자동 생성
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# 사용자 선택형 라벨 인터랙션(매뉴얼과 동일 UX 유지)
def interactive_label_group(question: str,
                            candidates: List[Dict[str, Any]],
                            llm_used_n: int = 2) -> bool:
    # 목적: CLI에서 다중 good/bad 후보 번호를 받아 save_group_jsonl에 기록
    # 동작:
    #  - 후보 목록 source를 출력하여 사람이 빠르게 식별
    #  - "good 번호", "bad 번호"를 콤마로 입력받아 중복/범위 체크
    #  - 선택이 유효하면 positives/negatives를 저장
    if not candidates:
        print("⚠️ 라벨링할 후보가 없습니다.")
        return False
    print("\n===== [라벨링] 다중 good/bad 선택 =====")
    print(f"[질문]\n{question}\n")
    print("[후보 목록]")
    for i, ch in enumerate(candidates, start=1):
        title = ch.get("source", "")
        print(f"{i}) {title}\n")
    def parse_indices(s: str, N: int) -> List[int]:
        if not s: return []
        vals = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 1 <= v <= N: vals.append(v-1)
        return sorted(set(vals))
    print("예) good: 1,3   bad: 2,5   (비우면 건너뜀)")
    pos_idx = parse_indices(input("good 번호: ").strip(), len(candidates))
    neg_idx = parse_indices(input("bad  번호: ").strip(), len(candidates))
    if not pos_idx and not neg_idx:
        print("➡️ 입력이 없어 저장하지 않습니다.")
        return False
    overlap = set(pos_idx) & set(neg_idx)
    if overlap:
        print(f"⚠️ 겹치는 번호 제외: {[i+1 for i in overlap]}")
        pos_idx = [i for i in pos_idx if i not in overlap]
        neg_idx = [i for i in neg_idx if i not in overlap]
    positives = [candidates[i].get("text", candidates[i].get("embedding_text", "")) for i in pos_idx]
    negatives = [candidates[i].get("text", candidates[i].get("embedding_text", "")) for i in neg_idx]
    pos_srcs  = [candidates[i].get("source", "") for i in pos_idx]
    neg_srcs  = [candidates[i].get("source", "") for i in neg_idx]
    if not positives and not negatives:
        print("⚠️ 유효한 선택이 없어 저장하지 않습니다.")
        return False
    save_group_jsonl(question, positives, negatives, pos_srcs, neg_srcs,
                     extra_meta={"retrieved_topk": len(candidates), "llm_used_topn": llm_used_n})
    print(f"✅ 그룹 저장 완료 → {TRIPLET_JSONL}")
    return True

# -------------------- 메인  --------------------
def main():
    # 0) 단독 실행 모드일 때 모델 로드
    _init_models_if_needed()
    
    # 1) LLM 로드 
    # - 실패 시 검색/라벨링만 수행(USE_LLM=False)
    try:
        model_llm, processor = load_llm(LLM_MODEL_ID)
        USE_LLM = True
        print("[INFO] LLM 로드 완료")
    except Exception as e:
        print(f"[경고] LLM 로드 실패 → 검색/라벨링만 사용: {e}")
        model_llm = processor = None
        USE_LLM = False

    # 2) 인덱스 루트 지정
    # - 디렉터리 스캔 → mapping/desired_categories/category_embeddings 구성
    # - 각 카테고리/전체 인덱스 로드
    parent_dir = PARENT_DIR
    cat_indices = init_rag_from_saved(parent_dir)
#수정
    globals()['cat_indices'] = cat_indices
#
    # 3) 간단 세션 로그
    # - 이후 질의/카테고리/프롬프트/응답/소요시간 등을 append하여 회고 가능
    log_path = SAVE_LOG
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n\n===== VARCO-VISION + RAG 세션 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    # 4) 대화형 루프
    # - exit 입력 시 종료
    # - 추천 카테고리 표출 → 번호/이름 입력 파싱 → 없으면 추천/전체로 폴백
    # - retrieve_docs로 top-5 검색 → 상위 2개로 프롬프트 구성 → LLM 생성(가능 시)
    # - 라벨링 단계로 이어짐
    while True:
        q = input("\n질문을 입력하세요 (종료: exit) >> ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            print("종료합니다.")
            break

        print("\n=== 전체 카테고리 목록 ===")
        for i in range(0, len(desired_categories)+1):
            print(f"{i}) {mapping.get(str(i), '')}", end="    ")
            if (i % 6) == 5:
                print()
        print("\n")

        # 추천 후보
        candidates_cat = classify_category(q)
        print("추천 카테고리:")
        for i, cat in enumerate(desired_categories, start=1):
            if cat in candidates_cat:
                print(f"{i}) {cat}")
        print("0) 전체")

        # choice = input("번호/이름 입력(미입력=추천→전체/첫번째) >> ").strip()
        # sel = parse_category_input(choice) if choice else None
        # if not sel:
        #     sel = candidates_cat[0] if candidates_cat else "전체"
        # if sel not in cat_indices or not cat_indices.get(sel):
        #     print(f"[경고] '{sel}' 인덱스가 없어 '전체'로 대체합니다.")
        #     sel = "전체"
#수정
        choice = input("번호/이름 (쉼표로 여러 개, 미입력=추천 상위/전체) >> ").strip()

        if choice:
            sel_list = parse_multi_category_input(choice)
        else:
            # 미입력: 추천이 있으면 최대 2개, 없으면 '전체'
            sel_list = candidates_cat[:2] if candidates_cat else ['전체']

        # 로드된 인덱스만 유지
        sel_list = [c for c in sel_list if c in cat_indices and cat_indices.get(c)]
        if not sel_list:
            sel_list = ['전체']
#

        # cfg = cat_indices[sel]
        # uq = preprocess_query(q)

        # # 검색(매뉴얼 하이브리드 + 본문 주입)
        # results_top5 = retrieve_docs(uq, cfg["model"], cfg["index"], cfg["docs"], cfg["chunks"], cfg["IDF"],
        #                              alpha=0.9, top_k=5, idx_dir=cfg.get("idx_dir"))
        # results_for_prompt = results_top5[:2]  # 매뉴얼 프롬프트 기준
#수정

        uq = preprocess_query(q)

        def _dedup_by_source(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """source 기준 중복 제거(첫 등장 유지). 필요 없으면 제거 가능."""
            seen, out = set(), []
            for r in rows:
                key = r.get("source")
                if key in seen:
                    continue
                seen.add(key)
                out.append(r)
            return out

        if sel_list == ['전체']:
            cfg = cat_indices['전체']
            results_top5 = retrieve_docs(
                uq, cfg["model"], cfg["index"], cfg["docs"], cfg["chunks"], cfg["IDF"],
                alpha=0.9, top_k=5, idx_dir=cfg.get("idx_dir")
            )
        else:
            aggregated: List[Dict[str, Any]] = []
            per_cat_k = 3  # 카테고리별 몇 개씩 뽑을지 (원하면 조절)
            for sel in sel_list:
                cfg = cat_indices[sel]
                part = retrieve_docs(
                    uq, cfg["model"], cfg["index"], cfg["docs"], cfg["chunks"], cfg["IDF"],
                    alpha=0.9, top_k=per_cat_k, idx_dir=cfg.get("idx_dir")
                )
                for r in part:
                    r["category"] = sel  # 디버깅/로그용 태그
                aggregated.extend(part)

            aggregated = _dedup_by_source(aggregated)
            aggregated.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            results_top5 = aggregated[:5]

        results_for_prompt = results_top5[:2]

        # print(f"\n== 검색 결과 (선택: {', '.join(sel_list)}) ==")
        # for r in results_for_prompt:
        #     cat_tag = f"[{r.get('category')}]" if r.get('category') else ""
        #     print(f"{cat_tag} {r.get('source','')}")


        # # 간단 결과 표시/로그
        # print(f"\n== {sel} 검색 결과 ==")
        # for r in results_for_prompt:
        #     print(r.get("source", ""))
#수정
        print(f"\n== 검색 결과 (선택: {', '.join(sel_list)}) ==")
        for r in results_for_prompt:
            cat_tag = f"[{r.get('category')}]" if r.get('category') else ""
            print(f"{cat_tag} {r.get('source','')}")

        # 로그도 sel → sel_list로 교체
        log.write(f"\n👤 질문: {q}\n")
        log.write(f"📂 선택 카테고리: {', '.join(sel_list)}\n")
#
        # LLM 응답(텍스트만)
        if USE_LLM:
            chatml_prompt = build_chatml_prompt(q, results_for_prompt, max_blocks=2, wrap_width=80)
            conversation = [{"role": "user", "content": [{"type": "text", "text": chatml_prompt}]}]
            gen = generate_llm_response(model_llm, processor, conversation, max_new_tokens=1024)
            print(f"\n✅ LLM 응답 (⏱ {gen['elapsed']:.2f}s)\n")
            print(gen["output"])
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n👤 질문: {q}\n")
                log.write(f"📂 선택 카테고리: {sel}\n")
                for r in results_for_prompt:
                    log.write(f" - {r.get('source','')}\n")
                log.write("\n--- Rendered Prompt ---\n")
                log.write(gen["rendered_prompt"] + "\n")
                log.write(f"\n🤖 VARCO 응답:\n{gen['output']}\n")
                log.write(f"⏱ 소요시간: {gen['elapsed']:.2f}초\n")
        else:
            print("\n[안내] LLM 비활성/로딩 실패로 답변 생성을 건너뜁니다.")

        input("\n(엔터를 누르면 라벨링 단계로 이동합니다) ")
        _ = interactive_label_group(q, results_top5, llm_used_n=min(2, len(results_top5)))

if __name__ == "__main__":
    main()
