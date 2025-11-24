# =============================================================
# run_rag_independent.py  (주석 상세판)
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

import os, re, json, time, textwrap, math, sys
from datetime import datetime
from typing import List, Dict
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
EMBED_MODEL_NAME = "dragonkue/BGE-m3-ko"                       # 임베딩 모델 이름
LLM_MODEL_ID     = "NCSOFT/VARCO-VISION-2.0-14B"               # LLM 모델 ID

# 날짜별 인덱스 로드 (indexes/law/YYYY-MM-DD 구조)
# 고정: 실제 인덱스가 존재하는 날짜로 설정
INDEX_DATE = "2025-11-11"  # 인덱스 생성 날짜 (고정)
PARENT_DIR       = str(SCRIPT_DIR.parent / "00_data" / "input" / "indexes" / "law" / INDEX_DATE)  # 인덱스 루트 디렉터리
SAVE_LOG         = str(SCRIPT_DIR.parent / "00_data" / "output" / "logs" / "result.txt")  # (옵션) 검색 결과 저장 파일 경로
TRIPLET_JSONL    = str(SCRIPT_DIR.parent / "00_data" / "output" / "training_data" / "triplets_group_bgem3.jsonl")  # 라벨 누적 저장 경로(JSONL)

# -------------------- 전역 모델 --------------------
# GPU 메모리 캐시 비우기(특히 재실행 시 잔여 캐시로 인한 OOM 방지용)
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

# -------------------- 유틸 (원본과 동일) --------------------
# 한국어 조사를 단순 접미 제거(키워드 정규화).
#  - 목적: "식품" vs "식품을" 같은 변형을 통일하여 키워드 일치율 향상.
#  - 주의: 형태소 분석이 아닌 단순 접미 제거이므로 일부 과도 제거나 누락 가능.
def remove_josa(word):
    for j in ['은','는','이','가','을','를','에','의','와','과','도','로','으로','에서','에게','한테','부터','까지','만','보다','처럼','조차','마저']:
        if word.endswith(j):
            return word[:-len(j)]
    return word

# 텍스트에서 키워드 후보 추출
#  - ratio로 텍스트 길이 대비 추출 개수 대략 조절(안정 범위 min_k~max_k)
#  - 예외 발생 시 빈 리스트 반환(파이프라인 중단 방지)
#  - 중복 제거(set) + 조사 제거 후 리스트화
def get_keywords(text, ratio=1, max_k=30, min_k=5):
    est_k = int(len(text) * ratio / 10)
    top_k = max(min_k, min(est_k, max_k))
    try:
        kws = kw_model.extract_keywords(text, top_n=top_k)
    except Exception:
        return []
    return list({remove_josa(k[0]) for k in kws if isinstance(k, (list, tuple)) and k and isinstance(k[0], str)})

# 질의 문자열 정규화
#  - "제 47" → "47" (숫자만 남김)
#  - "89조 의 2" → "89조의2" (표기 일관화)
#  - 한글/영문/숫자/공백만 남기고 나머지 제거
def preprocess_query(q):
    q = re.sub(r'제\s*(\d+)', r'\1', q)
    q = re.sub(r'(\d+)조\s*의\s*(\d+)', r'\1조의\2', q)
    q = re.sub(r'[^\w가-힣\s]', '', q)
    return q.strip()

# 소스 문자열(표제/메타) 정리
#  - (제n호), (YYYYMMDD) 패턴 제거 → 사용자 표시용 간소화
#  - 다중 공백 → 단일 공백
def clean_source(src: str) -> str:
    src = re.sub(r'\(제\d+호\)', '', src)
    src = re.sub(r'\(\d{8}\)', '', src)
    return re.sub(r'\s+', ' ', src).strip()


# -------------------- 카테고리/매핑 (원본과 동일) --------------------
# 검색 범위를 문서 카테고리별로 분리하여, 사용자 선택 또는 자동 추천에 활용
desired_categories = [
    "가축전염병 예방법","건강기능식품에 관한 법률","농약관리법",
    "먹는물관리법","사료관리법","수입식품안전관리","식품위생법",
    "식품ㆍ의약품분야 시험ㆍ검사","축산물 위생관리법","한국식품안전관리인증원의 설립 및 운영에 관한 법률"
]
# 숫자 선택(0=전체)과 한글 이름 입력을 모두 허용하기 위한 매핑 테이블
mapping = {str(i): cat for i, cat in enumerate(["전체"] + desired_categories)}

# 카테고리명 문장 임베딩 (주의: classify_category에서 사용하지 않음)
# 실제로는 RAG 검색 점수 기반으로 카테고리 추천 (186번 라인 참고)
# category_embeddings = embed_model.encode(
#     desired_categories, normalize_embeddings=True, convert_to_tensor=False
# )

# 사용자 입력을 숫자/문자 조합 모두 인식(예: "1", "1식품위생법", "식품위생법")
def parse_category_input(inp):
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
    - '0' 또는 '전체'가 하나라도 포함되면 ['전체']만 반환.
    - 중복 제거, 입력 순서 유지.
    """
    if not inp:
        return []
    tokens = [t for t in re.split(r'[,\s]+', inp.strip()) if t]
    if any(t == '0' or t == '전체' for t in tokens):
        return ['전체']

    seen, out = set(), []
    for t in tokens:
        cat = parse_category_input(t)  # 기존 단일 파서 재사용
        if cat and cat not in seen:
            seen.add(cat)
            out.append(cat)
    return out

# def classify_category(query: str, sem_threshold: float = 0.4) -> List[str]:
#     q_vec = embed_model.encode([query], normalize_embeddings=True)[0]
#     sims = np.dot(category_embeddings, q_vec)
#     idx = np.where(sims >= sem_threshold)[0]
#     if idx.size == 0:
#         return []
#     idx = idx[np.argsort(sims[idx])[::-1]][:2]

#     return [desired_categories[i] for i in idx]

def classify_category(query: str, sem_threshold: float = 0.0) -> List[str]:
    """
    기존 임베딩 기반 추천 대신,
    실제 RAG 검색 점수(retrieve_docs 결과 score)를 사용하여 상위 2개 카테고리를 추천한다.
    """
    # init_rag_from_saved()에서 만든 cat_indices를 그대로 사용
    indices = globals().get("cat_indices")
    if not isinstance(indices, dict):
        return []

    uq = preprocess_query(query)
    scored: list[tuple[str, float]] = []

    for cat, cfg in indices.items():
        if cat == "전체":
            continue  # 전체 인덱스는 routing 대상에서 제외

        try:
            results = retrieve_docs(
                uq,
                cfg["model"], cfg["index"], cfg["docs"], cfg["chunks"], cfg["IDF"],
                top_k=3,        # 각 카테고리에서 top-3 청크 검색
            )
            best_score = max((r.get("score", 0.0) for r in results), default=0.0)
        except Exception:
            best_score = 0.0

        if best_score > 0.0:
            scored.append((cat, best_score))

    # 점수 내림차순, 동점이면 이름 오름차순
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [c for c, _ in scored[:2]]


# -------------------- 저장된 인덱스 로더 (인덱스 생성 X, 읽기만) --------------------
# 지정 카테고리의 디스크 인덱스 묶음(chunks/docs/vectors/index) 로드
#  - 파일 존재성 검사 후, FAISS 인덱스 읽기 + 청크 로드
#  - enriched_text는 검색/프롬프트에 활용되는 본문 필드
#  - compute_idf(chunks)로 카테고리별 IDF 사전 준비(키워드 가중)
def load_saved_category(cat: str, parent_dir=PARENT_DIR) -> Dict:
    save_dir = os.path.join(parent_dir, f"idx_{cat}")
    jp = os.path.join(save_dir, "chunks.json")
    dp = os.path.join(save_dir, "docs.txt")
    vp = os.path.join(save_dir, "vectors.npy")
    ip = os.path.join(save_dir, "index.faiss")

    for p in (jp, dp, vp, ip):
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{cat}] 인덱스 파일이 없습니다: {p}")

    index = faiss.read_index(ip)
    with open(jp, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    docs = [c["enriched_text"] for c in chunks]
    return {"model": embed_model, "index": index, "chunks": chunks, "docs": docs, "IDF": compute_idf(chunks)}

# 모든 대상 카테고리에 대해 인덱스 시도 로드(없으면 건너뜀) + "전체"는 필수 로드
def init_rag_from_saved(parent_dir: str) -> Dict[str, Dict]:
    cat_indices = {}
    # 개별 카테고리 로드(없으면 pass로 무시)
    for cat in desired_categories:
        try:
            cat_indices[cat] = load_saved_category(cat, parent_dir)
        except FileNotFoundError:
            pass
    # 전체(통합 인덱스)는 반드시 존재해야 유효하게 동작
    cat_indices["전체"] = load_saved_category("전체", parent_dir)
    return cat_indices

# -------------------- IDF (원본과 동일) --------------------
# 청크에 미리 저장된 'keywords' 필드의 문서빈도(DF)로 IDF 가중치 산출
#  - IDF = log((N+1)/(df+1)) + 1 : 0 분모/분자를 피하고 과도한 값 방지
#  - 키워드 교집합에 대해 합산하여 쿼리-청크 키워드 점수 계산에 사용
def compute_idf(chunks):
    N = len(chunks)
    df = {}
    for ch in chunks:
        for kw in set(ch['keywords']):
            df[kw] = df.get(kw, 0) + 1
    return {kw: math.log((N + 1) / (cnt + 1)) + 1 for kw, cnt in df.items()}

# -------------------- 검색: 하이브리드(语义 + 키워드-IDF), 원본과 동일 --------------------
# 절차:
#  1) 의미 점수: 쿼리 임베딩 qv → FAISS로 전체 후보 검색(코사인 유사도 범위에 맞춰 clip)
#  2) 키워드 점수: 쿼리 키워드 vs 각 청크 키워드 교집합의 IDF 합
#  3) 두 점수 가중합( alpha*semantic + (1-alpha)*keyword ) 후 내림차순 Top-k
# 반환: 상위 청크(dict) 리스트
def retrieve_docs(query, model, index, docs, chunks, IDF, alpha=0.9, top_k=5):
    qv = model.encode([query], normalize_embeddings=True)[0]
    dists, I = index.search(np.array([qv]), len(docs))  # 모든 문서 대상 검색 후 정렬 인덱스 획득
    dists, I = dists[0], I[0]

    # 의미 점수: [0,1] 범위로 안전하게 클리핑
    sem = np.clip(dists, 0, 1)

    # 키워드 점수: 쿼리→키워드 추출 → 각 후보 청크의 키워드와 교집합 IDF 합
    qk = set(get_keywords(query))
    ks = np.array([sum(IDF.get(kw, 1.0) for kw in (qk & set(chunks[i]['keywords']))) for i in I], dtype=np.float32)
    if ks.max() > 0:
        ks /= (ks.max() + 1e-6)  # 0-1 정규화(분모 0 방지용 epsilon)

    # 하이브리드 최종 점수 및 Top-k 인덱스 선택
    scores = alpha * sem + (1 - alpha) * ks
    top_indices = I[np.argsort(scores)[::-1][:top_k]]
    
    # 점수를 청크에 추가하여 반환
    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk['score'] = round(float(scores[np.where(I == idx)[0][0]]), 2)  # 해당 청크의 점수 추가
        
        # text를 enriched_text로 교체 (법률은 enriched_text 사용)
        if 'enriched_text' in chunk:
            chunk['text'] = chunk['enriched_text']
        
        results.append(chunk)
    return results

# -------------------- 프롬프트 (원본 텍스트 그대로) --------------------
# 상위 N(기본 3)개 청크를 ChatML 포맷의 컨텍스트 블록으로 구성
#  - 본문은 가독성을 위해 wrap_width로 줄바꿈
#  - system 지침은 "오직 주어진 조문으로만 답변"하도록 한정
def build_chatml_prompt(question: str, results: list, max_blocks: int = 3, wrap_width: int = 80) -> str:
    context_blocks = []
    for i, chunk in enumerate(results[:max_blocks], start=1):
        title = clean_source(chunk["source"])  # 소스 표제 정리
        body = chunk["text"]
        wrapped_body = "\n".join(textwrap.wrap(body, width=wrap_width))
        context_blocks.append(f"{i}. {title}\n{wrapped_body}")
    context = "\n\n".join(context_blocks)

    prompt = f"""<|im_start|>system
당신은 한국 법령에 정통한 전문 법률 어시스턴트입니다. 아래에는 사용자 질문과 법령에서 발췌한 조문들이 주어집니다.
당신의 임무는 세 가지입니다:
1. 조문들을 읽고 질문과 관련 있는 내용만 골라 핵심을 요약하세요. 
2. 오직 주어진 조문만을 바탕으로 명확하고 신뢰도 높은 답변을 제공하세요.
3. 만약 모든 조문이 질문과 관련 없다면, 관련된 조문이 없음을 밝히고 답변을 유보하세요.
[출력 형식 예시]
1. 요약:
(여기에 핵심 요약)
2. 답변:
(여기에 답변 내용 — 해당 조문 번호 언급 포함)
<|im_end|>
<|im_start|>user
[사용자 질문]
{question}
[조문 내용]
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

# Chat 템플릿 적용 → vLLM 서버 호출 → 응답 반환
def generate_llm_response(model, processor, conversation, max_new_tokens=1024):
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

# -------------------- Triplet 라벨링 (원본과 동일) --------------------
# (질문, positives[], negatives[]) 한 건을 JSONL로 한 줄 저장
#  - 개행/탭 등 공백류는 단일 공백으로 정규화하여 저장
def save_group_jsonl(query: str,
                     positives: List[str],
                     negatives: List[str],
                     pos_sources: List[str] | None = None,
                     neg_sources: List[str] | None = None,
                     extra_meta: dict | None = None,
                     out_path: str = TRIPLET_JSONL):
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

# CLI에서 후보들을 보고 good/bad를 복수 선택 → JSONL 저장
#  - 입력 예: good: "1,3" / bad: "2,5"
#  - good/bad 중복 선택 시 겹치는 항목은 제외 처리
def interactive_label_group(question: str,
                            candidates: List[dict],
                            llm_used_n: int = 3) -> bool:
    if not candidates:
        print("⚠️ 라벨링할 후보가 없습니다.")
        return False

    print("\n===== [라벨링] 다중 good/bad 선택 =====")
    print(f"[질문]\n{question}\n")
    print("[후보 목록] (번호 참고)")
    for i, ch in enumerate(candidates, start=1):
        title = clean_source(ch["source"])
        body = ch["enriched_text"].strip()
        preview = body  # 미리보기(줄임 처리 없음)
        print(f"{i}) {title}\n   {preview}\n")

    print("예) good: 1,3   bad: 2,5   (비우면 건너뜀)")
    good_in = input("good(정답) 번호(콤마구분): ").strip()
    bad_in  = input("bad(오답)  번호(콤마구분): ").strip()

    def parse_indices(s: str, N: int) -> List[int]:
        if not s: return []
        vals = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 1 <= v <= N:
                    vals.append(v-1)
        return sorted(set(vals))

    pos_idx = parse_indices(good_in, len(candidates))
    neg_idx = parse_indices(bad_in, len(candidates))
    if not pos_idx and not neg_idx:
        print("➡️ 입력이 없어 저장하지 않습니다.")
        return False

    overlap = set(pos_idx) & set(neg_idx)
    if overlap:
        print(f"⚠️ 겹치는 번호 제외: {[i+1 for i in overlap]}")
        pos_idx = [i for i in pos_idx if i not in overlap]
        neg_idx = [i for i in neg_idx if i not in overlap]

    positives = [candidates[i]["enriched_text"] for i in pos_idx]
    negatives = [candidates[i]["enriched_text"] for i in neg_idx]
    pos_srcs  = [clean_source(candidates[i]["source"]) for i in pos_idx]
    neg_srcs  = [clean_source(candidates[i]["source"]) for i in neg_idx]

    if not positives and not negatives:
        print("⚠️ 유효한 선택이 없어 저장하지 않습니다.")
        return False

    meta = {
        "retrieved_topk": len(candidates),
        "llm_used_topn": llm_used_n,
        "pos_indices_1based": [i+1 for i in pos_idx],
        "neg_indices_1based": [i+1 for i in neg_idx]
    }
    save_group_jsonl(question, positives, negatives, pos_srcs, neg_srcs, meta)
    print(f"✅ 그룹 저장 완료 → {TRIPLET_JSONL}")
    return True

# -------------------- 메인 (인덱스는 읽기만) --------------------
# 실행 진입점: LLM 로드 → 인덱스 로드 → 질의 루프(검색/답변/라벨링)
def main():
    # 0) 단독 실행 모드일 때 모델 로드
    _init_models_if_needed()
    
    # 1) LLM 로드 (원본 동일)
    try:
        model_llm, processor = load_llm(LLM_MODEL_ID)
        USE_LLM = True
        print("[INFO] LLM 로드 완료")
    except Exception as e:
        # 모델 다운로드/메모리/네트워크 문제 등으로 실패 가능
        print(f"[경고] LLM 로드 실패 → 검색/라벨링만 사용: {e}")
        model_llm = processor = None
        USE_LLM = False

    # 2) 저장된 인덱스 로드 (인덱스 생성/전처리 없음)
    parent_dir = PARENT_DIR
    cat_indices = init_rag_from_saved(parent_dir)
    #수정
    globals()["cat_indices"] = cat_indices
    #
    # 간단 채팅 로그 파일(세션 시작 헤더만 기록)
    log_path = "chat_log.txt"
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n\n===== VARCO-VISION + RAG 세션 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    # 3) 대화형 루프: exit 입력 전까지 반복
    while True:
        q = input("\n질문을 입력하세요 (종료: exit) >> ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            print("종료합니다.")
            break

        # 카테고리 목록 표시(숫자 선택 용이)
        print("\n=== 전체 카테고리 목록 ===")
        for i in range(0, len(desired_categories)+1):
            print(f"{i}) {mapping[str(i)]}", end="    ")
            if (i % 6) == 5:
                print()
        print("\n")

        # 질의 기반 추천 카테고리 보여주기(키워드/의미 기반)
        candidates_cat = classify_category(q)
        print("=== 추천 카테고리 ===")
        for i, cat in enumerate(desired_categories, start=1):
            if cat in candidates_cat:
                print(f"{i}) {cat}")
        print("0) 전체")

        # 직접 선택(미입력 시 추천 1순위 또는 전체로 대체)
        
        # choice = input("번호/이름 입력(미입력=추천→전체/첫번째) >> ").strip()
        # sel = parse_category_input(choice) if choice else None
        # if not sel:
        #     sel = candidates_cat[0] if candidates_cat else "전체"

        # if sel not in cat_indices:
        #     print(f"[경고] '{sel}' 인덱스가 없어 '전체'로 대체합니다.")
        #     sel = "전체"
# 수정
        choice = input("번호/이름 (쉼표로 여러 개, 미입력=추천 상위/전체) >> ").strip()

        if choice:
            sel_list = parse_multi_category_input(choice)
        else:
            # 미입력: 추천이 있으면 최대 2개, 없으면 '전체'
            sel_list = candidates_cat[:2] if candidates_cat else ['전체']

        # 로드된 인덱스만 유지
        sel_list = [c for c in sel_list if c in cat_indices]
        if not sel_list:
            sel_list = ['전체']

        print("\n=== 선택된 카테고리 ===")
        print(", ".join(sel_list))
#
        # --- 검색(하이브리드) ---
        # cfg = cat_indices[sel]
        # uq = preprocess_query(q)  # 숫자/문장 규격화
        # results_top5 = retrieve_docs(uq, cfg["model"], cfg["index"], cfg["docs"], cfg["chunks"], cfg["IDF"], top_k=5)
        # results_for_prompt = results_top5[:3]

        # print(f"\n== {sel} 검색 결과 ==")
        # for r in results_for_prompt:
        #     print(r["source"])  # 사용자에게 출처만 표시(본문 미표시)

# 수정
        uq = preprocess_query(q)

        def _dedup_by_source(rows: List[dict]) -> List[dict]:
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
                top_k=5
            )
        else:
            aggregated = []
            per_cat_k = 3  # 카테고리별 몇 개씩 뽑을지
            for sel in sel_list:
                cfg = cat_indices[sel]
                part = retrieve_docs(
                    uq, cfg["model"], cfg["index"], cfg["docs"], cfg["chunks"], cfg["IDF"],
                    top_k=per_cat_k
                )
                for r in part:
                    r["category"] = sel  # 디버깅/로그용 태그
                aggregated.extend(part)

            aggregated = _dedup_by_source(aggregated)
            aggregated.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            results_top5 = aggregated[:5]

        results_for_prompt = results_top5[:3]

        print(f"\n== 검색 결과 (선택: {', '.join(sel_list)}) ==")
        for r in results_for_prompt:
            cat_tag = f"[{r.get('category')}]" if r.get('category') else ""
            print(f"{cat_tag} {r['source']}")

# 수정

        # --- LLM 답변 (원본과 동일 프롬프트 / 텍스트만) ---
        if USE_LLM:
            chatml_prompt = build_chatml_prompt(q, results_for_prompt, max_blocks=3, wrap_width=80)
            conversation = [{"role": "user", "content": [{"type": "text", "text": chatml_prompt}]}]
            gen = generate_llm_response(model_llm, processor, conversation, max_new_tokens=1024)

            print(f"\n✅ LLM 응답 (⏱ {gen['elapsed']:.2f}s)\n")
            print(gen["output"])  # 모델 출력 그대로 표시

            # 세부 로그(프롬프트 원문/응답/소요시간)
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n👤 질문: {q}\n")
                # log.write(f"📂 선택 카테고리: {sel}\n")
                # 수정
                log.write(f"📂 선택 카테고리: {', '.join(sel_list)}\n")
                #
                for r in results_for_prompt:
                    log.write(f" - {r['source']}\n")
                log.write("\n--- Rendered Prompt ---\n")
                log.write(gen["rendered_prompt"] + "\n")
                log.write(f"\n🤖 VARCO 응답:\n{gen['output']}\n")
                log.write(f"⏱ 소요시간: {gen['elapsed']:.2f}초\n")
        else:
            print("\n[안내] LLM 비활성/로딩 실패로 답변 생성을 건너뜁니다.")

        # --- 라벨링 (원본 로직) ---
        input("\n(엔터를 누르면 라벨링 단계로 이동합니다) ")
        _ = interactive_label_group(q, results_top5, llm_used_n=len(results_for_prompt))

if __name__ == "__main__":
    main()
