# -*- coding: utf-8 -*-
"""
법령 RAG - 전처리 + 인덱스 빌드 (검색/생성 이전까지 전부)
- 입력: PDF 또는 TXT (base_dir 아래)
- 처리: PDF→TXT(필요시), 헤더/형식 정리, 조문 파싱, 키워드 추출, 카테고리별 인덱스 생성
- 출력: idx_singlelevel/idx_<카테고리>/{chunks.json, vectors.npy, index.faiss, docs.txt}
        + 전체(통합) 카테고리도 동일 구조 생성

"""

# ---------------------------------------
# 0) 설정 + 기본 import
# ---------------------------------------

# ====== 설정 (상대 경로 사용) ======
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR        = str(SCRIPT_DIR.parent.parent / "00_data" / "input" / "raw_law" / "법률파일원본(pdf)")  # 입력 PDF/TXT 루트

# 날짜별 인덱스 저장 (indexes/law/YYYY-MM-DD 구조)
TODAY_DATE = datetime.now().strftime("%Y-%m-%d")
IDX_PARENT_DIR  = str(SCRIPT_DIR.parent.parent / "00_data" / "input" / "indexes" / "law" / TODAY_DATE)  # 출력 인덱스 디렉터리
EMBED_MODEL_NAME = "dragonkue/BGE-m3-ko"  # BGE 임베딩 모델

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import fitz, faiss, torch
import os, re, json
import numpy as np

# ---------------------------------------
# 1) 전역 모델 초기화 (원본 유지)
# ---------------------------------------
# ---------------------------------------
# 전역 모델 초기화 (원본 유지)
# ---------------------------------------
torch.cuda.empty_cache()
kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")          # 키워드 추출용, CPU 사용
embed_model = SentenceTransformer(EMBED_MODEL_NAME)  # 문장 임베딩용(BGE-m3-ko)


# ---------------------------------------
# 2) 카테고리/매핑
# ---------------------------------------
# ---------------------------------------
# 카테고리/매핑
# ---------------------------------------
desired_categories = [
    "가축전염병 예방법","건강기능식품에 관한 법률","농약관리법",
    "먹는물관리법","사료관리법","수입식품안전관리","식품위생법",
    "식품ㆍ의약품분야 시험ㆍ검사","축산물 위생관리법","한국식품안전관리인증원의 설립 및 운영에 관한 법률",
    "개인정보 보호법", "근로기준법", "주택임대차보호법", "도로교통법", "소비자기본법", "성폭력범죄의 처벌 등에 관한 특례법",
    "장애인차별금지 및 권리구제 등에 관한 법률", "양성평등기본법"
]

# ---------------------------------------
# 3) PDF → TXT 변환
# ---------------------------------------
# ---------------------------------------
# PDF → TXT 변환 (PyMuPDF 사용)
# ---------------------------------------
import fitz


def convert_pdf_to_txt(pdf_path: str, txt_path: str, gap_ratio: float = 0.45):
    doc = fitz.open(pdf_path) 
    out_lines = []
    
    for page in doc:
        # 단순 텍스트 모드: 읽은 순서대로 이어붙임
        page_text = page.get_text("text")
        out_lines.append(page_text)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))


# ---------------------------------------
# 4) 전처리(헤더 제거/생략 확장/개행 보정)
# ---------------------------------------
# ---------------------------------------
# 법제처/국가법령정보센터 헤더 제거
# ---------------------------------------
def remove_header(raw: str) -> str:
    HEADER_RE = re.compile(r'(?m)^[ \t]*법제처.*?국가법령정보센터.*\n[^\n]*\n?')
    while True:
        new = HEADER_RE.sub('', raw)
        if new == raw:
            return new
        raw = new

# ---------------------------------------
# "제n조 및 제m조 생략" → 두 줄로 확장
# ---------------------------------------
def expand_multi_delete(raw: str) -> str:
    pattern = re.compile(r'(?m)^\s*(제\s*(\d+)조)\s*및\s*(제\s*(\d+)조)\s*(생략)\s*$')
    return pattern.sub(lambda m: f"{m.group(1)} 생략\n{m.group(3)} 생략", raw)

# ---------------------------------------
# 법령 텍스트 전처리  ←★★ 개행 처리 순서/정규식만 수정
# ---------------------------------------
def preprocess_text(raw):
    #페이지번호, 법제처, 국가법령정보센터 헤더 제거
    raw = remove_header(raw) 
    # 조문 삭제·생략 구문 확장
    raw = expand_multi_delete(raw) 
    #“제N장”으로 시작하는 줄 전체 제거
    raw = re.sub(r'(?m)^\s*제\s*\d+\s*장[^\n]*\n?', '', raw)  

    #법령 내용 정리
    text = re.sub(r'[·※▶★■●◦◇ㆍ]', '', raw) # 특수문자 제거
    text = re.sub(r'-\n', '', text) # 하이폰으로 문장이 잘린 경우 붙이기
    text = re.sub(r'([\.!?]|다|요)\n+', r'\1 ', text) # 문장이 끝난 뒤 줄바꿈은 공백 한 칸으로 변경
    text = re.sub(r'[ \t]+', ' ', text) # 연속된 공백이나 탭은 하나의 공백으로 통일
    text = re.sub(r'([가-힣])\n([가-힣])', r'\1\2', text) # 한글 음절 사이 줄바꿈은 강제로 이어 붙이기
    text = re.sub(r'\n(?!제\s*\d+\s*조)', ' ', text) # 그 외 모든 줄바꿈은 공백 처리
    text = re.sub(r'\s*(부칙[^\n]*)', r'\n\1', text) # 부칙 앞에 줄바꿈 삽입으로 시작하게 처리

    # 법령명 앞에 줄바꿈 강제 삽입 
    text = re.sub(
        r'(?<![0-9\)\」])'  
        r'[^\w가-힣\)」]*\s*'    
        r'(제\s*\d+\s*조(?:의\s*\d+)?'                # 제 n 조 (+ '의 m' 옵션)
        r'(?:\s*\([^)]+\)|\s*삭제|\s*생략|\s*\[[^\]]+\])'   # ① (제목) 또는 ② 삭제 또는 ③ [종전 … 이동 …] ④ 생략
        r')',
        r'\n\1',
        text
    )
    
    # 예외처리 1: [시행일: … ] 제n조 라인 예외
    text = re.sub(
        r'(\[시행일:[^\]]+\])\s*\n+\s*(제\s*\d+\s*조)', 
        r'\1 \2',
        text
    )

    # 예외처리 2: 제N조( 제…조 … )  ← 괄호 안이 다른 조문 인용이면 예외
    text = re.sub(
        r'\n\s*'                                   
        r'(제\s*\d+\s*조(?:의\s*\d+)?\('           # '제 n 조(' (+ '의 m' 옵션)                  
        r'\s*(?:법\s*)?제\s*\d+\s*조)',            # 괄호 안이 바로 '(법)제 m 조'로 시작
        r' \1',                                 
        text
    )

    return text.strip()


# ---------------------------------------
# 5) 키워드 유틸(조사 제거/키워드 추출)
# ---------------------------------------
# ---------------------------------------
# 조사 제거 / 키워드
# ---------------------------------------
def remove_josa(word):
    for j in ['은','는','이','가','을','를','에','의','와','과','도','로','으로','에서','에게','한테','부터','까지','만','보다','처럼','조차','마저']:
        if word.endswith(j):
            return word[:-len(j)]
    return word

def get_keywords(text, ratio=1, max_k=30, min_k=5):
    est_k = int(len(text) * ratio / 10) # 텍스트 길이에 비례해서 추출할 키워드 개수(est_k) 계산
    top_k = max(min_k, min(est_k, max_k)) # 추출할 최종 키워드 개수
    try:
        kws = kw_model.extract_keywords(text, top_n=top_k)
    except Exception:
        return []
    return list({remove_josa(k[0]) for k in kws if isinstance(k, (list, tuple)) and k and isinstance(k[0], str)})


# ---------------------------------------
# 6) 조문 파싱 & 키워드
# ---------------------------------------
# ---------------------------------------
# 조문 파싱 & 키워드
# ---------------------------------------
def extract_fixed_keywords(src):
    pat = r'^(.*?)\s+제(\d+)조(?:\(([^)]+)\))?(?:\s+(\d+)항)?(?:\s+(\d+)호)?'
    m = re.match(pat, src)
    if not m:
        return set()
    law = m.group(1)
    parts = re.split(r'[()\s]+', law)
    law_k = {remove_josa(p) for p in parts if p}
    jo = f"{m.group(2)}조"
    hang = f"{m.group(4)}항" if m.group(4) else ''
    ho = f"{m.group(5)}호" if m.group(5) else ''
    title_k = set(re.split(r'[^\w가-힣]+', m.group(3) or ''))
    # 법령명 키워드 + (조, 항, 호) + 조 제목 키워드 반환
    return law_k | {jo, hang, ho} | title_k

def parse_law_blocks(text, file_name):
    chunks = []
    last_main_no: str | None = None # 마지막 조문 번호 ( 중복 조항 제거용 )
    is_buchik = False # 부칙 여부

    pat = re.compile(                       # < 제목 캡쳐 조건>
        r'(?m)^\s*'                                   # 행 맨 앞
        r'(?P<title>부칙|제\s*\d+\s*조(?:의\s*\d+)?'
        r'(?:'                                        
          r'\s*\([^)]+\)'                           #  (제목) |
          r'|\s*삭제'                                #   삭제  |
          r'|\s*생략'                                 #   생략 |
          r'|\s*\[[^\]]+\]'                           #  [종전 … 이동 …]
          r'|\s*$'  
        r')'                                        
        r')'                                          
        r'\s*'                                      
        r'(.*?)'                                     
        r'(?='                                        # < 본문 종료 조건 >
            r'(?:'                                    #
                r'^\s*제\s*\d+\s*조(?:의\s*\d+)?'
                 r'(?:\s*\([^)]+\)|\s*삭제|\s*생략|\s*\[[^\]]+\]|\s*$)|'
                r'^\s*부칙|'
                r'^\s*$'
            r')'
            r'|\Z'
        r')',
        re.S                                          # DOTALL
    )

    for title, body in pat.findall(text):
        t = title.strip()
        b = body.strip()
        if title.startswith('부칙'):
            is_buchik = True
            continue
        lawdef=''
        m = re.match(r'(제\s*\d+\s*조(?:의\s*\d+)?)(.*)', title)
        main_no  = m.group(1).strip()          # 제47조
        tail_dec = m.group(2).strip()          # [종전 …]  또는  삭제  또는  (제목)

        # 중복 조문 제거 
        if '삭제' in tail_dec and main_no == last_main_no:
            continue                   # 삭제 같은 타이틀이 연속 등장하면 두 번째는 스킵
        
        # 삭제 | [종전 … 이동 …] | 생략이 아닌 경우 lawdef 정의
        if '삭제' not in tail_dec and not '[종전' in tail_dec and '생략' not in tail_dec:
            lawdef=tail_dec

        # 길이<5 | 삭제 | [종전..] | 생략인 경우 body=tail_dec
        if (len(b) < 5 or re.match(r'^[^\w가-힣]+$', b)) or '삭제' in tail_dec or '[종전' in tail_dec or '생략' in tail_dec:
            if tail_dec:
                b = tail_dec
            else:
                continue

        src = f"{file_name} {main_no}{lawdef}"

        auto_k = get_keywords(b)
        fixed_k = extract_fixed_keywords(src)
        keywords = list(set(auto_k)|fixed_k)
        summary = re.findall(r'\((.*?)\)', t)
        enriched = f"{summary[-1]}에 대한 규정입니다. {b}" if summary else b
        
        # 부칙인 경우 source_str에 부칙 추가
        if is_buchik:
            src = f"{file_name} 부칙 {main_no}{lawdef}"
            
        chunks.append({
            "source": src,
            "enriched_text": enriched,
            "text": b,
            "keywords": keywords, 
        })

        last_main_no = main_no # 중복 조문 제거를 위해 마지막 조문 번호 저장
    is_buchik = False
    return chunks


# ---------------------------------------
# 7) 벡터화 & FAISS 인덱스
# ---------------------------------------
# ---------------------------------------
# 벡터화 & FAISS 인덱스
# ---------------------------------------
def build_index(chunks, model, save_dir="saved_index", encode_batch_size=8):
    os.makedirs(save_dir, exist_ok=True)
    docs = [c["enriched_text"] for c in chunks]
    if not docs:
        raise ValueError("빌드할 문서가 없습니다: chunks 리스트가 비어 있습니다.")

    index, all_vecs_batches = None, []

    # 텍스트를 batch 단위로 임베딩
    for i in range(0, len(docs), encode_batch_size):
        batch = docs[i:i+encode_batch_size]
        vecs = model.encode(
            batch,
            batch_size=encode_batch_size,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        npb = np.array(vecs, dtype=np.float32)
        all_vecs_batches.append(npb)

        if index is None:
            index = faiss.IndexFlatIP(npb.shape[1])
        index.add(npb)

        del batch, vecs, npb
        torch.cuda.empty_cache()

    all_vecs = np.vstack(all_vecs_batches)
    np.save(os.path.join(save_dir, "vectors.npy"), all_vecs)

    # 텍스트 원문 저장 (docs.txt)
    with open(os.path.join(save_dir, "docs.txt"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d.replace("\n", " ") + "\n")
    # 원본 chunks 데이터 저장 (JSON)
    with open(os.path.join(save_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    # FAISS 인덱스 저장
    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))
    return model, index, docs


# ---------------------------------------
# 8) 카테고리별 인덱스 로드/빌드 (idx_singlelevel 구조 생성)
# ---------------------------------------
# ---------------------------------------
# 카테고리별 인덱스 로드/빌드 (idx_singlelevel 구조 생성)
# ---------------------------------------
def load_or_build_category(cat, paths, model, encode_batch_size=8):
    # 카테고리별 인덱스를 저장할 기본 디렉토리 설정
    os.makedirs(IDX_PARENT_DIR, exist_ok=True)
    save_dir = os.path.join(IDX_PARENT_DIR, f"idx_{cat}")

    jp = os.path.join(save_dir, "chunks.json")
    dp = os.path.join(save_dir, "docs.txt")
    vp = os.path.join(save_dir, "vectors.npy")
    ip = os.path.join(save_dir, "index.faiss")

    # 이미 빌드된 인덱스가 존재하면 그대로 로드
    if os.path.isdir(save_dir) and all(os.path.isfile(p) for p in (jp, dp, vp, ip)):
        index = faiss.read_index(ip)
        with open(jp, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        docs = [c["enriched_text"] for c in chunks]
        return model, index, chunks, docs

    # 인덱스가 없으면 새로 빌드
    os.makedirs(save_dir, exist_ok=True)
    chunks = []

    for p in paths:
        if p.lower().endswith(".pdf"): # PDF -> TXT
            txt = p[:-4] + ".txt"
            if not os.path.exists(txt):
                convert_pdf_to_txt(p, txt)  
            p = txt

        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception:
            continue

        try:
            pr = preprocess_text(raw) # 청킹
            new_chunks = parse_law_blocks(pr, os.path.basename(p).rsplit(".", 1)[0])
            chunks.extend(new_chunks)
        except Exception:
            continue

    unique = {c['source']: c for c in chunks}
    chunks = list(unique.values())

    model, index, docs = build_index(chunks, model, save_dir, encode_batch_size=encode_batch_size)
    return model, index, chunks, docs


# ---------------------------------------
# 9) 인덱스 초기화(전처리 + 빌드)
# ---------------------------------------
# ---------------------------------------
# 인덱스 초기화(전처리 + 빌드)
# ---------------------------------------
def init_rag_indices(base_dir: str, model):
    import glob
    all_files = glob.glob(os.path.join(base_dir, "*.pdf")) + glob.glob(os.path.join(base_dir, "*.txt"))
    # 카테고리별 파일 그룹핑
    groups = {cat: [] for cat in desired_categories}
    for p in all_files:
        nm = os.path.basename(p)
        for cat in desired_categories:
            if cat in nm:
                groups[cat].append(p)
                break
    # 카테고리별 인덱스 빌드 or 로드
    cat_indices = {}
    for cat, paths in groups.items():
        if not paths:
            continue
        m, index, chunks, docs = load_or_build_category(cat, paths, model)  # 기본 배치 크기 사용
        cat_indices[cat] = {"model": m, "index": index, "chunks": chunks, "docs": docs}

    # "전체" 카테고리 인덱스 (모든 파일을 포함)
    m, idx, chunks, docs = load_or_build_category("전체", all_files, model)  # 기본 배치 크기 사용
    cat_indices["전체"] = {"model": m, "index": idx, "chunks": chunks, "docs": docs}
    print(f"[OK] idx_singlelevel/idx_전체 빌드 완료: {len(chunks)} chunks")
    return cat_indices


# ---------------------------------------
# 10) 메인: 전처리 + 인덱스 빌드만 수행
# ---------------------------------------
# ---------------------------------------
# 메인: 전처리 + 인덱스 빌드만 수행
# ---------------------------------------
def main():
    base_dir = input("문서 폴더 경로를 입력해주세요. (미입력 시 기본) : ").strip() or BASE_DIR
    _ = init_rag_indices(base_dir, embed_model)
    print("\n[완료] 전처리 + 인덱스 빌드 완료 (idx_singlelevel/* 생성)")

if __name__ == "__main__":
    main()
