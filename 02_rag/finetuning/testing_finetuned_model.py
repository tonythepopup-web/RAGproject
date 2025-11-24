# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader  # 사용 안 해도 무방
import numpy as np
import re, torch, faiss, glob
from keybert import KeyBERT
import os, json, math
import fitz  # PyMuPDF

torch.cuda.empty_cache()

# ----------------------------
# 전역 모델/설정 (상대 경로 사용)
# ----------------------------
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")

# 파인튜닝된 모델 경로 (00_data/finetuning/finetuned_embedding_model/ 내)
MODEL_DIR = str(SCRIPT_DIR.parent.parent / "00_data" / "finetuning" / "finetuned_embedding_model" / "cfg10_cosine_triplet_e1_m0.3")
embed_model = SentenceTransformer(MODEL_DIR, device="cuda")

# ----------------------------
# PDF → TXT 변환
# ----------------------------
def convert_pdf_to_txt(pdf_path: str, txt_path: str):
    """
    PyMuPDF(fitz)로 PDF 텍스트 추출 → TXT 저장
    """
    doc = fitz.open(pdf_path)
    all_text = []
    for page in doc:
        page_text = page.get_text("text")
        all_text.append(page_text)

    text = "\n".join(all_text)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[완료] PyMuPDF로 텍스트 추출 완료 → {txt_path}")

# ===============================================================================================
# 1. 전처리 - 법제처 헤더 제거
def remove_header(raw: str) -> str:
    HEADER_RE = re.compile(
        r'(?m)'
        r'^[ \t]*법제처.*?국가법령정보센터.*\n'
        r'[^\n]*\n?'
    )

    def strip_korean_law_header(raw: str) -> str:
        while True:
            new = HEADER_RE.sub('', raw)
            if new == raw:
                return new
            raw = new

    return strip_korean_law_header(raw)

# 1. 전처리 - “n조 및 m조 생략” 확장
def expand_multi_delete(raw: str) -> str:
    pattern = re.compile(
        r'(?m)^\s*(제\s*(\d+)조)\s*및\s*(제\s*(\d+)조)\s*(생략)\s*$'
    )
    def _repl(m: re.Match) -> str:
        first_title  = m.group(1)
        second_title = m.group(3)
        return f"{first_title} 생략\n{second_title} 생략"
    return pattern.sub(_repl, raw)

# 1. 전처리 - 메인
def preprocess_text(raw):
    raw = remove_header(raw)
    raw = expand_multi_delete(raw)
    raw = re.sub(r'(?m)^\s*제\s*\d+\s*장[^\n]*\n?', '', raw)  # “제N장 …” 제거
    text = re.sub(r'[·※▶★■●◦◇ㆍ]', '', raw)
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'([\.!?]|다|요)\n+', r'\1 ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'([가-힣])\n([가-힣])', r'\1\2', text)
    text = re.sub(r'\n(?!제\s*\d+\s*조)', ' ', text)
    text = re.sub(r'\s*(부칙[^\n]*)', r'\n\1', text)

    text = re.sub(
        r'(?<![0-9\)\」])'
        r'[^\w가-힣\)」]*\s*'
        r'(제\s*\d+\s*조(?:의\s*\d+)?'
        r'(?:\s*\([^)]+\)|\s*삭제|\s*생략|\s*\[[^\]]+\])'
        r')',
        r'\n\1',
        text
    )
    text = re.sub(
        r'(\[시행일:[^\]]+\])\s*\n+\s*(제\s*\d+\s*조)',
        r'\1 \2',
        text
    )
    text = re.sub(
        r'\n\s*'
        r'(제\s*\d+\s*조(?:의\s*\d+)?\('
        r'\s*(?:법\s*)?제\s*\d+\s*조)',
        r' \1',
        text
    )
    return text.strip()

# ----------------------------
# 키워드 추출 & 조사 제거
# ----------------------------
def get_keywords(text, ratio=1, max_k=30, min_k=5):
    est_k = int(len(text) * ratio / 10)
    top_k = max(min_k, min(est_k, max_k))
    kws = kw_model.extract_keywords(text, top_n=top_k)
    return list({remove_josa(k[0]) for k in kws})

def remove_josa(word):
    josa = ['은','는','이','가','을','를','에','의','와','과','도','로',
            '으로','에서','에게','한테','부터','까지','만','보다','처럼','조차','마저']
    for j in josa:
        if word.endswith(j):
            return word[:-len(j)]
    return word

# ----------------------------
# 조문 단위 파싱
# ----------------------------
def parse_law_blocks(text, file_name):
    chunks = []
    seen_src = set()
    pat = re.compile(
        r'(?m)^(제\s*\d+조(?:의\d+)?(?:\([^)]+\))?)\s*(.*?)(?=^제\s*\d+조|\Z)',
        re.S
    )

    for title, body in pat.findall(text):
        t = title.strip()
        b = body.strip()

        # 삭제/생략 패턴 제외
        if re.fullmatch(
            r'^(?:삭제|생략)\s*<\s*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*>?\s*$',
            b
        ):
            continue

        if len(b) < 5 or re.match(r'^[^\w가-힣]+$', b):
            continue

        src = f"{file_name} {t}"
        if src in seen_src:
            continue
        seen_src.add(src)

        summary = re.findall(r'\((.*?)\)', t)
        enriched = f"{summary[-1]}에 대한 규정입니다. {b}" if summary else b

        auto_k = get_keywords(b)
        fixed_k = extract_fixed_keywords(src)
        chunks.append({
            "source":       src,
            "enriched_text": enriched,
            "text":         b,
            "keywords":     list(set(auto_k) | fixed_k)
        })
    return chunks

def extract_fixed_keywords(src):
    pat=r'^(.*?)\s+제(\d+)조(?:\(([^)]+)\))?(?:\s+(\d+)항)?(?:\s+(\d+)호)?'
    m=re.match(pat, src)
    if not m:
        return set()
    law=m.group(1)
    parts=re.split(r'[()\s]+', law)
    law_k={remove_josa(p) for p in parts if p}

    jo=f"{m.group(2)}조"
    hang=f"{m.group(4)}항" if m.group(4) else ''
    ho=f"{m.group(5)}호" if m.group(5) else ''
    title_k=set(re.split(r'[^\w가-힣]+', m.group(3) or ''))

    return law_k|{jo,hang,ho}|title_k

# ----------------------------
# 인덱스 빌드
# ----------------------------
def build_index(chunks, model, save_dir="saved_index", encode_batch_size=8):
    """
    chunks: [{'enriched_text': ...}, ...]
    model: SentenceTransformer
    """
    os.makedirs(save_dir, exist_ok=True)
    docs = [c["enriched_text"] for c in chunks]
    if not docs:
        raise ValueError("빌드할 문서가 없습니다: chunks 리스트가 비어 있습니다.")

    index = None
    all_vecs_batches = []
    for i in range(0, len(docs), encode_batch_size):
        batch = docs[i : i + encode_batch_size]
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
            dim = npb.shape[1]
            index = faiss.IndexFlatIP(dim)
        index.add(npb)

        del batch, vecs, npb
        torch.cuda.empty_cache()

    # vectors.npy
    vp = os.path.join(save_dir, "vectors.npy")
    all_vecs = np.vstack(all_vecs_batches)
    if not os.path.exists(vp):
        np.save(vp, all_vecs)
        print(f"[저장 완료] 벡터 파일 → {vp}")
    else:
        print(f"[스킵] 이미 존재하는 벡터 파일: {vp}")

    # docs.txt
    dt = os.path.join(save_dir, "docs.txt")
    if not os.path.exists(dt):
        with open(dt, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(d.replace("\n", " ") + "\n")
        print(f"[저장 완료] docs.txt → {dt}")
    else:
        print(f"[스킵] 이미 존재하는 docs.txt: {dt}")

    # chunks.json
    jp = os.path.join(save_dir, "chunks.json")
    if not os.path.exists(jp):
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[저장 완료] chunks.json → {jp}")
    else:
        print(f"[스킵] 이미 존재하는 chunks.json: {jp}")

    # index.faiss
    ip = os.path.join(save_dir, "index.faiss")
    if not os.path.exists(ip):
        faiss.write_index(index, ip)
        print(f"[저장 완료] FAISS 인덱스 → {ip}")
    else:
        print(f"[스킵] 이미 존재하는 FAISS 인덱스: {ip}")

    return model, index, docs

# ----------------------------
# IDF 계산
# ----------------------------
def compute_idf(chunks):
    N = len(chunks)
    df = {}
    for ch in chunks:
        for kw in set(ch['keywords']):
            df[kw] = df.get(kw, 0) + 1
    return {kw: math.log((N + 1) / (cnt + 1)) + 1 for kw, cnt in df.items()}

# ----------------------------
# 검색 (rerank 제거 버전)
# ----------------------------
def retrieve_docs(
    query: str,
    model,
    index,
    docs,
    chunks,
    IDF,
    alpha: float = 0.9,
    top_k: int = 5,
    *,
    return_scores: bool = False
):
    """
    코사인(정규화 내적) + 키워드 IDF 하이브리드로 top_k 문서를 반환.
    Reranking은 사용하지 않습니다.
    """
    # 1) 쿼리 임베딩 & 전체 검색
    qv = model.encode([query], normalize_embeddings=True)[0]
    dists, I = index.search(np.array([qv]), len(docs))
    dists, I = dists[0], I[0]

    # 2) 하이브리드 점수
    sem = np.clip(dists, 0, 1)  # 내적=코사인, 음수 노이즈 컷
    qk  = set(get_keywords(query))
    ks  = np.array([
        sum(IDF.get(kw, 1.0) for kw in (qk & set(chunks[i]['keywords'])))
        for i in I
    ], dtype=np.float32)
    if ks.max() > 0:
        ks /= (ks.max() + 1e-6)

    hybrid = alpha * sem + (1 - alpha) * ks

    # 3) top_k 선택
    order   = np.argsort(hybrid)[::-1]
    top_pos = order[:top_k]
    top_idx = I[top_pos]

    candidates = [chunks[i] for i in top_idx]
    candidate_scores = [float(hybrid[p]) for p in top_pos]
    return (candidates, candidate_scores) if return_scores else candidates

# ----------------------------
# 보조 유틸
# ----------------------------
def clean_path(p): return p.strip().strip('"')

def preprocess_query(q):
    q=re.sub(r'제\s*(\d+)',r'\1',q)
    q=re.sub(r'(\d+)조\s*의\s*(\d+)',r'\1조의\2',q)
    q=re.sub(r'[^\w가-힣\s]','',q)
    print("전처리 결과:",q)
    return q.strip()

def clean_source(src: str) -> str:
    src = re.sub(r'\(제\d+호\)', '', src)
    src = re.sub(r'\(\d{8}\)', '', src)
    return re.sub(r'\s+', ' ', src).strip()

def save_query_result(query, docs, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        for d in docs:
            line = clean_source(d['source'])
            f.write(line + "\n")

# ----------------------------
# 카테고리/매핑
# ----------------------------
desired_categories = [
    "가축전염병 예방법","건강기능식품에 관한 법률","농약관리법",
    "먹는물관리법","사료관리법","수입식품안전관리","식품위생법",
    "식품ㆍ의약품분야 시험ㆍ검사","축산물 위생관리법","한국식품안전관리인증원의 설립 및 운영에 관한 법률"
]

mapping = { str(i): cat for i, cat in enumerate(["전체"] + desired_categories) }

category_embeddings = embed_model.encode(
    desired_categories,
    normalize_embeddings=True,
    convert_to_tensor=False
)

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

def classify_category(query: str, sem_threshold: float = 0.5) -> list[str]:
    kw_cands = [cat for cat in desired_categories if query.count(cat) >= 1]
    q_vec = embed_model.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(category_embeddings, q_vec)
    sem_cands = [
        desired_categories[i]
        for i, sim in enumerate(sims)
        if sim >= sem_threshold
    ]
    candidates = []
    for cat in kw_cands + sem_cands:
        if cat not in candidates:
            candidates.append(cat)
    return candidates

# ----------------------------
# 카테고리별 인덱스 로드/빌드
# ----------------------------
def load_or_build_category(cat, paths, model, encode_batch_size=8):
    PARENT_DIR = "idx_singlelevel_triplet_cosine"
    os.makedirs(PARENT_DIR, exist_ok=True)
    save_dir = os.path.join(PARENT_DIR, f"idx_{cat}")
    jp = os.path.join(save_dir, "chunks.json")
    dp = os.path.join(save_dir, "docs.txt")
    vp = os.path.join(save_dir, "vectors.npy")
    ip = os.path.join(save_dir, "index.faiss")

    # 캐시 있으면 로드
    if os.path.isdir(save_dir) and all(os.path.isfile(p) for p in (jp, dp, vp, ip)):
        print(f"[스킵] 캐시 존재: {save_dir}")
        index = faiss.read_index(ip)
        with open(jp, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        docs = [c["enriched_text"] for c in chunks]
        return model, index, chunks, docs

    # 없으면 생성
    os.makedirs(save_dir, exist_ok=True)
    chunks = []
    for p in paths:
        if p.lower().endswith(".pdf"):
            txt = p[:-4] + ".txt"
            if not os.path.exists(txt):
                convert_pdf_to_txt(p, txt)
            else:
                print(f"[스킵] 이미 존재하는 TXT: {txt}")
            p = txt

        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception as e:
            print(f"[오류] 파일 읽기 실패 ({p}): {e}. 스킵합니다.")
            continue

        try:
            pr = preprocess_text(raw)
            new_chunks = parse_law_blocks(pr, os.path.basename(p).rsplit(".",1)[0])
            chunks.extend(new_chunks)
        except Exception as e:
            print(f"[오류] 파싱 실패 ({p}): {e}. 스킵합니다.")
            continue

    # 중복 제거
    unique = {}
    for c in chunks:
        unique[c['source']] = c
    chunks = list(unique.values())

    # 인덱스 빌드
    model, index, docs = build_index(chunks, model, save_dir, encode_batch_size)
    return model, index, chunks, docs

# ----------------------------
# 메인
# ----------------------------
def main():
    # 1) 문서 폴더 경로
    base_dir = input("문서 폴더 경로를 입력해주세요. (입력이 없을 시 디폴트 주소 호출) :").strip() \
               or "/home/ubuntu/parksekyeong/전체_법률_파일"
    model = SentenceTransformer(MODEL_DIR)

    # 2) 전체 PDF/TXT 탐색
    all_files = glob.glob(os.path.join(base_dir, "*.pdf")) + \
                glob.glob(os.path.join(base_dir, "*.txt"))

    # 3) 카테고리별 그룹화
    groups = {cat: [] for cat in desired_categories}
    for p in all_files:
        nm = os.path.basename(p)
        for cat in desired_categories:
            if cat in nm:
                groups[cat].append(p)
                break

    # 4) 카테고리별 인덱스 로드/빌드
    cat_indices = {}
    for cat, paths in groups.items():
        if not paths:
            print(f"[경고] '{cat}'에 해당하는 파일이 없습니다. 스킵합니다.")
            continue
        model, index, chunks, docs = load_or_build_category(
            cat, paths, model, encode_batch_size=8
        )
        IDF = compute_idf(chunks)
        cat_indices[cat] = {
            "model":   model,
            "index":   index,
            "chunks":  chunks,
            "docs":    docs,
            "IDF":     IDF
        }

    # 5) 전체 인덱스
    m, idx, chunks, docs = load_or_build_category("전체", all_files, model)
    IDF = compute_idf(chunks)
    cat_indices["전체"] = {
        "model": m, "index": idx, "chunks": chunks, "docs": docs, "IDF": IDF
    }

    # 6-1) 카테고리별 벤치마크(메타데이터만) 자동 테스트
    ordered_categories = [
        "가축전염병 예방법","농약관리법","먹는물관리법","사료관리법","축산물 위생관리법",
        "식품ㆍ의약품분야 시험ㆍ검사","건강기능식품에 관한 법률","수입식품안전관리",
        "식품위생법","한국식품안전관리인증원의 설립 및 운영에 관한 법률"
    ]
    auto_test = input("카테고리별 벤치마크를 실행하시겠습니까? (y/n) >> ").strip().lower() == "y"
    if auto_test:
        file_path = "/home/ubuntu/parksekyeong/qnalist.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]

        combined_file = "/home/ubuntu/parksekyeong/result.txt"
        open(combined_file, "w", encoding="utf-8").close()

        for idx_cat, cat in enumerate(ordered_categories):
            start = idx_cat * 10
            end   = start + 10
            cat_qs = questions[start:end]

            model  = cat_indices[cat]["model"]
            index  = cat_indices[cat]["index"]
            docs   = cat_indices[cat]["docs"]
            chunks = cat_indices[cat]["chunks"]

            for q in cat_qs:
                uq = preprocess_query(q)
                results = retrieve_docs(
                    uq,
                    model, index,
                    docs, chunks,
                    cat_indices[cat]["IDF"],
                    top_k=5
                )
                with open(combined_file, "a", encoding="utf-8") as f:
                    for d in results:
                        f.write(clean_source(d["source"]) + "\n")

        print(f"[완료] 모든 카테고리 결과가 '{combined_file}'에 저장되었습니다.")
        return

    # 6-2) 질문+조문내용 CHATML 포맷 저장
    auto_test = input("카테고리별 벤치마크를 실행하시겠습니까? (y/n) >> ").strip().lower() == "y"
    if auto_test:
        file_path = "/home/ubuntu/parksekyeong/qnalist.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]

        output_file = "/home/ubuntu/parksekyeong/result_CHATML.txt"
        with open(output_file, "w", encoding="utf-8") as out:
            for idx_cat, cat in enumerate(ordered_categories):
                cat_qs = questions[idx_cat * 10 : (idx_cat + 1) * 10]
                for q in cat_qs:
                    cfg = cat_indices[cat]
                    uq = preprocess_query(q)
                    results = retrieve_docs(
                        uq,
                        cfg["model"],
                        cfg["index"],
                        cfg["docs"],
                        cfg["chunks"],
                        cfg["IDF"],
                        top_k=5
                    )
                    import textwrap
                    context_blocks = []
                    for i, chunk in enumerate(results[:4], start=1):
                        title = clean_source(chunk["source"])
                        body = chunk["text"]
                        wrapped_body = "\n".join(textwrap.wrap(body, width=80))
                        context_blocks.append(f"{i}. {title}\n{wrapped_body}")
                    context = "\n\n".join(context_blocks)

                    prompt = f"""<|im_start|>system
당신은 한국 법령에 정통한 전문 법률 어시스턴트입니다. 아래에는 사용자 질문과 법령에서 발췌한 조문들이 주어집니다.

당신의 임무는 세 가지입니다:
1. 조문들을 읽고 질문과 관련 있는 내용만 골라 핵심을 요약하세요. 
2. 오직 주어진 조문만을 바탕으로 명확하고 신뢰도 높은 답변을 제공하세요.
3. 만약 모든 조문이 질문과 관련 없다면, 관련된 조문이 없음을 밝히고 답변을 유보하세요.
4. 제시된 조문 중 참고할 사항이 없는 조문은 따로 표시해주세요.

[출력 형식 예시]
1. 요약:
(여기에 핵심 요약)

2. 답변:
(여기에 답변 내용 — 해당 조문 번호 언급 포함)

3. 참고 사항 없는 조 :
(여기에 몇조안지 나열)
<|im_end|>
<|im_start|>user
[사용자 질문]
{q}

[조문 내용]
{context}
<|im_end|>
<|im_start|>assistant
""".strip()

                    out.write(f"{prompt}\n")
                    out.write("=" * 100 + "\n\n")
        print(f"[완료] 포맷된 결과가 '{output_file}'에 저장되었습니다.")
        return

    # 7) 대화형 검색 루프
    while True:
        q = input("질문을 입력하세요 (종료는 exit) >> ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        print("\n=== 전체 카테고리 목록 ===")
        for i in range(0, 6):
            print(f"{i}) {mapping[str(i)]}", end="    ")
        print()
        for i in range(6, len(desired_categories) + 1):
            print(f"{i}) {mapping[str(i)]}", end="    ")
        print("\n")

        candidates = classify_category(q)
        print("\n가능성 있는 카테고리:")
        print("0) 전체")
        for i, cat in enumerate(desired_categories, start=1):
            if cat in candidates:
                print(f"{i}) {cat}")

        choice = input("번호/이름 입력 >> ").strip()
        sel = parse_category_input(choice) if choice else None
        if not sel:
            sel = "전체"

        uq = preprocess_query(q)
        cfg = cat_indices[sel]

        results = retrieve_docs(
            uq,
            cfg["model"],
            cfg["index"],
            cfg["docs"],
            cfg["chunks"],
            cfg["IDF"],
            top_k=5
        )

        save_query_result(uq, results, f"{sel}_results.txt")

        print(f"\n== {sel} 검색 결과 ==")
        for r in results:
            print(r["source"])

if __name__ == "__main__":
    main()
