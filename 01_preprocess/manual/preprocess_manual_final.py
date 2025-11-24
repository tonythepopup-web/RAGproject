"""
매뉴얼 JSON/JSONL → RAG 청크 변환 + 문서별 인덱스(idx_<제목>/)

- source = "<파일명> <enriched_text>"
- text   = 원본 blocks 배열 그대로 (보존/렌더링용)
- embedding_text = 임베딩/RAG 검색용 평탄화 문자열
- keywords = KeyBERT 기반 로직(get_keywords)으로 추출 (옵션)

출력
- 파일별 청크: {OUT_DIR}/{제목}_chunks.json
- 전체 청크:  {OUT_DIR}/all_chunks.json
- 문서별 인덱스: {OUT_DIR}/idx_<제목>/
    ├─ chunks.json   (매핑 최소본)
    ├─ vectors.npy   (BGE 임베딩)
    └─ index.faiss   (FAISS 인덱스; cosine=inner product on normalized)
"""

# ---------------------------------------
# 0) 설정 + 기본 import (전역 변수는 IN_DIR/OUT_DIR/EMBED_MODEL_NAME만 유지)
# ---------------------------------------
import os, json, glob
from typing import Any, Dict, List

import numpy as np
import faiss, torch
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from functools import lru_cache

# ====== 전역 설정 (상대 경로 사용) ======
# IN_DIR : 입력 JSON/JSONL이 위치한 디렉터리
# OUT_DIR: 산출물(idx_<제목>/, idx_all/)이 저장될 루트 디렉터리
# EMBED_MODEL_NAME: SentenceTransformer 임베딩 모델 이름
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
IN_DIR  = str(SCRIPT_DIR.parent.parent / "00_data" / "input" / "raw_manual" / "매뉴얼_1차_전처리결과물(json파일모음)")

# 날짜별 인덱스 저장 (indexes/manual/YYYY-MM-DD 구조)
TODAY_DATE = datetime.now().strftime("%Y-%m-%d")
OUT_DIR = str(SCRIPT_DIR.parent.parent / "00_data" / "input" / "indexes" / "manual" / TODAY_DATE)
EMBED_MODEL_NAME = "dragonkue/BGE-m3-ko"  # BGE 임베딩 모델


# ---------------------------------------
# 1) 키워드 유틸 (조사 제거 / KeyBERT 추출)  
#    - 사용 목적: RAG 검색 보조용 키워드 집합 생성.
#    - 핵심 포인트:
#        * remove_josa: 한국어 조사(을/를/은/는...)를 단어 말단에서 제거해 IDF/매칭 품질 향상
#        * _get_kw_model: KeyBERT 모델을 lazy-load(최초 1회만 로딩)하여 속도/메모리 최적화
#        * get_keywords: 본문 길이에 따른 top_k 자동 산정(ratio 기반), 중복 제거
#    - 예외 처리: KeyBERT 추출 실패 시 빈 리스트 반환(파이프라인 중단 방지)
# ---------------------------------------

def remove_josa(word: str) -> str:
    # 입력: word (단일 토큰 문자열)
    # 처리: 지정된 조사가 어미에 붙어 있으면 제거 → 어간 기준 키워드로 정규화
    # 반환: 조사 제거된 문자열(없으면 원문 그대로)
    for j in ['은','는','이','가','을','를','에','의','와','과','도','로','으로','에서','에게','한테','부터','까지','만','보다','처럼','조차','마저']:
        if word.endswith(j):
            return word[:-len(j)]
    return word

@lru_cache(maxsize=1)
def _get_kw_model() -> KeyBERT:
    return KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")

def get_keywords(text: str, ratio: int = 1, max_k: int = 30, min_k: int = 5) -> List[str]:
    # 입력:
    #   - text  : 키워드 추출 대상 전체 문자열(embedding_text 등)
    #   - ratio : 본문 길이에 비례해 top_k를 대략 산정하기 위한 비율(기본 1)
    #   - max_k : 상한(top_k 최대값)
    #   - min_k : 하한(top_k 최소값)
    # 처리:
    #   1) 본문 길이 기반으로 예상 k(est_k) 계산 → [min_k, max_k] 범위로 클리핑
    #   2) KeyBERT.extract_keywords(top_n=top_k)로 후보 추출
    #   3) 조사 제거(remove_josa) 및 공백 정리 → set으로 중복 제거
    # 예외:
    #   - 모델 로딩/추출 중 오류 시 [] 반환(파이프라인 지속)
    # 반환:
    #   - 중복 제거된 키워드 리스트(순서는 set 특성상 보장되지 않음)
    s = (text or "").strip()
    est_k = int(len(s) * ratio / 10)
    top_k = max(min_k, min(est_k, max_k))
    try:
        kws = _get_kw_model().extract_keywords(s, top_n=top_k)
    except Exception:
        return []
    uniq = set()
    for k in kws:
        # KeyBERT 반환 형태: [(키워드, 스코어), ...] 또는 유사 구조
        if isinstance(k, (list, tuple)) and k and isinstance(k[0], str):
            cleaned = remove_josa(k[0].strip())
            if cleaned:
                uniq.add(cleaned)
    return list(uniq)


# ---------------------------------------
# 2) 로딩 유틸 (JSON/JSONL)  
#    - 역할: 다양한 JSON 구조/확장자에 대응하는 안전한 로더.
#    - load_json_whole: JSON(dict/list)을 유연하게 리스트[dict]로 표준화
#    - load_jsonl_lines: JSONL 라인별 안전 파싱(문제 라인 번호까지 로깅)
#    - load_records: 확장자에 따라 위 둘 중 하나 선택
# ---------------------------------------

def ensure_dir(p: str):
    # 목적: 출력 경로가 없을 경우 생성(존재하면 무시)
    # 부작용: 디렉터리 생성(I/O)
    os.makedirs(p, exist_ok=True)


def load_json_whole(path: str) -> List[Dict[str, Any]]:
    # 입력: path(파일 경로, .json)
    # 처리:
    #   - json.load로 파싱
    #   - 최상위가 list: dict만 필터링해서 반환
    #   - 최상위가 dict: 흔한 래핑 키(data/records/items/rows/list) 중 리스트 찾기
    #                    없으면 단일 dict를 리스트로 감싸 반환
    # 예외 처리:
    #   - 파싱 실패 시 경고 로그 출력 후 빈 리스트 반환(파이프라인 지속)
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[경고] JSON 파싱 실패: {os.path.basename(path)} → {e}")
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ("data","records","items","rows","list"):
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [obj]
    return []


def load_jsonl_lines(path: str) -> List[Dict[str, Any]]:
    # 입력: path(파일 경로, .jsonl)
    # 처리: 라인 단위로 json.loads 수행, dict만 수집
    # 로그: 특정 라인 파싱 실패 시 라인 번호 포함 경고 출력
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception as e:
                print(f"[경고] JSONL 파싱 실패: {os.path.basename(path)}:{i} → {e}")
    return out


def load_records(path: str) -> List[Dict[str, Any]]:
    # 목적: 파일 확장자 기준으로 적절한 로더 선택
    # 반환: 리스트[dict] 표준화된 레코드
    return load_jsonl_lines(path) if path.lower().endswith(".jsonl") else load_json_whole(path)


# ---------------------------------------
# 3) 테이블 → 마크다운 변환 & 블록 평탄화 (임베딩 텍스트 생성)
#    - blocks 스키마:
#        [{ "type": "text", "text": "..."},
#         { "type": "table", "columns": [...], "rows": [[...], ...] }, ...]
#    - 목적: 비정형 텍스트 + 표 데이터를 임베딩에 적합한 평탄 문자열로 변환
#    - table_to_markdown: 표를 간단한 Markdown 형태로 직렬화([표] 프리픽스 포함)
#    - blocks_to_embedding_text: text/table 외 타입은 [type] 토큰으로 대체(정보 유지)
# ---------------------------------------

def _normalize_cell(s: Any) -> str:
    # 목적: 셀 값의 공백을 단일 공백으로 정규화하여 노이즈 감소
    return " ".join(str(s).split())

def table_to_markdown(columns, rows) -> str:
    # 입력:
    #   - columns: 컬럼명 리스트(없을 수 있음)
    #   - rows   : 2차원 리스트 형태의 표 데이터
    # 처리:
    #   - columns 미존재 시 rows 최대 길이 기준으로 "열1..열N" 자동 생성
    #   - 헤더 중복 제거 → Markdown 표로 직렬화
    # 반환:
    #   - "[표]\n<markdown table>" 형식 문자열(표 없거나 잘못된 형식이면 빈 문자열)
    if not isinstance(rows, list) or not rows:
        return ""
    if not isinstance(columns, list) or not columns:
        max_cols = 0
        for r in rows:
            if isinstance(r, list):
                lr = len(r)
                if lr > max_cols:
                    max_cols = lr
        columns = [f"열{i+1}" for i in range(max_cols)]
    cols = [str(c) for c in columns]
    sep = " | ".join(["---"] * len(cols))

    header = " | ".join(cols)
    body_lines = []
    L = len(cols)
    for r in rows:
        if not isinstance(r, list):
            body_lines.append(" | ".join([""] * L))
            continue
        rv = [_normalize_cell(v) for v in r[:L]]
        if len(rv) < L:
            rv.extend([""] * (L - len(rv)))
        body_lines.append(" | ".join(rv))

    return "[표]\n" + "\n".join((header, sep, *body_lines))


def blocks_to_embedding_text(blocks: Any) -> str:
    # 입력: blocks(list of dict)
    # 처리:
    #   - type=="text" → text 수집
    #   - type=="table" → table_to_markdown 변환 후 수집
    #   - 기타 타입 → "[type]" 토큰으로 치환(예: [image], [figure] 등)
    # 반환: 수집된 파트들을 \n\n로 이어붙인 평탄 문자열
    if not isinstance(blocks, list):
        return ""
    parts_append = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text":
            t = b.get("text")
            if isinstance(t, str) and t.strip():
                parts_append.append(t)
        elif btype == "table":
            md = table_to_markdown(b.get("columns", []), b.get("rows", []))
            if md:
                parts_append.append(md)
        else:
            parts_append.append(f"[{b.get('type', 'block')}]")
    parts = [p.strip() for p in parts_append if isinstance(p, str) and p.strip()]
    return "\n\n".join(parts)


# ---------------------------------------
# 4) 파일 → 청크 빌드  
#    - 한 파일(JSON/JSONL) 내 여러 record를 반복 처리하여 chunk 리스트 생성
#    - source: 파일명 + enriched_text(없으면 chapter/paragraph 기반 문장 생성)
#    - text  : 원본 blocks/items 배열 그대로 저장(보존/렌더링용) // 이유는 전처리 할때 생성된 코드의 차이 인해 blocks 혹은 items로 동일한 항목에 대해 키 이름이 다름
#    - embedding_text: source + 평탄화된 본문(검색/임베딩 입력) 
#    - keywords: get_keywords로 추출(예외 시 빈 리스트)
#    - 로그:
#        * [스킵] 레코드에 blocks/items 없음
#        * [스킵] embedding_text 생성 실패(비어있음)
#        * [BUILD] 요약: records/chunks/스킵 수
# ---------------------------------------

def build_chunks_from_file(json_path: str) -> List[Dict[str, Any]]:
    title = os.path.basename(json_path).rsplit(".", 1)[0]
    records = load_records(json_path)
    if not records:
        print(f"[BUILD] {title}: records=0, chunks=0, skipped=0")
        return []

    chunks: List[Dict[str, Any]] = []
    skipped = 0

    for idx, rec in enumerate(records):
        # blocks/items 중 하나를 본문 단위로 간주
        units = None
        if isinstance(rec.get("blocks"), list):
            units = rec["blocks"]
        elif isinstance(rec.get("items"), list):
            units = rec["items"]

        if not units:
            skipped += 1
            print(f"[스킵] {title} rec#{idx}: blocks/items 없음")
            continue

        # enriched_text 부족 시 chapter/paragraph로 대체 문장 구성
        enriched = rec.get("enriched_text")
        if not isinstance(enriched, str) or not enriched.strip():
            chapter   = rec.get("chapter") if isinstance(rec.get("chapter"), str) else ""
            paragraph = rec.get("paragraph") if isinstance(rec.get("paragraph"), str) else ""
            if chapter and paragraph:
                enriched = f"'{chapter}'의 '{paragraph}'에 관한 내용입니다."
            elif chapter:
                enriched = f"'{chapter}'에 관한 내용입니다."
            else:
                enriched = "매뉴얼의 특정 내용입니다."

        source_str = f"{title} {enriched.strip()}"

        # 평탄화 본문 생성(표는 Markdown, 그 외 타입은 [type] 토큰)
        embedding_body = blocks_to_embedding_text(units)
        if not embedding_body:
            skipped += 1
            print(f"[스킵] {title} rec#{idx}: embedding_text 생성 실패")
            continue

        embedding_text = f"{source_str}\n\n{embedding_body}"

        # 최종 청크 객체 구성
        item: Dict[str, Any] = {
            "source": source_str,
            "text": units,
            "embedding_text": embedding_text
        }

        # 키워드 기본 생성(실패 시 빈 리스트)
        try:
            item["keywords"] = get_keywords(embedding_text)
        except Exception:
            item["keywords"] = []

        chunks.append(item)

    print(f"[BUILD] {title}: records={len(records)}, chunks={len(chunks)}, skipped={skipped}")
    return chunks


# ---------------------------------------
# 5) 문서별 인덱스 생성 (idx_<제목>/vectors.npy + index.faiss + chunks.json)
#    - 입력: title(문서명), chunks_path(해당 문서의 chunks.json 경로), model(SentenceTransformer)
#    - 동작:
#        * chunks.json 로드 → embedding_text만 추출/정리
#        * model.encode(normalize_embeddings=True)로 코사인 검색용 벡터화
#        * FAISS IndexFlatIP(내적) 구성 및 저장
#        * vectors.npy(벡터 배열), index.faiss(인덱스), chunks.json(매핑 최소본) 저장
#    - 주의:
#        * 저장 경로: OUT_DIR/idx_<title>/
# ---------------------------------------

def build_index_for_document(title: str, chunks_path: str, model: SentenceTransformer, batch_size: int = 32) -> None:
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] {title}: chunks 파일 읽기 실패 → {e}")
        return

    if not isinstance(data, list):
        print(f"[SKIP] {title}: chunks 파일 형식 오류")
        return

    # embedding_text가 존재/비공백인 항목만 사용
    usable = [c for c in data if isinstance(c, dict) and isinstance(c.get("embedding_text"), str) and c["embedding_text"].strip()]
    if not usable:
        print(f"[SKIP] {title}: usable chunks 없음")
        return

    save_dir = os.path.join(OUT_DIR, f"idx_{title}")
    ensure_dir(save_dir)

    # 개행/캐리지리턴 정리 후 인코딩
    docs = [c["embedding_text"].replace("\r", " ").strip() for c in usable]
    with torch.inference_mode():
        vecs = model.encode(
            docs,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    mat = np.asarray(vecs, dtype=np.float32)

    # FAISS 인덱스: 내적(IP) → 정규화 벡터 사용 시 코사인 유사도와 동일
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    # 산출물 저장
    np.save(os.path.join(save_dir, "vectors.npy"), mat)
    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))

    # chunks 최소본 저장(검색결과-본문 매핑용)
    idx_chunks = [{
        "source": c.get("source",""),
        "text": c.get("text", []),
        "embedding_text": c.get("embedding_text",""),
        "keywords": c.get("keywords", [])
    } for c in usable]
    with open(os.path.join(save_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(idx_chunks,  f, ensure_ascii=False, indent=2)

    print(f"[OK] {title}: idx_{title}/index.faiss, vectors.npy, chunks.json 생성")


# ---------------------------------------
# 6) 메인 파이프라인
#    (1) 입력 스캔 → 파일별 청크 생성/저장
#    (2) 전체 청크 저장
#    (3) 임베딩 모델 로드
#    (4) 파일별 문서 인덱스 생성
#    - 스킵 로직:
#        * 문서별 산출물 3종(chunks.json/index.faiss/vectors.npy)이 모두 있으면 해당 문서는 스킵
#        * idx_all 역시 3종 모두 존재 시 스킵
#    - 최적화:
#        * 모든 것이 최신이면 임베딩 모델 로드를 수행하지 않고 즉시 종료
# ---------------------------------------

def main():
    ensure_dir(OUT_DIR)

    # 입력 파일 목록 수집(.json, .jsonl)
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.json")) + glob.glob(os.path.join(IN_DIR, "*.jsonl")))
    if not files:
        # 설계: 치명적 조건 → 예외 발생시켜 조기 인지
        raise FileNotFoundError(f"입력 폴더에 JSON/JSONL 없음: {IN_DIR}")

    # (1) 파일별 청크 생성 -> 곧바로 idx/<제목>/chunks.json(풀)로 저장
    idx_paths = []        # 인덱스 생성이 필요한 (title, chunks_path) 목록
    all_chunks_mem = []   # 전체 인덱스용 누적 버퍼(idx_all/chunks.json 용)
    skip_all = True       # 모든 문서/idx_all이 최신이면 전체 스킵

    for p in files:
        title = os.path.basename(p).rsplit(".", 1)[0]
        idx_dir = os.path.join(OUT_DIR, f"idx_{title}")
        idx_chunks_path = os.path.join(idx_dir, "chunks.json")
        idx_index_path  = os.path.join(idx_dir, "index.faiss")
        idx_vecs_path   = os.path.join(idx_dir, "vectors.npy")

        # 문서별: 산출물 3종 모두 있으면 스킵(재계산 비용 절약)
        if os.path.isfile(idx_chunks_path) and os.path.isfile(idx_index_path) and os.path.isfile(idx_vecs_path):
            print(f"[SKIP] {title}: 기존 인덱스가 있어 건너뜀")
            # idx_all 업데이트 대비: 기존 청크를 메모리에 누적(가능하면)
            try:
                with open(idx_chunks_path, "r", encoding="utf-8") as f:
                    old_chunks = json.load(f)
                if isinstance(old_chunks, list):
                    all_chunks_mem.extend(old_chunks)
            except Exception as e:
                print(f"[경고] {title}: 기존 chunks.json 읽기 실패 → {e}")
            continue

        # 신규/미완성 → 작업 필요
        skip_all = False
        try:
            chunks = build_chunks_from_file(p)
        except Exception as e:
            # 한 파일 오류가 전체 파이프라인을 막지 않도록 예외 흡수
            print(f"[오류] {os.path.basename(p)} → {e}")
            chunks = []

        ensure_dir(idx_dir)
        with open(idx_chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        if chunks:
            idx_paths.append((title, idx_chunks_path))
            all_chunks_mem.extend(chunks)

    print("\n[완료] idx 디렉터리별 chunks.json 저장 완료")

    # 설계 의도: 개별 문서 청크를 그대로 합쳐 idx_all/chunks.json 생성 → 전역 검색 용이
    all_dir = os.path.join(OUT_DIR, "idx_all")
    all_chunks_path = os.path.join(all_dir, "chunks.json")
    all_index_path  = os.path.join(all_dir, "index.faiss")
    all_vecs_path   = os.path.join(all_dir, "vectors.npy")

    #  전체(idx_all): 산출물 3종 모두 있으면 스킵
    if os.path.isfile(all_chunks_path) and os.path.isfile(all_index_path) and os.path.isfile(all_vecs_path):
        print("[SKIP] idx_all: 기존 인덱스가 있어 건너뜀")
    else:
        if all_chunks_mem:
            ensure_dir(all_dir)
            with open(all_chunks_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks_mem, f, ensure_ascii=False, indent=2)
            idx_paths.append(("all", all_chunks_path))
            skip_all = False

    # 전부 최신이면 임베딩 모델 로드조차 하지 않고 종료(시간/메모리 절약)
    if not idx_paths and skip_all:
        print("[SKIP] 모든 문서와 idx_all이 이미 최신 상태입니다. 작업을 종료합니다.")
        return

    # (3) 임베딩 모델 로드
    # 참고: GPU가 있으면 'cuda', 없으면 'cpu'로 자동 선택
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 임베딩 모델 로드: {EMBED_MODEL_NAME} ({device})")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    # (4) 문서별 인덱스 생성 (새로 만든/갱신 필요한 것만)
    for title, idx_path in idx_paths:
        try:
            build_index_for_document(title, idx_path, model)
        except Exception as e:
            # 인덱싱 실패 개별 로그 후 다음 문서 진행
            print(f"[ERR] {title}: {e}")




if __name__ == "__main__":
    # 진입점: 모듈로 임포트되는 경우 실행되지 않도록 보호
    main()
