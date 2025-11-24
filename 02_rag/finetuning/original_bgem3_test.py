# -*- coding: utf-8 -*-
"""
eval_zeroshot.py
- 목적: 튜닝 없이(학습 0) 기본 백본 모델의 성능을 기존 파이프라인과 "동일한" 스플릿(outer 5-fold)로 측정
- 출력: zeroshot_detail.csv  (results_detail.csv와 동일 스키마)
- 이후 baseline_summarize.py로 요약/비교 가능
"""

import os, re, json, math, hashlib, random
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# ======= 기존 실험과 동일한 경로/상수로 맞춰주세요 =======
QNA_TXT_PATH      = "./qnalist.txt"
LABEL_JSONL_PATH  = "./triplets_group_bgem3.jsonl"
CHUNKS_JSON_PATH  = "/home/ubuntu/parksekyeong/idx_singlelevel/idx_전체/chunks.json"
OUTPUT_DETAIL     = "./zeroshot_detail.csv"

MODEL_NAME        = "dragonkue/BGE-m3-ko"
PROMPT_Q          = "query: "
PROMPT_P          = "passage: "
USE_GPU           = torch.cuda.is_available()
EVAL_BATCH_CPU    = 256
OUTER_K_FOLDS     = 5

# ======= 텍스트 전처리/입출력 =======
def norm_text(s: str) -> str:
    if s is None: return ""
    s = s.replace("\t"," ").replace("\r"," ").replace("\n"," ")
    return re.sub(r"\s+", " ", s).strip()

def read_qna(path: str):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = norm_text(line)
            if t: queries.append(t)
    q2id = {q: i for i, q in enumerate(queries)}
    # 문서 그룹: 10문항=1문서 그룹 가정
    doc_ids = [i // 10 for i in range(len(queries))]
    return queries, q2id, doc_ids

def read_labels_jsonl(path: str):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                rec["query"]     = norm_text(rec.get("query", ""))
                rec["positives"] = [norm_text(x) for x in rec.get("positives", []) if norm_text(x)]
                rec["negatives"] = [norm_text(x) for x in rec.get("negatives", []) if norm_text(x)]
                recs.append(rec)
            except Exception as e:
                print("[WARN] JSONL parse fail:", e)
    return recs

def match_labels_to_qids(queries, q2id, label_recs):
    qid2 = defaultdict(lambda: {"positives":[], "negatives":[]})
    for rec in label_recs:
        q = rec.get("query","")
        qid = q2id.get(q)
        if qid is None: 
            continue
        qid2[qid]["positives"].extend(rec.get("positives",[]))
        qid2[qid]["negatives"].extend(rec.get("negatives",[]))
    return dict(qid2)

# ======= 코퍼스(미리 청킹된 조문) =======
def load_prechunked_corpus(chunks_json_path, prompt_p=PROMPT_P):
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    corpus, txt2pids, pid2doc = {}, {}, {}
    for i, ch in enumerate(chunks):
        body = (ch.get("text") or ch.get("enriched_text") or "").strip()
        if not body:
            continue
        pid = f"P{i:06d}"
        corpus[pid] = prompt_p + body
        # 조문ID(=source 해시)
        doc_id = hashlib.md5((ch.get("source","")).encode("utf-8")).hexdigest()
        pid2doc[pid] = doc_id
        txt2pids.setdefault(body, []).append(pid)
    return corpus, txt2pids, pid2doc

# ======= 라벨 텍스트 → 청크 매칭 =======
def _tokset(s: str):
    import re
    return set(re.findall(r"[가-힣A-Za-z0-9]+", s or ""))

def match_pids_for_label_text(label_text: str, txt2pids: dict, jaccard_thr=0.6):
    t = (label_text or "").strip()
    if not t:
        return []
    if t in txt2pids:
        return txt2pids[t]
    # 부분 포함
    cands = []
    for body in txt2pids.keys():
        if t in body or body in t:
            cands.extend(txt2pids[body])
    if cands:
        return cands
    # 자카드 유사도
    tset = _tokset(t)
    best, best_sim = [], 0.0
    for body, pids in txt2pids.items():
        bset = _tokset(body)
        inter = len(tset & bset); union = len(tset | bset) or 1
        sim = inter / union
        if sim > best_sim:
            best, best_sim = pids, sim
    return best if best_sim >= jaccard_thr else []

# ======= 평가 구조 만들기 =======
def build_dev_struct_from_prechunks(qids_dev, queries, qid2posneg, corpus, txt2pids, prompt_q=PROMPT_Q):
    q_dict, relevant_pids = {}, {}
    for qid in qids_dev:
        qkey = str(qid)
        q_dict[qkey] = prompt_q + queries[qid]
        rel = set()
        for t in qid2posneg.get(qid, {}).get("positives", []):
            for pid in match_pids_for_label_text(t, txt2pids):
                rel.add(pid)
        if rel:
            relevant_pids[qkey] = rel
    return q_dict, corpus, relevant_pids

# ======= IR Metrics =======
@dataclass
class IRMetrics:
    hit1: float; hit3: float; hit5: float
    mrr: float; ndcg3: float; ndcg5: float
    mfr: float; n_query: int

def _dcg_at_k(gains: List[int], k: int) -> float:
    dcg = 0.0
    for i, g in enumerate(gains[:k], start=1):
        dcg += (2**g - 1) / math.log2(i + 1)
    return dcg

def _ndcg_at_k(gains: List[int], k: int) -> float:
    dcg = _dcg_at_k(gains, k)
    ideal = _dcg_at_k(sorted(gains, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0

_P_EMB_CACHE: Dict[Tuple[int, Tuple[str,...]], torch.Tensor] = {}

def _truncate_list(texts, tokenizer, max_len):
    out = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out

@torch.no_grad()
def precompute_corpus_embeddings(model, corpus_texts, batch_size=256, use_gpu=True):
    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    emb = model.encode(
        corpus_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device
    )
    return emb.to('cpu')

@torch.no_grad()
def compute_rank_metrics_doclevel_cached(
    model: SentenceTransformer,
    queries: Dict[str,str],    # {qid: "query: ..."}
    corpus: Dict[str,str],     # {pid: "passage: ..."}
    relevant_pids: Dict[str,Set[str]],   # {qid: {pid,...}}
    pid2doc: Dict[str,str],    # pid -> doc_id
    tokenizer,
    max_len_eval: int,
    batch_size_cpu: int = EVAL_BATCH_CPU
) -> IRMetrics:
    if not queries: return IRMetrics(0,0,0,0,0,0,0,0)

    p_ids   = tuple(corpus.keys())
    p_texts = _truncate_list([corpus[pid] for pid in p_ids], tokenizer, max_len_eval)

    cache_key = (id(model), p_ids)
    if cache_key not in _P_EMB_CACHE:
        _P_EMB_CACHE[cache_key] = precompute_corpus_embeddings(model, p_texts, batch_size=256, use_gpu=True)
    p_emb = _P_EMB_CACHE[cache_key]  # CPU

    q_ids   = list(queries.keys())
    q_texts = _truncate_list([queries[qid] for qid in q_ids], tokenizer, max_len_eval)

    q_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q_emb = model.encode(
        q_texts,
        batch_size=batch_size_cpu,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=q_device
    ).to('cpu')

    sim = q_emb.float() @ p_emb.float().T  # cosine (normed)

    hit1=hit3=hit5=0; mrr_sum=ndcg3_sum=ndcg5_sum=0.0; mfr_sum=0.0; valid_q=0
    for i, qid in enumerate(q_ids):
        rel_p = relevant_pids.get(qid)
        if not rel_p: 
            continue
        valid_q += 1

        scores = sim[i].tolist()
        from collections import defaultdict as dd
        doc_score = dd(lambda: -1e9)
        doc_rel   = dd(int)
        for j, pid in enumerate(p_ids):
            d = pid2doc[pid]
            if scores[j] > doc_score[d]:
                doc_score[d] = scores[j]
            if pid in rel_p:
                doc_rel[d] = 1

        ordered_docs = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
        gains = [doc_rel[d] for d,_ in ordered_docs]

        if any(gains[:1]): hit1 += 1
        if any(gains[:3]): hit3 += 1
        if any(gains[:5]): hit5 += 1
        try:
            first_pos = gains.index(1) + 1
            mrr_sum += 1.0 / first_pos
            mfr_sum += float(first_pos)
        except ValueError:
            mfr_sum += float(len(gains)+1)
        ndcg3_sum += _ndcg_at_k(gains, 3)
        ndcg5_sum += _ndcg_at_k(gains, 5)

    if valid_q == 0: 
        return IRMetrics(0,0,0,0,0,0,0,0)
    return IRMetrics(hit1/valid_q, hit3/valid_q, hit5/valid_q,
                     mrr_sum/valid_q, ndcg3_sum/valid_q, ndcg5_sum/valid_q,
                     mfr_sum/valid_q, valid_q)

# ======= 메인 =======
def main():
    import csv

    # 재현성 고정 (평가라 큰 영향 없지만 통일)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    queries, q2id, doc_ids = read_qna(QNA_TXT_PATH)
    label_recs = read_labels_jsonl(LABEL_JSONL_PATH)
    qid2posneg = match_labels_to_qids(queries, q2id, label_recs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_max = getattr(tokenizer, "model_max_length", 1024)
    safe_len  = min(512, model_max - 8)

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = safe_len

    corpus, txt2pids, pid2doc = load_prechunked_corpus(CHUNKS_JSON_PATH)
    print(f"[INFO] pre-chunked corpus loaded: {len(corpus)} chunks")

    # Outer 5-fold (문서 그룹 단위)
    idx_all = np.arange(len(queries))
    groups  = np.array(doc_ids)
    outer_gkf = GroupKFold(n_splits=OUTER_K_FOLDS)

    rows = []
    CONFIG_KEY = "base_zeroshot"  # 튜닝 없이 고정

    for outer_fold, (tr_idx, te_idx) in enumerate(outer_gkf.split(idx_all, groups=groups), start=1):
        # 평가에 쓸 outer test 쿼리들(라벨 있는 것만)
        qids_outer_test = [i for i in te_idx if i in qid2posneg and qid2posneg[i].get("positives")]

        if not qids_outer_test:
            print(f"[WARN] outer_fold{outer_fold}: empty test set (skip)")
            continue

        test_struct = build_dev_struct_from_prechunks(qids_outer_test, queries, qid2posneg, corpus, txt2pids)
        metrics = compute_rank_metrics_doclevel_cached(
            model, *test_struct, pid2doc, tokenizer, max_len_eval=safe_len, batch_size_cpu=EVAL_BATCH_CPU
        )

        rows.append(dict(
            phase="test", outer_fold=outer_fold, seed="", config_key=CONFIG_KEY,
            loss_main="", epochs=0, temperature="", margin="",
            lr=0.0, warmup_ratio=0.0, weight_decay=0.0,
            hit1=metrics.hit1, hit3=metrics.hit3, hit5=metrics.hit5,
            mrr=metrics.mrr, ndcg3=metrics.ndcg3, ndcg5=metrics.ndcg5,
            mfr=metrics.mfr, nq=metrics.n_query
        ))
        print(f"[ZEROSHOT OUTER {outer_fold}] Hit@1={metrics.hit1:.3f}, Hit@3={metrics.hit3:.3f}, MRR={metrics.mrr:.3f}")

    # 저장 (results_detail 포맷과 동일)
    if rows:
        with open(OUTPUT_DETAIL, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[SAVE] zeroshot detail: {OUTPUT_DETAIL}")
    else:
        print("[WARN] no rows to save (no valid outer test folds)")

if __name__ == "__main__":
    main()
