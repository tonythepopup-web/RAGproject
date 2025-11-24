# -*- coding: utf-8 -*-
"""
Nested CV + Article-Chunk Retrieval (pre-chunked)
- 입력 데이터:
  * qnalist.txt (질문 100개, 10개씩 1문서 그룹 가정)
  * triplets_group_bgem3.jsonl (query, positives[], negatives[])
  * chunks.json (이미 '조문 단위'로 청킹된 파일)  <-- 재청킹 없음!
- 백본: dragonkue/BGE-m3-ko (SentenceTransformer)
- 손실:
  A) mp_infonce (멀티 포지티브/네거티브 동시 최적화)
  B) cosine+triplet (트리플릿 손실 + 코사인 손실)
- 평가: 청크 스코어 → 조문 단위로 max 집계하여 Hit@K, MRR, nDCG 계산
- 최적화:
  * AMP(torch.amp) 사용
  * 모델 인스턴스별 코퍼스 임베딩 캐시 (id(model) 기준)
  * Triplet neg 샘플링 상한
  * safe_len으로 프롬프트 포함 토큰 길이 강제 트렁케이션
- 저장: CSV만 (results_detail.csv, results_summary.csv). 모델/임베딩 저장 없음.
"""

import os, re, json, math, random, csv, hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer, InputExample, losses
from transformers import AutoTokenizer
from sklearn.model_selection import GroupKFold

# =========================
# 경로/고정
# =========================
QNA_TXT_PATH      = "./qnalist.txt"
LABEL_JSONL_PATH  = "./triplets_group_bgem3.jsonl"
# 이미 만들어 둔 조문 청크 파일 경로로 바꿔주세요.
CHUNKS_JSON_PATH  = "/home/ubuntu/parksekyeong/idx_singlelevel/idx_전체/chunks.json"

OUTPUT_DIR        = "./runs_nested_cv_article_prechunk"
MODEL_NAME        = "dragonkue/BGE-m3-ko"

USE_GPU           = torch.cuda.is_available()
PROMPT_Q          = "query: "
PROMPT_P          = "passage: "

# ---- 학습 예제 상한 ----
MAX_POS_CHUNKS_TOTAL    = 12   # mp_infonce용: 한 쿼리의 pos 총합 상한
MAX_NEG_CHUNKS_TOTAL    = 24   # 공통: 한 쿼리의 neg 총합 상한
MAX_TRIPLET_NEG_PER_POS = 6    # triplet: pos 1개 당 neg 샘플 상한

# ---- 평가 ----
EVAL_BATCH_CPU    = 256  # encode 배치 (q_emb)

# ---- Nested CV ----
OUTER_K_FOLDS     = 5
INNER_SEEDS       = [42, 123, 2025]

# ---- 그리드 (12개) ----
GRID = [
    # mp_infonce (epochs × temperature)
    {"loss_main":"mp_infonce","epochs":1,"temperature":0.05,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.01},
    {"loss_main":"mp_infonce","epochs":2,"temperature":0.05,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.01},
    {"loss_main":"mp_infonce","epochs":3,"temperature":0.05,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.01},
    {"loss_main":"mp_infonce","epochs":1,"temperature":0.10,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.01},
    {"loss_main":"mp_infonce","epochs":2,"temperature":0.10,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.01},
    {"loss_main":"mp_infonce","epochs":3,"temperature":0.10,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.01},

    # cosine+triplet (epochs × margin)
    {"loss_main":"cosine+triplet","epochs":1,"margin":0.2,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.0,"batch":16},
    {"loss_main":"cosine+triplet","epochs":2,"margin":0.2,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.0,"batch":16},
    {"loss_main":"cosine+triplet","epochs":3,"margin":0.2,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.0,"batch":16},
    {"loss_main":"cosine+triplet","epochs":1,"margin":0.3,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.0,"batch":16},
    {"loss_main":"cosine+triplet","epochs":2,"margin":0.3,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.0,"batch":16},
    {"loss_main":"cosine+triplet","epochs":3,"margin":0.3,"lr":2e-5,"warmup_ratio":0.1,"weight_decay":0.0,"batch":16},
]

# =========================
# 유틸/입출력
# =========================
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
    if len(queries) != 100:
        print(f"[WARN] qnalist 개수={len(queries)} (기대: 100). 계속 진행합니다.")
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
                print("[WARN] JSONL 파싱 실패:", e)
    return recs

def match_labels_to_qids(queries, q2id, label_recs):
    qid2 = defaultdict(lambda: {"positives":[], "negatives":[]})
    miss = 0
    for rec in label_recs:
        q = rec.get("query","")
        qid = q2id.get(q)
        if qid is None:
            miss += 1
            continue
        qid2[qid]["positives"].extend(rec.get("positives",[]))
        qid2[qid]["negatives"].extend(rec.get("negatives",[]))
    if miss:
        print(f"[WARN] qnalist와 매치 실패 라벨 {miss}줄")
    return dict(qid2)

# =========================
# 조문 청크 로더 (재청킹 없음)
# =========================
def load_prechunked_corpus(chunks_json_path, prompt_p=PROMPT_P):
    """
    chunks.json을 읽어 corpus/txt2pids/pid2doc 생성.
    - corpus   : {pid: 'passage: <조문 본문 텍스트>'}
    - txt2pids : {본문(str): [pid...]}
    - pid2doc  : {pid: 조문ID(=source 해시)}  # Hit@K를 '조문' 기준으로 집계
    """
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    corpus, txt2pids, pid2doc = {}, {}, {}
    for i, ch in enumerate(chunks):
        body = (ch.get("text") or ch.get("enriched_text") or "").strip()
        if not body:
            continue
        pid = f"P{i:06d}"
        corpus[pid] = prompt_p + body
        doc_id = hashlib.md5((ch.get("source","")).encode("utf-8")).hexdigest()
        pid2doc[pid] = doc_id
        txt2pids.setdefault(body, []).append(pid)
    return corpus, txt2pids, pid2doc

# =========================
# 라벨 텍스트 → 조문 청크 매칭 (재청킹 없이 매칭만)
# =========================
def _tokset(s: str):
    return set(re.findall(r"[가-힣A-Za-z0-9]+", s or ""))

def match_pids_for_label_text(label_text: str, txt2pids: dict, jaccard_thr=0.6):
    t = (label_text or "").strip()
    if not t:
        return []
    # 1) 완전 일치
    if t in txt2pids:
        return txt2pids[t]
    # 2) 부분 포함
    cands = []
    for body in txt2pids.keys():
        if t in body or body in t:
            cands.extend(txt2pids[body])
    if cands:
        return cands
    # 3) 토큰 Jaccard
    tset = _tokset(t)
    best, best_sim = [], 0.0
    for body, pids in txt2pids.items():
        bset = _tokset(body)
        inter = len(tset & bset); union = len(tset | bset) or 1
        sim = inter / union
        if sim > best_sim:
            best, best_sim = pids, sim
    return best if best_sim >= jaccard_thr else []

# =========================
# Dev/Test 구조 (조문 청크 기반)
# =========================
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

# =========================
# 학습 예제 (조문 청크 기반)
# =========================
def build_examples_from_prechunks(qids_train, queries, qid2posneg, txt2pids, corpus, prompt_q=PROMPT_Q):
    pairs_cos, triplets = [], []
    for qid in qids_train:
        qtext = prompt_q + queries[qid]
        pos_chunks, neg_chunks = [], []
        for t in qid2posneg.get(qid, {}).get("positives", []):
            for pid in match_pids_for_label_text(t, txt2pids):
                pos_chunks.append(corpus[pid])
        for t in qid2posneg.get(qid, {}).get("negatives", []):
            for pid in match_pids_for_label_text(t, txt2pids):
                neg_chunks.append(corpus[pid])

        # 상한 적용
        if MAX_POS_CHUNKS_TOTAL and len(pos_chunks) > MAX_POS_CHUNKS_TOTAL:
            pos_chunks = pos_chunks[:MAX_POS_CHUNKS_TOTAL]
        if MAX_NEG_CHUNKS_TOTAL and len(neg_chunks) > MAX_NEG_CHUNKS_TOTAL:
            neg_chunks = neg_chunks[:MAX_NEG_CHUNKS_TOTAL]

        # cosine
        for p in pos_chunks: pairs_cos.append(InputExample(texts=[qtext, p], label=1.0))
        for n in neg_chunks: pairs_cos.append(InputExample(texts=[qtext, n], label=0.0))

        # triplet (샘플링)
        if pos_chunks and neg_chunks:
            for p in pos_chunks:
                k = min(len(neg_chunks), MAX_TRIPLET_NEG_PER_POS)
                neg_sample = random.sample(neg_chunks, k) if k>0 else []
                for n in neg_sample:
                    triplets.append(InputExample(texts=[qtext, p, n]))
    return pairs_cos, triplets

def build_qid2chunks_for_mp(qids_train, queries, qid2posneg, txt2pids, corpus, prompt_q=PROMPT_Q):
    """
    mp_infonce용 qid -> {pos_chunks:[], neg_chunks:[]} 생성
    """
    out = {}
    for qid in qids_train:
        pos_chunks, neg_chunks = [], []
        for t in qid2posneg.get(qid, {}).get("positives", []):
            for pid in match_pids_for_label_text(t, txt2pids):
                pos_chunks.append(corpus[pid])
        for t in qid2posneg.get(qid, {}).get("negatives", []):
            for pid in match_pids_for_label_text(t, txt2pids):
                neg_chunks.append(corpus[pid])

        if MAX_POS_CHUNKS_TOTAL and len(pos_chunks) > MAX_POS_CHUNKS_TOTAL:
            pos_chunks = pos_chunks[:MAX_POS_CHUNKS_TOTAL]
        if MAX_NEG_CHUNKS_TOTAL and len(neg_chunks) > MAX_NEG_CHUNKS_TOTAL:
            neg_chunks = neg_chunks[:MAX_NEG_CHUNKS_TOTAL]
        out[qid] = {"pos_chunks": pos_chunks, "neg_chunks": neg_chunks}
    return out

# =========================
# IR Metrics (문서=조문 단위 집계) + 코퍼스 임베딩 캐시
# =========================
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

# 코퍼스 임베딩 캐시: key=(id(model), p_ids_tuple)
_P_EMB_CACHE: Dict[Tuple[int, Tuple[str,...]], torch.Tensor] = {}

def _truncate_list(texts, tokenizer, max_len):
    out = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out

def precompute_corpus_embeddings(model, corpus_texts, batch_size=256, use_gpu=True):
    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    with torch.no_grad():
        emb = model.encode(
            corpus_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=device
        )
    # 캐시는 CPU에 저장 (메모리 절약을 원하면 .half() 후 matmul 전에 float())
    return emb.to('cpu')

def compute_rank_metrics_doclevel_cached(
    model: SentenceTransformer,
    queries: Dict[str,str],              # {qid: "query: ..."}
    corpus: Dict[str,str],               # {pid: "passage: ..."}
    relevant_pids: Dict[str,Set[str]],   # {qid: {pid,...}}
    pid2doc: Dict[str,str],              # pid -> doc_id(조문ID)
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
    p_emb = _P_EMB_CACHE[cache_key]  # [np, H] CPU

    q_ids   = list(queries.keys())
    q_texts = _truncate_list([queries[qid] for qid in q_ids], tokenizer, max_len_eval)

    q_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        q_emb = model.encode(
            q_texts,
            batch_size=batch_size_cpu,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=q_device
        ).to('cpu')  # 보통 fp32

    # 코사인(정규화 가정) → 내적
    sim = q_emb.float() @ p_emb.float().T  # [nq, np]

    hit1=hit3=hit5=0; mrr_sum=ndcg3_sum=ndcg5_sum=0.0; mfr_sum=0.0; valid_q=0
    for i, qid in enumerate(q_ids):
        rel_p = relevant_pids.get(qid)
        if not rel_p: continue
        valid_q += 1

        scores = sim[i].tolist()
        doc_score = defaultdict(lambda: -1e9)
        doc_rel   = defaultdict(int)
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

    if valid_q == 0: return IRMetrics(0,0,0,0,0,0,0,0)
    return IRMetrics(hit1/valid_q, hit3/valid_q, hit5/valid_q,
                     mrr_sum/valid_q, ndcg3_sum/valid_q, ndcg5_sum/valid_q,
                     mfr_sum/valid_q, valid_q)

# =========================
# mp_infonce
# =========================
class MultiPositiveInfoNCELoss(nn.Module):
    def __init__(self, model, temperature: float = 0.05):
        super().__init__(); self.model = model; self.tau = float(temperature)
    def forward(self, sentence_features, labels):
        out_q = self.model(sentence_features[0])['sentence_embedding']  # (1,H)
        out_p = self.model(sentence_features[1])['sentence_embedding']  # (B,H)
        q_emb = F.normalize(out_q, dim=-1); p_emb = F.normalize(out_p, dim=-1)
        logits = (q_emb @ p_emb.t()) / self.tau      # (1,B)
        pos_mask = labels.bool()                     # (1,B)
        if not pos_mask.any(): return logits.new_tensor(0.0)
        log_den = torch.logsumexp(logits, dim=1)
        logits_pos = torch.where(pos_mask, logits, torch.full_like(logits, float('-inf')))
        log_num = torch.logsumexp(logits_pos, dim=1)
        return -(log_num - log_den).mean()

class MPDataset(Dataset):
    """qid → (query_text, pos_chunk_texts[], neg_chunk_texts[])"""
    def __init__(self, qids, queries, qid2chunks, prompt_q=PROMPT_Q):
        self.items = []
        for qid in qids:
            chunks = qid2chunks.get(qid, {"pos_chunks":[], "neg_chunks":[]})
            pos = [t for t in chunks["pos_chunks"] if t]
            neg = [t for t in chunks["neg_chunks"] if t]
            if not pos:
                continue
            if MAX_POS_CHUNKS_TOTAL and len(pos) > MAX_POS_CHUNKS_TOTAL:
                pos = pos[:MAX_POS_CHUNKS_TOTAL]
            if MAX_NEG_CHUNKS_TOTAL and len(neg) > MAX_NEG_CHUNKS_TOTAL:
                neg = neg[:MAX_NEG_CHUNKS_TOTAL]
            q_text = prompt_q + queries[qid]
            self.items.append((qid, q_text, pos, neg))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def mp_collate_fn(batch, tokenizer, max_seq_len):
    assert len(batch) == 1, "Use DataLoader(..., batch_size=1)"
    _, q_text, pos_list, neg_list = batch[0]
    passages = pos_list + neg_list
    B = len(passages)
    pos_mask = torch.zeros((1, B), dtype=torch.bool)
    pos_mask[0, :len(pos_list)] = True

    q_feat = tokenizer(q_text, padding='max_length', truncation=True, max_length=max_seq_len, return_tensors='pt')
    p_feat = tokenizer(passages, padding='max_length', truncation=True, max_length=max_seq_len, return_tensors='pt')
    features_q = {'input_ids': q_feat['input_ids'], 'attention_mask': q_feat['attention_mask']}
    features_p = {'input_ids': p_feat['input_ids'], 'attention_mask': p_feat['attention_mask']}
    return [features_q, features_p], pos_mask

def train_mp_once(model, tokenizer, qids_train, queries, qid2chunks, cfg, max_seq_len):
    device = torch.device("cuda") if USE_GPU else torch.device("cpu")
    model = model.to(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except: pass
    try:
        model._first_module().auto_model.gradient_checkpointing_enable()
    except Exception:
        pass

    ds = MPDataset(qids_train, queries, qid2chunks, PROMPT_Q)
    ld = DataLoader(ds, shuffle=True, batch_size=1,
                    collate_fn=lambda b: mp_collate_fn(b, tokenizer, max_seq_len),
                    num_workers=0, pin_memory=False)

    criterion = MultiPositiveInfoNCELoss(model, temperature=float(cfg["temperature"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    steps = len(ld) * int(cfg["epochs"])
    warmup_steps = int(steps * float(cfg["warmup_ratio"]))
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    from torch.amp import GradScaler, autocast
    scaler = GradScaler("cuda") if USE_GPU else None

    model.train()
    for _ in range(int(cfg["epochs"])):
        for features, pos_mask in ld:
            features = [{k: v.to(device, non_blocking=True) for k, v in f.items()} for f in features]
            pos_mask = pos_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=USE_GPU):
                loss = criterion(features, pos_mask)
            if USE_GPU:
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            scheduler.step()

    return model

# =========================
# cosine + triplet
# =========================
def train_cosine_triplet_once(model, examples_cos, examples_triplet, cfg):
    if USE_GPU:
        model = model.to(torch.device("cuda"))
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except: pass
        try:
            model._first_module().auto_model.gradient_checkpointing_enable()
        except Exception:
            pass

    train_objectives = []
    loader_kwargs = dict(num_workers=0, pin_memory=False)
    if examples_cos:
        dl_cos = DataLoader(examples_cos, shuffle=True, batch_size=int(cfg["batch"]), **loader_kwargs)
        train_objectives.append((dl_cos, losses.CosineSimilarityLoss(model)))
    if examples_triplet:
        dl_tri = DataLoader(examples_triplet, shuffle=True, batch_size=max(8, int(cfg["batch"])//2), **loader_kwargs)
        try:
            tloss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=float(cfg["margin"]))
        except TypeError:
            tloss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, margin=float(cfg["margin"]))
        train_objectives.append((dl_tri, tloss))
    if not train_objectives:
        return model

    steps_per_epoch = sum(len(ld) for ld,_ in train_objectives)
    warmup_steps = math.ceil(steps_per_epoch * int(cfg["epochs"]) * float(cfg["warmup_ratio"]))
    model.fit(
        train_objectives=train_objectives,
        epochs=int(cfg["epochs"]),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(cfg["lr"])},
        weight_decay=float(cfg["weight_decay"]),
        output_path=None,  # 저장 안 함
        use_amp=True
    )
    return model

# =========================
# 메인 (Nested CV)
# =========================
def make_docwise_split(queries: List[str], num_dev_per_doc: int = 2, seed: int = 42):
    rng = random.Random(seed)
    doc2qids = defaultdict(list)
    for qid in range(len(queries)):
        doc2qids[qid // 10].append(qid)
    train, dev = [], []
    for _, qids in sorted(doc2qids.items()):
        dv = rng.sample(qids, num_dev_per_doc)
        tr = [q for q in qids if q not in dv]
        dev.extend(dv); train.extend(tr)
    return train, dev

def main():
    # 재현성
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if USE_GPU: torch.cuda.manual_seed_all(42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[INFO] Device:", torch.cuda.get_device_name(0) if USE_GPU else "CPU only")

    # 데이터 적재
    queries, q2id, doc_ids = read_qna(QNA_TXT_PATH)
    label_recs = read_labels_jsonl(LABEL_JSONL_PATH)
    qid2posneg = match_labels_to_qids(queries, q2id, label_recs)
    if not qid2posneg:
        print("[ERR] 라벨 매칭 결과 없음"); return

    # 토크나이저 & 모델 & 안전 max_len
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_max = getattr(tokenizer, "model_max_length", 1024)
    safe_len  = min(512, model_max - 8)  # 프리픽스 포함 여유
    # pre-trained 모델 하나를 먼저 로딩(튜닝용 모델은 fold별로 새로 만듦)
    base_model = SentenceTransformer(MODEL_NAME)
    base_model.max_seq_length = safe_len

    # 조문 청크 로드 (재청킹 없음)
    corpus, txt2pids, pid2doc = load_prechunked_corpus(CHUNKS_JSON_PATH)
    print(f"[INFO] pre-chunked corpus loaded: {len(corpus)} chunks")

    # 바깥 5-fold (문서 그룹)
    outer_gkf = GroupKFold(n_splits=OUTER_K_FOLDS)
    idx_all = np.arange(len(queries))
    groups  = np.array(doc_ids)

    rows = []  # detail

    for outer_fold, (tr_idx, te_idx) in enumerate(outer_gkf.split(idx_all, groups=groups), start=1):
        qids_outer_train = [i for i in tr_idx if i in qid2posneg]
        qids_outer_test  = [i for i in te_idx if i in qid2posneg and qid2posneg[i].get("positives")]
        if not qids_outer_train or not qids_outer_test:
            print(f"[WARN] outer_fold{outer_fold}: empty train/test. skip")
            continue

        tune_scores = defaultdict(list)  # cfg_key -> [ (hit3, mrr), ... ]

        for cfg_id, cfg in enumerate(GRID, start=1):
            cfg_key = f"cfg{cfg_id}_{cfg['loss_main']}_e{cfg['epochs']}" + \
                      (f"_t{cfg['temperature']}" if cfg["loss_main"]=="mp_infonce" else f"_m{cfg['margin']}")

            for sd in INNER_SEEDS:
                inner_tr, inner_dv = make_docwise_split(queries, num_dev_per_doc=2, seed=sd)
                inner_tr = [i for i in inner_tr if i in qids_outer_train and i in qid2posneg]
                inner_dv = [i for i in inner_dv if i in qids_outer_train and i in qid2posneg and qid2posneg[i].get("positives")]
                if not inner_tr or not inner_dv:
                    continue

                # fresh model per trial
                model_t = SentenceTransformer(MODEL_NAME)
                model_t.max_seq_length = safe_len

                if cfg["loss_main"] == "mp_infonce":
                    qid2chunks_mp = build_qid2chunks_for_mp(inner_tr, queries, qid2posneg, txt2pids, corpus)
                    model_t = train_mp_once(model_t, tokenizer, inner_tr, queries, qid2chunks_mp, cfg, max_seq_len=safe_len)
                else:
                    ex_cos, ex_tri = build_examples_from_prechunks(inner_tr, queries, qid2posneg, txt2pids, corpus)
                    model_t = train_cosine_triplet_once(model_t, ex_cos, ex_tri, cfg)

                dev_struct = build_dev_struct_from_prechunks(inner_dv, queries, qid2posneg, corpus, txt2pids)
                metrics = compute_rank_metrics_doclevel_cached(
                    model_t, *dev_struct, pid2doc, tokenizer, max_len_eval=safe_len, batch_size_cpu=EVAL_BATCH_CPU
                )

                tune_scores[cfg_key].append((metrics.hit3, metrics.mrr))
                rows.append(dict(
                    phase="tune", outer_fold=outer_fold, seed=sd, config_key=cfg_key,
                    loss_main=cfg["loss_main"], epochs=cfg["epochs"],
                    temperature=cfg.get("temperature",""), margin=cfg.get("margin",""),
                    lr=cfg["lr"], warmup_ratio=cfg["warmup_ratio"], weight_decay=cfg["weight_decay"],
                    hit1=metrics.hit1, hit3=metrics.hit3, hit5=metrics.hit5,
                    mrr=metrics.mrr, ndcg3=metrics.ndcg3, ndcg5=metrics.ndcg5,
                    mfr=metrics.mfr, nq=metrics.n_query
                ))

                del model_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not tune_scores:
            print(f"[WARN] outer_fold{outer_fold}: no tuning scores"); continue

        # best config by mean Hit@3 then mean MRR
        best_cfg_key = None; best_pair = (-1.0, -1.0)
        for k, lst in tune_scores.items():
            hit3_mean = float(np.mean([x[0] for x in lst]))
            mrr_mean  = float(np.mean([x[1] for x in lst]))
            pair = (hit3_mean, mrr_mean)
            if pair > best_pair:
                best_pair = pair; best_cfg_key = k

        # config dict 복원
        best_cfg = None
        for cfg_id, cfg in enumerate(GRID, start=1):
            k = f"cfg{cfg_id}_{cfg['loss_main']}_e{cfg['epochs']}" + \
                (f"_t{cfg['temperature']}" if cfg["loss_main"]=="mp_infonce" else f"_m{cfg['margin']}")
            if k == best_cfg_key:
                best_cfg = cfg; break
        if best_cfg is None:
            print(f"[WARN] outer_fold{outer_fold}: best config not found"); continue

        # -------- Final Train on OUTER TRAIN, Test on OUTER TEST --------
        model_f = SentenceTransformer(MODEL_NAME)
        model_f.max_seq_length = safe_len

        if best_cfg["loss_main"] == "mp_infonce":
            qid2chunks_mp = build_qid2chunks_for_mp(qids_outer_train, queries, qid2posneg, txt2pids, corpus)
            model_f = train_mp_once(model_f, tokenizer, qids_outer_train, queries, qid2chunks_mp, best_cfg, max_seq_len=safe_len)
        else:
            ex_cos, ex_tri = build_examples_from_prechunks(qids_outer_train, queries, qid2posneg, txt2pids, corpus)
            model_f = train_cosine_triplet_once(model_f, ex_cos, ex_tri, best_cfg)

        test_struct = build_dev_struct_from_prechunks(qids_outer_test, queries, qid2posneg, corpus, txt2pids)
        metrics_t = compute_rank_metrics_doclevel_cached(
            model_f, *test_struct, pid2doc, tokenizer, max_len_eval=safe_len, batch_size_cpu=EVAL_BATCH_CPU
        )

        rows.append(dict(
            phase="test", outer_fold=outer_fold, seed="", config_key=best_cfg_key,
            loss_main=best_cfg["loss_main"], epochs=best_cfg["epochs"],
            temperature=best_cfg.get("temperature",""), margin=best_cfg.get("margin",""),
            lr=best_cfg["lr"], warmup_ratio=best_cfg["warmup_ratio"], weight_decay=best_cfg["weight_decay"],
            hit1=metrics_t.hit1, hit3=metrics_t.hit3, hit5=metrics_t.hit5,
            mrr=metrics_t.mrr, ndcg3=metrics_t.ndcg3, ndcg5=metrics_t.ndcg5,
            mfr=metrics_t.mfr, nq=metrics_t.n_query
        ))
        print(f"[OUTER {outer_fold}] BEST={best_cfg_key} | Test Hit@1={metrics_t.hit1:.3f}, Hit@3={metrics_t.hit3:.3f}, MRR={metrics_t.mrr:.3f}, nDCG@3={metrics_t.ndcg3:.3f}")

        del model_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ===== CSV 저장 =====
    if rows:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        detail_path = os.path.join(OUTPUT_DIR, "results_detail.csv")
        with open(detail_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[SAVE] 상세 결과: {detail_path}")

        # 요약
        tune_by_cfg = defaultdict(list)
        test_by_cfg = defaultdict(list)
        for r in rows:
            if r["phase"]=="tune": tune_by_cfg[r["config_key"]].append(r)
            elif r["phase"]=="test": test_by_cfg[r["config_key"]].append(r)

        def agg_mean_std(lst, key):
            vals = [float(x[key]) for x in lst]
            return (float(np.mean(vals)) if vals else 0.0,
                    float(np.std(vals))  if vals else 0.0)

        summary = []
        cfg_keys = sorted(set(list(tune_by_cfg.keys()) + list(test_by_cfg.keys())))
        for k in cfg_keys:
            t_runs = tune_by_cfg.get(k, [])
            s_runs = test_by_cfg.get(k, [])
            hit3_t_m, hit3_t_s = agg_mean_std(t_runs, "hit3")
            mrr_t_m,  mrr_t_s  = agg_mean_std(t_runs,  "mrr")
            hit3_s_m, hit3_s_s = agg_mean_std(s_runs, "hit3")
            mrr_s_m,  mrr_s_s  = agg_mean_std(s_runs,  "mrr")
            nq_t = float(np.mean([int(x["nq"]) for x in t_runs])) if t_runs else 0.0
            nq_s = float(np.mean([int(x["nq"]) for x in s_runs])) if s_runs else 0.0
            any_row = (t_runs or s_runs)[0] if (t_runs or s_runs) else {}
            summary.append(dict(
                config_key=k,
                loss_main=any_row.get("loss_main",""),
                epochs=any_row.get("epochs",""),
                temperature=any_row.get("temperature",""),
                margin=any_row.get("margin",""),
                tune_hit3_mean=hit3_t_m,  tune_hit3_std=hit3_t_s,
                tune_mrr_mean=mrr_t_m,    tune_mrr_std=mrr_t_s,
                tune_nq_avg=nq_t,
                test_hit3_mean=hit3_s_m,  test_hit3_std=hit3_s_s,
                test_mrr_mean=mrr_s_m,    test_mrr_std=mrr_s_s,
                test_nq_avg=nq_s,
                outer_folds=len(s_runs)
            ))

        if summary:
            summary_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
            with open(summary_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                w.writeheader(); w.writerows(summary)
            print(f"[SAVE] 요약 결과: {summary_path}")
        else:
            print("[WARN] 요약 생성할 항목이 없습니다.")
    else:
        print("[WARN] 저장할 결과가 없습니다.")

if __name__ == "__main__":
    main()