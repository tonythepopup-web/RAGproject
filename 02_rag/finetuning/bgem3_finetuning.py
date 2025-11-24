# -*- coding: utf-8 -*-
"""
BGE-m3-ko 튜닝 + 내보내기(export)
- loss 종류: mp_infonce / cosine+triplet
- 입력:
  * qnalist.txt (질문 100개, 10개씩 1문서 그룹 가정)
  * triplets_group_bgem3.jsonl (query, positives[], negatives[])
  * chunks.json (사전 청킹된 조문 청크; 재청킹 없음)
- 출력(필수):
  * 튜닝된 SentenceTransformer 모델 디렉토리
- 출력(옵션 --export-faiss):
  * corpus_embeddings.npy
  * faiss.index
"""

import os, re, json, math, random, csv, hashlib, argparse
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

# ----------------------------
# 기본 경로/하이퍼
# ----------------------------
MODEL_NAME = "dragonkue/BGE-m3-ko"
PROMPT_Q   = "query: "
PROMPT_P   = "passage: "

# 학습 예제 상한
MAX_POS_CHUNKS_TOTAL    = 12
MAX_NEG_CHUNKS_TOTAL    = 24
MAX_TRIPLET_NEG_PER_POS = 6

# encode 배치
EVAL_BATCH = 256

# ----------------------------
# 유틸
# ----------------------------
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
    doc_ids = [i // 10 for i in range(len(queries))]  # 10문항 = 1문서 그룹 가정
    return queries, q2id, doc_ids

def read_labels_jsonl(path: str):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            rec["query"]     = norm_text(rec.get("query", ""))
            rec["positives"] = [norm_text(x) for x in rec.get("positives", []) if norm_text(x)]
            rec["negatives"] = [norm_text(x) for x in rec.get("negatives", []) if norm_text(x)]
            recs.append(rec)
    return recs

def match_labels_to_qids(queries, q2id, label_recs):
    qid2 = defaultdict(lambda: {"positives":[], "negatives":[]})
    for rec in label_recs:
        qid = q2id.get(rec.get("query",""))
        if qid is None: continue
        qid2[qid]["positives"].extend(rec.get("positives",[]))
        qid2[qid]["negatives"].extend(rec.get("negatives",[]))
    return dict(qid2)

def load_prechunked_corpus(chunks_json_path, prompt_p=PROMPT_P):
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    corpus, txt2pids, pid2doc = {}, {}, {}
    for i, ch in enumerate(chunks):
        body = (ch.get("text") or ch.get("enriched_text") or "").strip()
        if not body: continue
        pid = f"P{i:06d}"
        corpus[pid] = prompt_p + body

        # doc_id 안정화(충돌 방지)
        src = (ch.get("source") or "").strip()
        key_parts = [src, str(ch.get("law_id","")), str(ch.get("article_id","")), body[:64]]
        doc_id = hashlib.md5(("||".join(key_parts)).encode("utf-8")).hexdigest()
        pid2doc[pid] = doc_id

        txt2pids.setdefault(body, []).append(pid)
    return corpus, txt2pids, pid2doc

def _tokset(s: str):
    return set(re.findall(r"[가-힣A-Za-z0-9]+", s or ""))

def match_pids_for_label_text(label_text: str, txt2pids: dict, jaccard_thr=0.6, top_k=3):
    t = (label_text or "").strip()
    if not t: return []
    if t in txt2pids: return txt2pids[t]

    # 포함 일치
    cands = []
    for body, pids in txt2pids.items():
        if t in body or body in t: cands.append((1.0, pids))
    if cands:
        out = []
        for _, ps in cands[:top_k]: out.extend(ps)
        return out

    # Jaccard top-k
    tset = _tokset(t)
    scored = []
    for body, pids in txt2pids.items():
        bset = _tokset(body)
        sim = len(tset & bset) / (len(tset | bset) or 1)
        if sim >= jaccard_thr: scored.append((sim, pids))
    scored.sort(reverse=True, key=lambda x: x[0])
    out = []
    for _, ps in scored[:top_k]: out.extend(ps)
    return out

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
        if MAX_POS_CHUNKS_TOTAL and len(pos_chunks) > MAX_POS_CHUNKS_TOTAL:
            pos_chunks = pos_chunks[:MAX_POS_CHUNKS_TOTAL]
        if MAX_NEG_CHUNKS_TOTAL and len(neg_chunks) > MAX_NEG_CHUNKS_TOTAL:
            neg_chunks = neg_chunks[:MAX_NEG_CHUNKS_TOTAL]

        for p in pos_chunks: pairs_cos.append(InputExample(texts=[qtext, p], label=1.0))
        for n in neg_chunks: pairs_cos.append(InputExample(texts=[qtext, n], label=0.0))

        if pos_chunks and neg_chunks:
            for p in pos_chunks:
                k = min(len(neg_chunks), MAX_TRIPLET_NEG_PER_POS)
                for n in random.sample(neg_chunks, k) if k>0 else []:
                    triplets.append(InputExample(texts=[qtext, p, n]))
    return pairs_cos, triplets

def build_qid2chunks_for_mp(qids_train, queries, qid2posneg, txt2pids, corpus, prompt_q=PROMPT_Q):
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

# ----------------------------
# mp_infonce 손실
# ----------------------------
class MultiPositiveInfoNCELoss(nn.Module):
    def __init__(self, model, temperature: float = 0.05):
        super().__init__(); self.model = model; self.tau = float(temperature)
    def forward(self, sentence_features, labels):
        out_q = self.model(sentence_features[0])['sentence_embedding']  # (1,H)
        out_p = self.model(sentence_features[1])['sentence_embedding']  # (B,H)
        q_emb = F.normalize(out_q, dim=-1); p_emb = F.normalize(out_p, dim=-1)
        logits = (q_emb @ p_emb.t()) / self.tau
        pos_mask = labels.bool()                     # (1,B)
        if not pos_mask.any(): return logits.new_tensor(0.0)
        log_den = torch.logsumexp(logits, dim=1)
        logits_pos = torch.where(pos_mask, logits, torch.full_like(logits, float('-inf')))
        log_num = torch.logsumexp(logits_pos, dim=1)
        return -(log_num - log_den).mean()

class MPDataset(Dataset):
    """qid → (query, pos_chunks[], neg_chunks[])"""
    def __init__(self, qids, queries, qid2chunks, prompt_q=PROMPT_Q):
        self.items = []
        for qid in qids:
            chunks = qid2chunks.get(qid, {"pos_chunks":[], "neg_chunks":[]})
            pos = [t for t in chunks["pos_chunks"] if t]
            neg = [t for t in chunks["neg_chunks"] if t]
            if not pos: continue
            if MAX_POS_CHUNKS_TOTAL and len(pos) > MAX_POS_CHUNKS_TOTAL: pos = pos[:MAX_POS_CHUNKS_TOTAL]
            if MAX_NEG_CHUNKS_TOTAL and len(neg) > MAX_NEG_CHUNKS_TOTAL: neg = neg[:MAX_NEG_CHUNKS_TOTAL]
            q_text = prompt_q + queries[qid]
            self.items.append((qid, q_text, pos, neg))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def mp_collate_fn(batch, tokenizer, max_seq_len):
    assert len(batch) == 1, "Use batch_size=1"
    _, q_text, pos_list, neg_list = batch[0]
    passages = pos_list + neg_list
    # 큰 배치 보호(동적 제한)
    MAX_PASSAGES = 64
    if len(passages) > MAX_PASSAGES:
        keep_pos = min(len(pos_list), MAX_PASSAGES//2)
        keep_neg = MAX_PASSAGES - keep_pos
        pos_idx = list(range(len(pos_list)))[:keep_pos]
        neg_idx = random.sample(range(len(pos_list), len(passages)), min(keep_neg, len(passages)-len(pos_list)))
        sel = pos_idx + neg_idx
        passages = [passages[i] for i in sel]
        pos_len = keep_pos
    else:
        pos_len = len(pos_list)

    pos_mask = torch.zeros((1, len(passages)), dtype=torch.bool)
    pos_mask[0, :pos_len] = True

    q_feat = tokenizer(q_text, padding='max_length', truncation=True, max_length=max_seq_len, return_tensors='pt')
    p_feat = tokenizer(passages, padding='max_length', truncation=True, max_length=max_seq_len, return_tensors='pt')
    features_q = {'input_ids': q_feat['input_ids'], 'attention_mask': q_feat['attention_mask']}
    features_p = {'input_ids': p_feat['input_ids'], 'attention_mask': p_feat['attention_mask']}
    return [features_q, features_p], pos_mask

def train_mp(model, tokenizer, qids_train, queries, qid2chunks, epochs=1, lr=2e-5, warmup_ratio=0.1, weight_decay=0.01, temperature=0.05, max_seq_len=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    try:
        torch.set_float32_matmul_precision("high")
        model._first_module().auto_model.gradient_checkpointing_enable()
    except: pass

    ds = MPDataset(qids_train, queries, qid2chunks, PROMPT_Q)
    ld = DataLoader(ds, shuffle=True, batch_size=1,
                    collate_fn=lambda b: mp_collate_fn(b, tokenizer, max_seq_len),
                    num_workers=0, pin_memory=False)
    criterion = MultiPositiveInfoNCELoss(model, temperature=float(temperature))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    steps = len(ld) * int(epochs)
    warmup_steps = int(steps * float(warmup_ratio))
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    from torch.amp import GradScaler, autocast
    scaler = GradScaler("cuda") if torch.cuda.is_available() else None

    model.train()
    for _ in range(int(epochs)):
        for features, pos_mask in ld:
            features = [{k: v.to(device, non_blocking=True) for k, v in f.items()} for f in features]
            pos_mask = pos_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=torch.cuda.is_available()):
                loss = criterion(features, pos_mask)
            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            scheduler.step()
    return model

def train_cosine_triplet(model, examples_cos, examples_triplet, epochs=1, lr=2e-5, warmup_ratio=0.1, weight_decay=0.0, margin=0.3, batch=16):
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        try:
            torch.set_float32_matmul_precision("high")
            model._first_module().auto_model.gradient_checkpointing_enable()
        except: pass

    train_objectives = []
    loader_kwargs = dict(num_workers=0, pin_memory=False)
    if examples_cos:
        dl_cos = DataLoader(examples_cos, shuffle=True, batch_size=int(batch), **loader_kwargs)
        train_objectives.append((dl_cos, losses.CosineSimilarityLoss(model)))
    if examples_triplet:
        dl_tri = DataLoader(examples_triplet, shuffle=True, batch_size=max(8, int(batch)//2), **loader_kwargs)
        try:
            tloss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=float(margin))
        except TypeError:
            tloss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, margin=float(margin))
        train_objectives.append((dl_tri, tloss))
    if not train_objectives:
        return model

    steps_per_epoch = sum(len(ld) for ld,_ in train_objectives)
    warmup_steps = math.ceil(steps_per_epoch * int(epochs) * float(warmup_ratio))
    model.fit(
        train_objectives=train_objectives,
        epochs=int(epochs),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(lr)},
        weight_decay=float(weight_decay),
        output_path=None,
        use_amp=True
    )
    return model

# ----------------------------
# Export: 모델 + (옵션) 임베딩/FAISS
# ----------------------------
def export_model(model: SentenceTransformer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model.save(out_dir)
    print(f"[SAVE] model -> {out_dir}")

def export_corpus_and_faiss(model: SentenceTransformer, corpus: Dict[str,str], out_dir: str, max_len=512, batch_size=256):
    os.makedirs(out_dir, exist_ok=True)
    texts = list(corpus.values())

    with torch.no_grad():
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            truncation=True, max_length=max_len,
            device=("cuda" if torch.cuda.is_available() else "cpu")
        ).cpu().float().numpy()  # [N, D]

    np.save(os.path.join(out_dir, "corpus_embeddings.npy"), emb)
    print(f"[SAVE] corpus_embeddings.npy ({emb.shape})")

    try:
        import faiss
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine(정규화 가정) → 내적
        index.add(emb)
        faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
        print(f"[SAVE] faiss.index (IndexFlatIP, {emb.shape[0]} vectors)")
    except Exception as e:
        print("[WARN] FAISS export skipped:", e)

# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qna", type=str, required=True, help="qnalist.txt")
    ap.add_argument("--labels", type=str, required=True, help="triplets_group_bgem3.jsonl")
    ap.add_argument("--chunks", type=str, required=True, help="prechunked chunks.json")
    ap.add_argument("--loss", type=str, choices=["mp_infonce","cosine+triplet"], required=True)
    ap.add_argument("--out", type=str, required=True, help="모델 저장 디렉토리")
    ap.add_argument("--export-faiss", action="store_true", help="코퍼스 임베딩/FAISS 인덱스도 저장")
    # 공통/기본 하이퍼
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_len", type=int, default=512)
    # mp_infonce
    ap.add_argument("--temperature", type=float, default=0.05)
    # cosine+triplet
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=16)

    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    # 데이터 로드
    queries, q2id, _ = read_qna(args.qna)
    label_recs       = read_labels_jsonl(args.labels)
    qid2posneg       = match_labels_to_qids(queries, q2id, label_recs)
    corpus, txt2pids, pid2doc = load_prechunked_corpus(args.chunks)

    # 토크나이저/모델 & safe_len
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_max = getattr(tokenizer, "model_max_length", 1024)
    safe_len  = min(int(args.max_len), model_max - 8)

    # 모든 질의 중 라벨 존재하는 qid만 학습에 사용
    qids_train = [qid for qid in range(len(queries)) if qid in qid2posneg]

    # 베이스 모델
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = safe_len

    if args.loss == "mp_infonce":
        qid2chunks_mp = build_qid2chunks_for_mp(qids_train, queries, qid2posneg, txt2pids, corpus)
        model = train_mp(model, tokenizer, qids_train, queries, qid2chunks_mp,
                         epochs=args.epochs, lr=args.lr, warmup_ratio=args.warmup_ratio,
                         weight_decay=args.weight_decay, temperature=args.temperature,
                         max_seq_len=safe_len)
    else:
        ex_cos, ex_tri = build_examples_from_prechunks(qids_train, queries, qid2posneg, txt2pids, corpus)
        model = train_cosine_triplet(model, ex_cos, ex_tri,
                                     epochs=args.epochs, lr=args.lr, warmup_ratio=args.warmup_ratio,
                                     weight_decay=args.weight_decay, margin=args.margin, batch=args.batch)

    # 내보내기: 모델
    export_model(model, args.out)

    # (옵션) 코퍼스 임베딩 + FAISS
    if args.export_faiss:
        export_corpus_and_faiss(model, corpus, args.out, max_len=safe_len, batch_size=EVAL_BATCH)

if __name__ == "__main__":
    main()
