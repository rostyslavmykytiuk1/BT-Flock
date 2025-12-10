# make_500_candidate_datasets.py
import json, math, random
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

# --- Config ---
DATA_PATH = Path("backtest/eval_data.jsonl")         # training pool
EVAL_PATH = Path("backtest/eval_data.jsonl")    # eval data for coverage scoring
OUT_DIR = Path("data/candidates")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # light; CPU-friendly

BATCH_SIZE = 4          # conservative for CPU
MAX_LEN = 512           # trim for CPU speed/memory
N_CLUSTERS = 60         # clusters for coverage
N_DATASETS = 100        # how many final subsets to emit
N_CANDIDATES = 150      # generate this many then keep best 100
SET_SIZE = 250          # size per subset
TEMP = 0.7              # lower -> more weight on hard examples
COVERAGE_WEIGHT = 0.2   # weight for eval coverage vs hardness
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

def load_data(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def format_dialog(sample, tokenizer) -> str:
    system = sample.get("system")
    conv = sample["conversations"]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(conv)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

def compute_nll(data, tokenizer, model):
    scores = []
    model.eval()
    with torch.no_grad():
        for batch_idx in tqdm(range(0, len(data), BATCH_SIZE), desc="NLL"):
            batch = data[batch_idx:batch_idx+BATCH_SIZE]
            texts = [format_dialog(s, tokenizer) for s in batch]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            ).to(model.device)

            labels = enc.input_ids.clone()
            outputs = model(**enc, labels=labels)
            logits = outputs.logits
            log_probs = -torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                reduction="none",
            ).view(labels.size(0), -1)

            # mask pads
            mask = (labels[:, 1:] != tokenizer.pad_token_id).float()
            log_probs = log_probs * mask
            seq_lens = mask.sum(dim=1).clamp(min=1)
            nll = -log_probs.sum(dim=1) / seq_lens  # positive NLL
            scores.extend(nll.cpu().tolist())
    return scores

def embed_texts(data):
    # embed only user turns concatenated (and system if present)
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(EMB_MODEL, device="cpu")
    texts = []
    for s in data:
        sys = s.get("system") or ""
        user_turns = " ".join([c["content"] for c in s["conversations"] if c["role"] == "user"])
        texts.append((sys + " " + user_turns).strip())
    embs = emb_model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    return normalize(embs)

def cluster_indices(embs):
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, batch_size=512)
    labels = kmeans.fit_predict(embs)
    clusters = {i: [] for i in range(N_CLUSTERS)}
    for idx, lab in enumerate(labels):
        clusters[lab].append(idx)
    return clusters

def build_datasets(nlls, clusters, n_sets, set_size, temp):
    datasets = []
    # precompute softmax weights per cluster (NLL^temp)
    weights = {}
    for cid, idxs in clusters.items():
        if not idxs:
            continue
        vals = torch.tensor([nlls[i] for i in idxs], dtype=torch.float32)
        # convert to probabilities; higher NLL => higher weight
        probs = torch.softmax(vals / temp, dim=0).tolist()
        weights[cid] = (idxs, probs)

    # quota per cluster proportional to size (at least 1)
    total = sum(len(v) for v in clusters.values())
    base_quota = {
        cid: max(1, round(len(idxs) / total * set_size))
        for cid, idxs in clusters.items()
    }
    # adjust to exact set_size
    while sum(base_quota.values()) > set_size:
        # reduce from largest quota
        cid = max(base_quota, key=base_quota.get)
        base_quota[cid] -= 1
    while sum(base_quota.values()) < set_size:
        cid = max(base_quota, key=lambda c: len(clusters[c]))
        base_quota[cid] += 1

    for _ in range(n_sets):
        chosen = []
        for cid, quota in base_quota.items():
            if cid not in weights:
                continue
            idxs, probs = weights[cid]
            # Cap quota to available items in cluster
            actual_quota = min(quota, len(idxs))
            picked = set()
            attempts = 0
            max_attempts = actual_quota * 10  # prevent infinite loop
            while len(picked) < actual_quota and attempts < max_attempts:
                sel = random.choices(idxs, weights=probs, k=1)[0]
                if sel not in picked:
                    picked.add(sel)
                attempts += 1
            chosen.extend(picked)
        
        # Ensure we have exactly set_size items by topping up globally
        all_probs = torch.softmax(torch.tensor(nlls) / temp, dim=0).tolist()
        chosen_set = set(chosen)
        while len(chosen) < set_size:
            sel = random.choices(range(len(nlls)), weights=all_probs, k=1)[0]
            if sel not in chosen_set:
                chosen.append(sel)
                chosen_set.add(sel)
        
        # Trim to exactly set_size if somehow we exceeded (shouldn't happen)
        datasets.append(chosen[:set_size])
    return datasets


def score_subset(idxs, nlls, embs, eval_embs):
    # Hardness: mean NLL of selected items
    hardness = sum(nlls[i] for i in idxs) / len(idxs)
    # Coverage: average max similarity from eval samples to this subset
    sub_embs = embs[idxs]  # (subset, dim)
    sims = eval_embs @ sub_embs.T  # (eval, subset)
    coverage = sims.max(axis=1).mean()
    return hardness + COVERAGE_WEIGHT * coverage

def main():
    data = load_data(DATA_PATH)
    eval_data = load_data(EVAL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    ).to("cpu")

    # 1) NLL scores
    nlls = compute_nll(data, tokenizer, model)

    # 2) Embeddings + clusters
    embs = embed_texts(data)
    eval_embs = embed_texts(eval_data)
    clusters = cluster_indices(embs)

    # 3) Build candidate datasets
    candidates = build_datasets(nlls, clusters, n_sets=N_CANDIDATES, set_size=SET_SIZE, temp=TEMP)

    # 4) Score candidates against eval coverage + hardness, pick best 100
    import numpy as np
    embs_np = np.array(embs)
    eval_np = np.array(eval_embs)
    scored = []
    for idxs in candidates:
        scored.append((idxs, score_subset(idxs, nlls, embs_np, eval_np)))
    scored.sort(key=lambda x: x[1], reverse=True)
    subsets = [s[0] for s in scored[:N_DATASETS]]

    # Verify each dataset has exactly SET_SIZE items
    for i, subset in enumerate(subsets):
        if len(subset) != SET_SIZE:
            print(f"Warning: dataset {i} has {len(subset)} items, expected {SET_SIZE}")
            # Ensure exactly SET_SIZE by padding or trimming
            if len(subset) < SET_SIZE:
                # Add more items if needed
                all_probs = torch.softmax(torch.tensor(nlls) / TEMP, dim=0).tolist()
                subset_set = set(subset)
                while len(subset) < SET_SIZE:
                    sel = random.choices(range(len(nlls)), weights=all_probs, k=1)[0]
                    if sel not in subset_set:
                        subset.append(sel)
                        subset_set.add(sel)
            subsets[i] = subset[:SET_SIZE]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, subset in enumerate(subsets):
        assert len(subset) == SET_SIZE, f"Dataset {i} has {len(subset)} items, expected {SET_SIZE}"
        out_path = OUT_DIR / f"dataset_{i:03d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for idx in subset:
                f.write(json.dumps(data[idx], ensure_ascii=False) + "\n")
    print(f"Wrote {len(subsets)} datasets to {OUT_DIR}, each with exactly {SET_SIZE} items")

if __name__ == "__main__":
    main()
