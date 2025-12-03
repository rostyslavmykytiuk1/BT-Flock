# Next Plan: NLP-Based Row Scoring

## Analysis of Current Results

### Key Findings from Top vs Bottom Comparison:

**Top miners have:**
- ✅ **7.9% longer assistant responses** - More detailed explanations
- ✅ **6.6% longer rows overall** - More content per row
- ✅ **8.7% more exclamation marks** - More engaging/emphatic content
- ✅ **5.8% more punctuation diversity** - More varied writing style
- ✅ **5.7% more numbers** - More factual/data-driven content
- ✅ **4.7% more conversation turns** - More interactive

**Top miners have LESS:**
- ❌ **5.6% fewer questions** - Less interrogative, more declarative
- ❌ **Slightly lower vocab diversity** - But difference is tiny (0.3%)

### Problem with Current Metrics:
- Differences are **small** (5-8%) - not very predictive
- Simple metrics don't capture **semantic value** to the model
- Need to understand **what the model actually learns** from each row

---

## New Approach: NLP-Based Information Value

### Core Idea:
Instead of counting words/characters, measure **how much information** each row provides to the model based on:
1. **Semantic novelty** - How different is this from what model already knows?
2. **Information density** - How much new knowledge per token?
3. **Learning signal strength** - How clear is the teaching signal?
4. **Complementarity** - How well does this complement other rows?

---

## Proposed Solution: Embedding-Based Scoring

### Method 1: Semantic Information Content ⭐ (Best)

**Concept**: Use sentence embeddings to measure:
- **Semantic distance from training distribution** - Rows far from common patterns are more valuable
- **Information entropy** - Rows with high semantic entropy = more diverse knowledge
- **Embedding diversity** - How spread out are row embeddings in semantic space?

**Implementation**:
1. Use sentence-transformers (CPU mode) to embed all rows
2. Calculate semantic diversity within each miner's dataset
3. Measure distance from "common patterns" (centroid of all rows)
4. Score = base_weight * semantic_diversity * distance_from_common

**Why this works**:
- Models learn from **semantic patterns**, not word counts
- Diverse semantics = more learning opportunities
- Novel semantics = new knowledge for model

---

### Method 2: Information Gain Estimation

**Concept**: Estimate how much the model would learn from each row

**Metrics**:
- **Surprisal** - How unexpected is this content? (higher = more to learn)
- **KL divergence** - How different from model's current distribution?
- **Mutual information** - How much information does this add?

**Implementation**:
- Use language model (small, CPU-friendly) to calculate:
  - Perplexity of row (how surprising)
  - Cross-entropy (information content)
  - Compare to baseline distribution

---

### Method 3: Topic Modeling + Coverage

**Concept**: Ensure rows cover diverse topics/knowledge areas

**Implementation**:
1. Use LDA or BERTopic to identify topics in all rows
2. Score rows by:
   - Topic diversity within dataset
   - Coverage of important topics
   - Novelty of topics (rare topics = more valuable)

---

### Method 4: Training Signal Quality

**Concept**: Measure how "teachable" each row is

**Metrics**:
- **Clarity** - Is the teaching signal clear? (user question → clear answer)
- **Completeness** - Does answer fully address question?
- **Correctness proxy** - Does format/structure suggest quality?
- **Difficulty** - Is this at the right difficulty level for model?

---

## Recommended Implementation Plan

### Phase 1: Semantic Embedding Approach (Start Here) ⭐

**Why**: 
- Captures what models actually "see"
- CPU-friendly with sentence-transformers
- Directly measures semantic diversity (what we need)

**Steps**:
1. Install `sentence-transformers` (CPU mode)
2. Embed all rows using a good model (e.g., `all-MiniLM-L6-v2` - small, fast)
3. For each miner's dataset:
   - Calculate average pairwise cosine distance (semantic diversity)
   - Calculate distance from global centroid (novelty)
   - Combine: `semantic_score = diversity * novelty`
4. Score rows: `final_score = base_weight * semantic_score`

**Expected improvement**: Should capture semantic interactions better than word-level metrics

---

### Phase 2: Information Content (If Phase 1 works)

**Why**: 
- Directly estimates learning value
- More theoretically grounded

**Steps**:
1. Use small language model (e.g., GPT-2) to calculate:
   - Perplexity of each row
   - Cross-entropy (information content)
2. Score by information content
3. Combine with base weights

---

### Phase 3: Ensemble Approach

**Why**: 
- Combines multiple signals
- More robust

**Steps**:
1. Combine:
   - Base weight (emission-based)
   - Semantic diversity score
   - Information content score
   - Pattern-based features (from current analysis)
2. Weight each component
3. Optimize weights based on correlation

---

## Quick Start: Semantic Embedding Approach

### Step 1: Add dependency
```bash
# Add to pyproject.toml or install directly
pip install sentence-transformers
```

### Step 2: Create semantic scoring script
- Embed all rows
- Calculate semantic diversity per miner
- Score rows based on semantic value
- Compare with emission rank

### Step 3: Test and iterate
- Measure correlation improvement
- Tune parameters
- Compare with current approach

---

## Why This Should Work Better

### Current Problem:
- Word-level metrics don't capture **meaning**
- Model learns from **semantic patterns**, not word counts
- Row interactions are **semantic**, not lexical

### Semantic Approach:
- ✅ Captures what model actually processes
- ✅ Measures semantic diversity (real diversity)
- ✅ Can identify novel/valuable content
- ✅ Accounts for semantic interactions

---

## Expected Outcomes

**If semantic approach works**:
- Higher correlation with emission (0.60+ → 0.70+)
- Better understanding of what makes rows valuable
- More accurate row selection

**If it doesn't work**:
- Try information content approach
- Or combine multiple signals (ensemble)
- Or accept that simple metrics are the limit without actual training

---

## Next Steps

1. **Implement semantic embedding approach** (Phase 1)
2. **Test on all miners** - measure correlation
3. **If good**: Use for row selection
4. **If not**: Try Phase 2 (information content)
5. **Iterate** based on results

