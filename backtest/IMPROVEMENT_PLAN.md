# Improvement Plan for Row Selection (No GPU Required)

## Problem Summary
Current approach sums individual row weights, but model training evaluates the **entire dataset together**. Rows interact in non-additive ways (redundancy, complementarity, negative interactions).

## Proposed Solutions (CPU-Only)

### 1. **Diversity-Based Selection** ⭐ (Highest Priority)
**Goal**: Ensure selected rows are diverse to avoid redundancy

**Implementation**:
- Use lightweight text embeddings (sentence-transformers CPU mode, or TF-IDF)
- Calculate similarity matrix between rows
- Select rows that are dissimilar to already-selected rows
- Combine with weight: `score = weight * (1 - max_similarity_to_selected)`

**Benefits**:
- Reduces redundancy (similar rows don't add much value)
- Encourages coverage of different topics/styles
- CPU-friendly (can use small models or TF-IDF)

**Tools Needed**:
- `sentence-transformers` (CPU mode) OR
- `scikit-learn` for TF-IDF + cosine similarity

---

### 2. **Pattern Analysis from Top Miners** ⭐
**Goal**: Learn patterns from successful miners' datasets

**Implementation**:
- Analyze top N miners' datasets for patterns:
  - Average row length
  - Vocabulary diversity
  - Topic distribution (simple keyword-based)
  - Conversation structure patterns
- Score rows based on how well they match successful patterns
- Adjust weights: `adjusted_weight = base_weight * pattern_score`

**Benefits**:
- Captures implicit knowledge from successful miners
- No model training needed
- Can identify what makes datasets effective

**Analysis to do**:
- Compare top 10% vs bottom 10% miners' datasets
- Find distinguishing characteristics
- Use as features for row scoring

---

### 3. **Clustering + Representative Selection**
**Goal**: Select diverse rows by choosing from different clusters

**Implementation**:
- Cluster all rows using K-means or hierarchical clustering (on embeddings)
- Within each cluster, select highest-weight rows
- Ensures coverage across different "types" of rows

**Benefits**:
- Guarantees diversity
- Simple to implement
- Can control diversity vs weight tradeoff

---

### 4. **Incremental Greedy Selection with Diversity**
**Goal**: Select rows one-by-one, considering complementarity

**Implementation**:
- Start with highest-weight row
- For each next selection:
  - Score candidates: `score = weight - λ * similarity_to_selected`
  - Select highest-scoring row
  - Update similarity scores
- λ controls diversity vs weight tradeoff

**Benefits**:
- Considers interactions incrementally
- Can balance weight and diversity
- Simple greedy algorithm

---

### 5. **Statistical Feature Scoring**
**Goal**: Use simple features to predict row quality

**Features to extract** (CPU-friendly):
- Text length (tokens/characters)
- Vocabulary diversity (unique words / total words)
- Question complexity (question marks, question words)
- Response length ratio
- Keyword presence (domain-specific)
- Conversation structure (number of turns, system prompts)

**Implementation**:
- Extract features for all rows
- Train simple model (linear regression, random forest) on:
  - Features as X
  - Row weight as y (proxy for quality)
- Use model to score rows: `predicted_quality = model.predict(features)`
- Combine: `final_score = base_weight * predicted_quality`

**Benefits**:
- Captures patterns without deep learning
- Fast on CPU
- Interpretable

---

### 6. **Ensemble of Strategies**
**Goal**: Combine multiple heuristics

**Implementation**:
- Run multiple selection strategies:
  1. Weight-based (current)
  2. Diversity-based
  3. Pattern-based
  4. Feature-based
- Score each row by: `ensemble_score = weighted_average(strategy_scores)`
- Select top rows by ensemble score

**Benefits**:
- More robust than single strategy
- Can weight strategies based on validation

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days)
1. **Diversity-based selection** - Easy to implement, likely high impact
2. **Pattern analysis** - Analyze top miners' datasets for insights

### Phase 2: Medium Effort (3-5 days)
3. **Incremental greedy with diversity** - Improve selection algorithm
4. **Statistical feature scoring** - Add feature-based signals

### Phase 3: Advanced (1 week+)
5. **Clustering approach** - More sophisticated diversity
6. **Ensemble method** - Combine all strategies

---

## Quick Start: Diversity-Based Selection

### Step 1: Install dependencies
```bash
# Option A: Lightweight (TF-IDF)
# Already have scikit-learn via dependencies

# Option B: Better embeddings (CPU mode)
pip install sentence-transformers
```

### Step 2: Calculate row embeddings/similarities
- Extract text from each row (user + assistant content)
- Compute embeddings or TF-IDF vectors
- Calculate pairwise similarity matrix

### Step 3: Modify selection algorithm
- When selecting row i:
  - Calculate max similarity to already-selected rows
  - Adjust score: `adjusted_score = weight * (1 - max_similarity)`
  - Select highest adjusted_score

### Step 4: Tune diversity weight
- Parameter λ: `score = weight - λ * max_similarity`
- Test different λ values
- Find optimal balance

---

## Validation Strategy

Since we can't train models, validate by:
1. **Correlation with emission**: Does new approach improve correlation?
2. **Diversity metrics**: Measure actual diversity of selected rows
3. **Pattern matching**: Do selected rows match successful miners' patterns?
4. **A/B testing**: Compare old vs new selection on same data

---

## Expected Improvements

- **Current correlation**: ~0.70 Spearman
- **Target**: 0.75-0.80+ with diversity considerations
- **Key metric**: How well does data_weight rank match emission rank?

---

## Next Steps

1. Start with **Diversity-Based Selection** (easiest, high impact)
2. Analyze top miners' datasets for patterns
3. Implement incremental improvements
4. Measure correlation improvements
5. Iterate based on results

