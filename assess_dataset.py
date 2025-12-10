"""
Dataset Assessment Script

Assesses the selected training dataset and estimates evaluation loss
without actually training the model.

This script:
1. Loads training data (250 examples)
2. Computes baseline loss on eval data using pretrained model
3. Analyzes training data characteristics
4. Estimates potential improvement based on training data quality
5. Provides coverage metrics (how well training data represents eval data)
"""

import json
import os
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file where each line is a JSON object."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_conversation(data: Dict, tokenizer) -> str:
    """Format conversation data into the chat template format."""
    messages = []
    
    # Add system message if present
    system = data.get("system")
    if system and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    
    # Add conversations
    for conv in data.get("conversations", []):
        role = conv.get("role", "")
        content = conv.get("content", "").strip()
        if role and content:
            messages.append({"role": role, "content": content})
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return text


def compute_loss_single(
    model,
    tokenizer,
    text: str,
    max_length: int = 2048,
    device: str = "cpu"
) -> float:
    """Compute loss for a single text."""
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss.item()
    
    return loss


def compute_loss_batch(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 4,
    max_length: int = 2048,
    device: str = "cpu"
) -> List[float]:
    """Compute loss for texts efficiently."""
    losses = []
    
    for text in tqdm(texts, desc="Computing losses", leave=False):
        if not text.strip():
            losses.append(float('inf'))  # Invalid text
            continue
            
        loss = compute_loss_single(
            model,
            tokenizer,
            text,
            max_length=max_length,
            device=device
        )
        losses.append(loss)
    
    return losses


def get_example_features(data: Dict) -> Dict:
    """Extract features for analysis."""
    has_system = data.get("system") is not None and data.get("system", "").strip() != ""
    num_turns = len(data.get("conversations", []))
    total_length = sum(len(conv.get("content", "")) for conv in data.get("conversations", []))
    
    # Extract user messages for topic analysis
    user_messages = [conv.get("content", "") for conv in data.get("conversations", []) 
                     if conv.get("role") == "user"]
    
    # Length buckets
    if total_length < 200:
        length_bucket = "short"
    elif total_length < 500:
        length_bucket = "medium"
    else:
        length_bucket = "long"
    
    return {
        "has_system": has_system,
        "num_turns": num_turns,
        "total_length": total_length,
        "length_bucket": length_bucket,
        "user_messages": user_messages
    }


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract top keywords from text (simple word frequency)."""
    # Simple word extraction (can be improved with proper tokenization)
    words = text.lower().split()
    # Filter out very short words and common stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                  "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
                  "has", "had", "do", "does", "did", "will", "would", "could", "should"}
    words = [w for w in words if len(w) > 3 and w not in stop_words]
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(top_n)]


def compute_coverage_metrics(
    train_data: List[Dict],
    eval_data: List[Dict],
    train_features: List[Dict],
    eval_features: List[Dict]
) -> Dict:
    """Compute how well training data covers eval data."""
    
    # Feature distribution comparison
    train_system_ratio = sum(1 for f in train_features if f["has_system"]) / len(train_features)
    eval_system_ratio = sum(1 for f in eval_features if f["has_system"]) / len(eval_features)
    
    train_length_dist = Counter(f["length_bucket"] for f in train_features)
    eval_length_dist = Counter(f["length_bucket"] for f in eval_features)
    
    # Normalize distributions
    train_length_norm = {k: v / len(train_features) for k, v in train_length_dist.items()}
    eval_length_norm = {k: v / len(eval_features) for k, v in eval_length_dist.items()}
    
    # Compute KL divergence (simplified)
    length_kl = 0.0
    for bucket in ["short", "medium", "long"]:
        train_p = train_length_norm.get(bucket, 0.001)
        eval_p = eval_length_norm.get(bucket, 0.001)
        if eval_p > 0:
            length_kl += eval_p * np.log(eval_p / train_p)
    
    # Keyword overlap (simple approach)
    train_keywords = set()
    for features in train_features:
        for msg in features["user_messages"]:
            train_keywords.update(extract_keywords(msg, top_n=5))
    
    eval_keywords = set()
    for features in eval_features:
        for msg in features["user_messages"]:
            eval_keywords.update(extract_keywords(msg, top_n=5))
    
    keyword_overlap = len(train_keywords & eval_keywords) / len(eval_keywords) if eval_keywords else 0.0
    
    return {
        "system_ratio_train": train_system_ratio,
        "system_ratio_eval": eval_system_ratio,
        "system_ratio_diff": abs(train_system_ratio - eval_system_ratio),
        "length_distribution_train": train_length_norm,
        "length_distribution_eval": eval_length_norm,
        "length_kl_divergence": length_kl,
        "keyword_overlap": keyword_overlap,
        "train_keywords_count": len(train_keywords),
        "eval_keywords_count": len(eval_keywords),
        "common_keywords_count": len(train_keywords & eval_keywords)
    }


def estimate_improvement_potential(
    train_losses: List[float],
    eval_losses: List[float],
    train_perplexities: List[float],
    eval_perplexities: List[float]
) -> Dict:
    """Estimate potential improvement from training."""
    
    # Baseline metrics
    baseline_eval_loss = np.mean(eval_losses)
    baseline_eval_ppl = np.mean(eval_perplexities)
    
    # Training data difficulty
    train_avg_loss = np.mean(train_losses)
    train_avg_ppl = np.mean(train_perplexities)
    
    # If training data has higher loss, it's harder - more potential for improvement
    difficulty_ratio = train_avg_loss / baseline_eval_loss if baseline_eval_loss > 0 else 1.0
    
    # Estimate: if we reduce loss on hard examples, eval should improve
    # Simple heuristic: improvement proportional to difficulty ratio
    estimated_improvement_ratio = 0.1 * difficulty_ratio  # Conservative estimate
    estimated_final_loss = baseline_eval_loss * (1 - estimated_improvement_ratio)
    
    # Confidence based on training data quality
    train_loss_std = np.std(train_losses)
    eval_loss_std = np.std(eval_losses)
    
    # Lower variance in training data = more consistent = better
    consistency_score = 1.0 / (1.0 + train_loss_std) if train_loss_std > 0 else 1.0
    
    return {
        "baseline_eval_loss": baseline_eval_loss,
        "baseline_eval_perplexity": baseline_eval_ppl,
        "train_avg_loss": train_avg_loss,
        "train_avg_perplexity": train_avg_ppl,
        "difficulty_ratio": difficulty_ratio,
        "estimated_improvement_ratio": estimated_improvement_ratio,
        "estimated_final_loss": estimated_final_loss,
        "estimated_improvement": baseline_eval_loss - estimated_final_loss,
        "consistency_score": consistency_score,
        "train_loss_std": train_loss_std,
        "eval_loss_std": eval_loss_std
    }


def print_statistics(
    train_data: List[Dict],
    eval_data: List[Dict],
    train_losses: List[float],
    eval_losses: List[float],
    train_perplexities: List[float],
    eval_perplexities: List[float],
    train_features: List[Dict],
    eval_features: List[Dict],
    coverage: Dict,
    improvement: Dict
):
    """Print comprehensive statistics."""
    
    print("\n" + "="*80)
    print("DATASET ASSESSMENT REPORT")
    print("="*80)
    
    # Dataset sizes
    print(f"\n📊 Dataset Sizes:")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Evaluation examples: {len(eval_data)}")
    print(f"  Ratio: {len(train_data)/len(eval_data)*100:.2f}%")
    
    # Baseline performance (pretrained model on eval data)
    print(f"\n🎯 Baseline Performance (Pretrained Model on Eval Data):")
    print(f"  Average Loss: {improvement['baseline_eval_loss']:.4f}")
    print(f"  Average Perplexity: {improvement['baseline_eval_perplexity']:.2f}")
    print(f"  Loss Std Dev: {improvement['eval_loss_std']:.4f}")
    print(f"  Min Loss: {np.min(eval_losses):.4f}")
    print(f"  Max Loss: {np.max(eval_losses):.4f}")
    print(f"  Median Loss: {np.median(eval_losses):.4f}")
    
    # Training data characteristics
    print(f"\n📚 Training Data Characteristics:")
    print(f"  Average Loss: {improvement['train_avg_loss']:.4f}")
    print(f"  Average Perplexity: {improvement['train_avg_perplexity']:.2f}")
    print(f"  Loss Std Dev: {improvement['train_loss_std']:.4f}")
    print(f"  Difficulty Ratio (train/eval): {improvement['difficulty_ratio']:.2f}")
    print(f"  Consistency Score: {improvement['consistency_score']:.3f}")
    
    # Feature distribution
    print(f"\n📈 Feature Distribution:")
    print(f"  System Messages:")
    print(f"    Training: {coverage['system_ratio_train']*100:.1f}%")
    print(f"    Evaluation: {coverage['system_ratio_eval']*100:.1f}%")
    print(f"    Difference: {coverage['system_ratio_diff']*100:.1f}%")
    
    print(f"\n  Length Distribution:")
    print(f"    Training:")
    for bucket, ratio in sorted(coverage['length_distribution_train'].items()):
        print(f"      {bucket}: {ratio*100:.1f}%")
    print(f"    Evaluation:")
    for bucket, ratio in sorted(coverage['length_distribution_eval'].items()):
        print(f"      {bucket}: {ratio*100:.1f}%")
    
    # Coverage metrics
    print(f"\n🔍 Coverage Metrics:")
    print(f"  Keyword Overlap: {coverage['keyword_overlap']*100:.1f}%")
    print(f"  Training Keywords: {coverage['train_keywords_count']}")
    print(f"  Eval Keywords: {coverage['eval_keywords_count']}")
    print(f"  Common Keywords: {coverage['common_keywords_count']}")
    print(f"  Length KL Divergence: {coverage['length_kl_divergence']:.4f} (lower is better)")
    
    # Improvement estimates
    print(f"\n🚀 Estimated Improvement Potential:")
    print(f"  Estimated Final Loss: {improvement['estimated_final_loss']:.4f}")
    print(f"  Estimated Improvement: {improvement['estimated_improvement']:.4f}")
    print(f"  Improvement Ratio: {improvement['estimated_improvement_ratio']*100:.1f}%")
    print(f"\n  ⚠️  Note: These are rough estimates. Actual results depend on:")
    print(f"     - Training hyperparameters")
    print(f"     - Model architecture")
    print(f"     - Training duration")
    print(f"     - Data quality and diversity")
    
    # Quality assessment
    print(f"\n✅ Quality Assessment:")
    
    # Coverage score
    coverage_score = (
        (1.0 - coverage['system_ratio_diff']) * 0.3 +
        coverage['keyword_overlap'] * 0.4 +
        (1.0 / (1.0 + coverage['length_kl_divergence'])) * 0.3
    )
    
    # Difficulty score (higher is better - means training on harder examples)
    difficulty_score = min(improvement['difficulty_ratio'], 2.0) / 2.0
    
    # Consistency score
    consistency_score = improvement['consistency_score']
    
    overall_score = (coverage_score * 0.4 + difficulty_score * 0.3 + consistency_score * 0.3)
    
    print(f"  Coverage Score: {coverage_score:.3f} (higher = better representation)")
    print(f"  Difficulty Score: {difficulty_score:.3f} (higher = training on harder examples)")
    print(f"  Consistency Score: {consistency_score:.3f} (higher = more consistent)")
    print(f"  Overall Quality Score: {overall_score:.3f}/1.0")
    
    if overall_score > 0.7:
        print(f"  ✅ Excellent training dataset quality!")
    elif overall_score > 0.5:
        print(f"  ⚠️  Good training dataset, but could be improved")
    else:
        print(f"  ❌ Training dataset may need better selection")
    
    print("\n" + "="*80)


def save_report(
    output_file: str,
    train_data: List[Dict],
    eval_data: List[Dict],
    train_losses: List[float],
    eval_losses: List[float],
    train_perplexities: List[float],
    eval_perplexities: List[float],
    train_features: List[Dict],
    eval_features: List[Dict],
    coverage: Dict,
    improvement: Dict
):
    """Save detailed report to JSON file."""
    report = {
        "dataset_sizes": {
            "train": len(train_data),
            "eval": len(eval_data),
            "ratio": len(train_data) / len(eval_data) if len(eval_data) > 0 else 0
        },
        "baseline_performance": {
            "eval_loss_mean": float(np.mean(eval_losses)),
            "eval_loss_std": float(np.std(eval_losses)),
            "eval_loss_min": float(np.min(eval_losses)),
            "eval_loss_max": float(np.max(eval_losses)),
            "eval_loss_median": float(np.median(eval_losses)),
            "eval_perplexity_mean": float(np.mean(eval_perplexities)),
            "eval_perplexity_std": float(np.std(eval_perplexities))
        },
        "training_data_characteristics": {
            "train_loss_mean": float(np.mean(train_losses)),
            "train_loss_std": float(np.std(train_losses)),
            "train_perplexity_mean": float(np.mean(train_perplexities)),
            "train_perplexity_std": float(np.std(train_perplexities))
        },
        "coverage_metrics": coverage,
        "improvement_estimates": improvement
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def assess_single_training_file(
    train_file_path: str,
    eval_data: List[Dict],
    eval_texts: List[str],
    eval_losses: List[float],
    eval_perplexities: List[float],
    eval_features: List[Dict],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    max_length: int
) -> Dict:
    """Assess a single training file against eval data."""
    
    filename = os.path.basename(train_file_path)
    
    # Load training data
    try:
        train_data = load_jsonl(train_file_path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None
    
    if len(train_data) == 0:
        print(f"⚠️  Empty training file: {filename}")
        return None
    
    # Format training conversations
    train_texts = []
    for data in tqdm(train_data, desc=f"Formatting {filename[:30]}", leave=False):
        try:
            text = format_conversation(data, tokenizer)
            train_texts.append(text)
        except Exception as e:
            train_texts.append("")
    
    # Compute losses on training data
    train_losses = compute_loss_batch(
        model, tokenizer, train_texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device
    )
    
    # Compute perplexities
    train_perplexities = [np.exp(loss) for loss in train_losses if np.isfinite(loss)]
    
    # Extract features
    train_features = [get_example_features(data) for data in train_data]
    
    # Compute coverage metrics
    coverage = compute_coverage_metrics(train_data, eval_data, train_features, eval_features)
    
    # Estimate improvement
    improvement = estimate_improvement_potential(
        train_losses, eval_losses, train_perplexities, eval_perplexities
    )
    
    # Compute quality scores
    coverage_score = (
        (1.0 - coverage['system_ratio_diff']) * 0.3 +
        coverage['keyword_overlap'] * 0.4 +
        (1.0 / (1.0 + coverage['length_kl_divergence'])) * 0.3
    )
    difficulty_score = min(improvement['difficulty_ratio'], 2.0) / 2.0
    consistency_score = improvement['consistency_score']
    overall_score = (coverage_score * 0.4 + difficulty_score * 0.3 + consistency_score * 0.3)
    
    return {
        "train_file": train_file_path,
        "train_size": len(train_data),
        "train_losses": train_losses,
        "train_perplexities": train_perplexities,
        "train_features": train_features,
        "coverage": coverage,
        "improvement": improvement,
        "scores": {
            "coverage_score": coverage_score,
            "difficulty_score": difficulty_score,
            "consistency_score": consistency_score,
            "overall_score": overall_score
        }
    }


def calculate_ranks(all_results: List[Dict]) -> Dict:
    """Calculate rank for each result based on overall_score, preserving original order."""
    # Create list of (index, score) tuples
    scores_with_index = [(i, result['scores']['overall_score']) for i, result in enumerate(all_results)]
    # Sort by score (descending) to determine ranks
    sorted_scores = sorted(scores_with_index, key=lambda x: x[1], reverse=True)
    
    # Create rank mapping: index -> rank (handle ties)
    rank_map = {}
    current_rank = 1
    
    for i, (idx, score) in enumerate(sorted_scores):
        # If this score is different from previous, update rank
        if i > 0 and score < sorted_scores[i-1][1]:
            current_rank = i + 1
        rank_map[idx] = current_rank
    
    return rank_map


def print_comparison_report(all_results: List[Dict], eval_data: List[Dict]):
    """Print comparison report for all training files in original order."""
    
    # Calculate ranks
    rank_map = calculate_ranks(all_results)
    
    print(f"\n{'Rank':<6} {'File':<60} {'Score':<8} {'Est. Loss':<12} {'Train Loss':<12} {'Size':<8}")
    print("-" * 110)
    
    # Print in original order with ranks
    for idx, result in enumerate(all_results):
        rank = rank_map[idx]
        filename = os.path.basename(result['train_file'])
        if len(filename) > 58:
            filename = filename[:55] + "..."
        score = result['scores']['overall_score']
        est_loss = result['improvement']['estimated_final_loss']
        train_loss = result['improvement']['train_avg_loss']
        size = result['train_size']
        print(f"{rank:<6} {filename:<60} {score:<8.3f} {est_loss:<12.4f} {train_loss:<12.4f} {size:<8}")
    
    # Print baseline eval loss
    if all_results:
        baseline_loss = all_results[0]['improvement']['baseline_eval_loss']
        print(f"\nBaseline Eval Loss (pretrained model): {baseline_loss:.4f}")


def save_comparison_report_txt(all_results: List[Dict], eval_data: List[Dict], output_file: str = "ass_res.txt"):
    """Save comparison report to text file in original order."""
    
    # Calculate ranks
    rank_map = calculate_ranks(all_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{'Rank':<6} {'File':<60} {'Score':<8} {'Est. Loss':<12} {'Train Loss':<12} {'Size':<8}\n")
        f.write("-" * 110 + "\n")
        
        # Write in original order with ranks
        for idx, result in enumerate(all_results):
            rank = rank_map[idx]
            filename = os.path.basename(result['train_file'])
            if len(filename) > 58:
                filename = filename[:55] + "..."
            score = result['scores']['overall_score']
            est_loss = result['improvement']['estimated_final_loss']
            train_loss = result['improvement']['train_avg_loss']
            size = result['train_size']
            f.write(f"{rank:<6} {filename:<60} {score:<8.3f} {est_loss:<12.4f} {train_loss:<12.4f} {size:<8}\n")
        
        # Write baseline eval loss
        if all_results:
            baseline_loss = all_results[0]['improvement']['baseline_eval_loss']
            f.write(f"\nBaseline Eval Loss (pretrained model): {baseline_loss:.4f}\n")


def save_comparison_report(all_results: List[Dict], output_file: str, eval_data: List[Dict]):
    """Save comparison report to JSON file."""
    
    # Prepare summary data
    comparison_data = {
        "eval_data_size": len(eval_data),
        "num_training_files": len(all_results),
        "baseline_eval_loss": all_results[0]['improvement']['baseline_eval_loss'] if all_results else None,
        "results": []
    }
    
    # Sort by overall score
    sorted_results = sorted(all_results, key=lambda x: x['scores']['overall_score'], reverse=True)
    
    for rank, result in enumerate(sorted_results, 1):
        result_data = {
            "rank": rank,
            "train_file": result['train_file'],
            "train_size": result['train_size'],
            "scores": result['scores'],
            "improvement": {
                "estimated_final_loss": result['improvement']['estimated_final_loss'],
                "estimated_improvement": result['improvement']['estimated_improvement'],
                "difficulty_ratio": result['improvement']['difficulty_ratio'],
                "train_avg_loss": result['improvement']['train_avg_loss'],
                "train_avg_perplexity": result['improvement']['train_avg_perplexity']
            },
            "coverage": {
                "keyword_overlap": result['coverage']['keyword_overlap'],
                "system_ratio_diff": result['coverage']['system_ratio_diff'],
                "length_kl_divergence": result['coverage']['length_kl_divergence']
            }
        }
        comparison_data["results"].append(result_data)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)


def main():
    # Default training file paths (can be overridden by command line)
    default_train_paths = [
        "backtest/hf_datajsonl/Brent123456_FlockOff_HK_08_079e2235b71461bfa7a01dc5f170516b7ecc8562.jsonl",
        "backtest/hf_datajsonl/Brent123456_FlockOff_HK_22_f845249fd5d325651c38067239114cd2aa0e25b7.jsonl",
        "backtest/hf_datajsonl/FluxSphere_FFQYr4jLzvoO7wNz_99c4c797c8b6aee68623a1d81f6301e201551781.jsonl",
        "backtest/hf_datajsonl/Brent123456_FlockOff_JI_15_be2e0adf40b3f926a4ca0aa3392f19497c794e84.jsonl",
        "backtest/hf_datajsonl/policewoman_Streetgirl10_2d5251db5275a723c029ace8f75fe31c7b9ab91a.jsonl",
        "backtest/hf_datajsonl/shining02_EvalD_105058_f94561c85af952989e367263be2ee3b029d0cb5a.jsonl",
        "backtest/hf_datajsonl/JokerJokerJoker_KingOfHell13_42e787c1c85259cde3967c4ec440e518daa84cee.jsonl",
        "backtest/hf_datajsonl/sharkAI333_my_repo_6_7fba2ef48deffd611fedc848f336c61a9fc676e8.jsonl",
        "backtest/hf_datajsonl/nest102_Parrot02_7343ffafce2661d195487874a15a7c0b3269b47e.jsonl",
        "backtest/hf_datajsonl/mama-chen_IronLady14_c282d72a1cd6cc0702da2f67c84e10e234656a83.jsonl",
        "backtest/hf_datajsonl/DriftVale_i5AxxDccqFxXcq2l_3784cf687388c948ca2d676e1149cc339389e80d.jsonl",
        "backtest/hf_datajsonl/policewoman_Streetgirl11_a53d11cb1c66383b5287a627889f921e65b7a506.jsonl",
        "backtest/hf_datajsonl/nest102_Parrot04_8281efde7ddae65d724f7819ca78a06d1e666083.jsonl"
    ]
    
    default_eval_path = "./backtest/eval_data.jsonl"
    
    parser = argparse.ArgumentParser(description="Assess training dataset(s) and estimate eval loss")
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to single training JSONL file (if not provided, uses default list)"
    )
    parser.add_argument(
        "--train_files",
        type=str,
        nargs="+",
        default=None,
        help="Paths to multiple training JSONL files (overrides default list)"
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=default_eval_path,
        help=f"Path to evaluation JSONL file (default: {default_eval_path})"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name for loss computation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for model downloads"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="dataset_assessment_report.json",
        help="Path to save detailed JSON report"
    )
    parser.add_argument(
        "--sample_eval",
        type=int,
        default=None,
        help="Optional: Sample N examples from eval set for faster computation"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Determine which training files to use
    if args.train_file:
        # Single file mode
        train_files = [args.train_file]
    elif args.train_files:
        # Multiple files from command line
        train_files = args.train_files
    else:
        # Use default list
        train_files = default_train_paths
    
    # Load eval data (only once)
    eval_data = load_jsonl(args.eval_file)
    
    # Sample eval data if requested
    if args.sample_eval and args.sample_eval < len(eval_data):
        np.random.seed(42)
        indices = np.random.choice(len(eval_data), args.sample_eval, replace=False)
        eval_data = [eval_data[i] for i in indices]
    
    # Load model (only once)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map="auto" if device == "cuda" else None,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    # Format eval conversations (only once)
    eval_texts = []
    for data in tqdm(eval_data, desc="Formatting eval data"):
        try:
            text = format_conversation(data, tokenizer)
            eval_texts.append(text)
        except Exception as e:
            eval_texts.append("")
    
    # Compute eval losses (only once)
    eval_losses = compute_loss_batch(
        model, tokenizer, eval_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device
    )
    
    # Compute eval perplexities
    eval_perplexities = [np.exp(loss) for loss in eval_losses if np.isfinite(loss)]
    
    # Extract eval features (only once)
    eval_features = [get_example_features(data) for data in eval_data]
    
    # Assess each training file
    all_results = []
    
    for train_file in train_files:
        if not os.path.exists(train_file):
            continue
        
        result = assess_single_training_file(
            train_file,
            eval_data,
            eval_texts,
            eval_losses,
            eval_perplexities,
            eval_features,
            model,
            tokenizer,
            device,
            args.batch_size,
            args.max_length
        )
        
        if result is not None:
            all_results.append(result)
    
    if len(all_results) == 0:
        print("❌ No valid training files were assessed!")
        return
    
    # Print comparison report (sorted)
    print_comparison_report(all_results, eval_data)
    
    # Save comparison report to text file
    save_comparison_report_txt(all_results, eval_data, "ass_res.txt")
    
    # Save comparison report to JSON
    comparison_report_file = args.report_file.replace(".json", "_comparison.json") if args.report_file.endswith(".json") else f"{args.report_file}_comparison.json"
    save_comparison_report(all_results, comparison_report_file, eval_data)
    
    # If single file mode, also save individual report
    if len(all_results) == 1:
        result = all_results[0]
        save_report(
            args.report_file,
            load_jsonl(result['train_file']), eval_data,
            result['train_losses'], eval_losses,
            result['train_perplexities'], eval_perplexities,
            result['train_features'], eval_features,
            result['coverage'], result['improvement']
        )


if __name__ == "__main__":
    main()

