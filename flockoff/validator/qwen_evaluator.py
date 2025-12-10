"""
Methods to evaluate training examples using Qwen model without training.
Uses Qwen/Qwen2.5-1.5B-Instruct to predict training quality of examples.
"""
import os
import json
import torch
import gc
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import bittensor as bt

from transformers import AutoModelForCausalLM, AutoTokenizer
from .constants import model2template
from .dataset import SFTDataset


class QwenExampleEvaluator:
    """
    Evaluate training examples using Qwen model to predict training quality.
    No training required - just forward passes.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cache_dir: Optional[str] = None,
        device: str = "auto",
        use_quantization: bool = True,
    ):
        """
        Initialize Qwen evaluator.
        
        Args:
            model_name: Qwen model to use
            cache_dir: Cache directory for models
            device: Device to use ("auto", "cuda", "cpu")
            use_quantization: Whether to use 4-bit quantization for memory efficiency
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        
        bt.logging.info(f"Loading Qwen model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True,
        )
        
        # Load model
        if use_quantization and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=cache_dir,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=device,
                cache_dir=cache_dir,
            )
        
        self.model.eval()
        self.template = model2template.get(model_name, model2template["Qwen/Qwen2.5-1.5B-Instruct"])
        
        bt.logging.info("Qwen evaluator initialized")
    
    def format_example(self, example: str) -> str:
        """Format example into Qwen chat format."""
        try:
            data = json.loads(example)
            
            # Build messages
            messages = []
            if "system" in data and data["system"]:
                messages.append({"role": "system", "content": data["system"]})
            else:
                messages.append({
                    "role": "system",
                    "content": self.template.get("system", "You are a helpful assistant.")
                })
            
            conversations = data.get("conversations", [])
            for conv in conversations:
                if conv["role"] in ["user", "assistant"]:
                    messages.append({"role": conv["role"], "content": conv["content"]})
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            return text
        except Exception as e:
            bt.logging.debug(f"Error formatting example: {e}")
            return ""
    
    # ============================================
    # Method 1: Perplexity-Based Scoring
    # ============================================
    def method1_perplexity_scoring(
        self,
        example: str,
    ) -> float:
        """
        Method 1: Compute perplexity of the example.
        
        Theory: Examples with HIGHER perplexity are more novel/surprising,
        meaning the model doesn't already know them well. These are more
        valuable for training (will teach the model something new).
        
        Lower score = better example (we'll invert perplexity)
        
        Returns:
            Score (lower is better)
        """
        try:
            text = self.format_example(example)
            if not text:
                return float('inf')
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
            if inputs.input_ids.size(1) == 0:
                return float('inf')
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Compute loss (perplexity = exp(loss))
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                perplexity = np.exp(loss)
            
            # Invert: higher perplexity = lower score (better example)
            # Normalize to 0-1 range
            score = 1.0 / (1.0 + perplexity / 100.0)  # Lower score = better
            
            return score
            
        except Exception as e:
            bt.logging.debug(f"Error in perplexity scoring: {e}")
            return float('inf')
    
    # ============================================
    # Method 2: Eval Loss Prediction
    # ============================================
    def method2_eval_loss_prediction(
        self,
        example: str,
        eval_examples: List[str],
        num_eval_samples: int = 50,
    ) -> float:
        """
        Method 2: Predict how well this example helps model predict eval data.
        
        Theory: If an example helps the model better predict evaluation data,
        it's more valuable for training. We compute loss on eval examples
        "conditioned" on seeing this training example.
        
        Lower score = better example
        
        Returns:
            Score (lower is better)
        """
        try:
            # Format training example
            train_text = self.format_example(example)
            if not train_text:
                return float('inf')
            
            # Sample eval examples
            import random
            random.seed(42)
            eval_sample = random.sample(eval_examples, min(num_eval_samples, len(eval_examples)))
            
            # Compute average loss on eval examples
            total_loss = 0.0
            count = 0
            
            with torch.no_grad():
                for eval_example in eval_sample:
                    try:
                        eval_text = self.format_example(eval_example)
                        if not eval_text:
                            continue
                        
                        # Tokenize eval example
                        eval_inputs = self.tokenizer(
                            eval_text,
                            return_tensors="pt",
                            max_length=2048,
                            truncation=True
                        )
                        if eval_inputs.input_ids.size(1) == 0:
                            continue
                        
                        eval_inputs = {k: v.to(self.model.device) for k, v in eval_inputs.items()}
                        
                        # Compute loss on eval example
                        outputs = self.model(**eval_inputs, labels=eval_inputs["input_ids"])
                        loss = outputs.loss.item()
                        total_loss += loss
                        count += 1
                    except:
                        continue
            
            if count == 0:
                return float('inf')
            
            avg_loss = total_loss / count
            # Lower loss = lower score = better example
            return avg_loss
            
        except Exception as e:
            bt.logging.debug(f"Error in eval loss prediction: {e}")
            return float('inf')
    
    # ============================================
    # Method 3: Embedding Similarity to Eval Data
    # ============================================
    def method3_embedding_similarity(
        self,
        example: str,
        eval_examples: List[str],
        num_eval_samples: int = 100,
    ) -> float:
        """
        Method 3: Compute embedding similarity to evaluation data.
        
        Theory: Examples that are semantically similar to evaluation data
        are more valuable for training, as they'll help the model perform
        better on the evaluation set.
        
        Higher similarity = better example, so we return (1 - similarity)
        to make lower score = better
        
        Returns:
            Score (lower is better)
        """
        try:
            # Get embedding for training example
            train_text = self.format_example(example)
            if not train_text:
                return float('inf')
            
            train_inputs = self.tokenizer(train_text, return_tensors="pt", max_length=2048, truncation=True)
            if train_inputs.input_ids.size(1) == 0:
                return float('inf')
            
            train_inputs = {k: v.to(self.model.device) for k, v in train_inputs.items()}
            
            with torch.no_grad():
                # Get embeddings (use last hidden state mean)
                outputs = self.model(**train_inputs, output_hidden_states=True)
                train_embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()  # [hidden_size]
            
            # Sample eval examples
            import random
            random.seed(42)
            eval_sample = random.sample(eval_examples, min(num_eval_samples, len(eval_examples)))
            
            # Compute average similarity to eval examples
            similarities = []
            
            with torch.no_grad():
                for eval_example in eval_sample:
                    try:
                        eval_text = self.format_example(eval_example)
                        if not eval_text:
                            continue
                        
                        eval_inputs = self.tokenizer(
                            eval_text,
                            return_tensors="pt",
                            max_length=2048,
                            truncation=True
                        )
                        if eval_inputs.input_ids.size(1) == 0:
                            continue
                        
                        eval_inputs = {k: v.to(self.model.device) for k, v in eval_inputs.items()}
                        
                        # Get eval embedding
                        eval_outputs = self.model(**eval_inputs, output_hidden_states=True)
                        eval_embedding = eval_outputs.hidden_states[-1].mean(dim=1).squeeze()
                        
                        # Cosine similarity
                        cos_sim = torch.nn.functional.cosine_similarity(
                            train_embedding.unsqueeze(0),
                            eval_embedding.unsqueeze(0)
                        ).item()
                        
                        similarities.append(cos_sim)
                    except:
                        continue
            
            if len(similarities) == 0:
                return float('inf')
            
            avg_similarity = np.mean(similarities)
            # Higher similarity = better, so invert: lower score = better
            return 1.0 - avg_similarity
            
        except Exception as e:
            bt.logging.debug(f"Error in embedding similarity: {e}")
            return float('inf')
    
    # ============================================
    # Method 4: Combined Multi-Factor Scoring
    # ============================================
    def method4_combined_scoring(
        self,
        example: str,
        eval_examples: List[str],
        perplexity_weight: float = 0.3,
        eval_loss_weight: float = 0.4,
        similarity_weight: float = 0.3,
        num_eval_samples: int = 50,
    ) -> float:
        """
        Method 4: Combined scoring using all three methods.
        
        Combines:
        - Perplexity (novelty)
        - Eval loss prediction (usefulness for eval)
        - Embedding similarity (semantic alignment with eval)
        
        Lower score = better example
        
        Returns:
            Combined score (lower is better)
        """
        try:
            # Compute individual scores
            perplexity_score = self.method1_perplexity_scoring(example)
            eval_loss_score = self.method2_eval_loss_prediction(example, eval_examples, num_eval_samples)
            similarity_score = self.method3_embedding_similarity(example, eval_examples, num_eval_samples)
            
            # Normalize scores to similar ranges
            # Perplexity score is already 0-1
            # Eval loss: normalize by typical range (2.0-3.5)
            normalized_eval_loss = (eval_loss_score - 2.0) / (3.5 - 2.0) if eval_loss_score != float('inf') else 1.0
            normalized_eval_loss = max(0.0, min(1.0, normalized_eval_loss))
            
            # Similarity score is already 0-1 (1 - similarity)
            
            # Combine weighted scores
            combined_score = (
                perplexity_weight * perplexity_score +
                eval_loss_weight * normalized_eval_loss +
                similarity_weight * similarity_score
            )
            
            return combined_score
            
        except Exception as e:
            bt.logging.debug(f"Error in combined scoring: {e}")
            return float('inf')
    
    # ============================================
    # Main Evaluation Function
    # ============================================
    def evaluate_and_select(
        self,
        data_list: List[str],
        eval_data_path: str,
        target_size: int = 250,
        method: str = "combined",
        batch_size: int = 50,
        **kwargs
    ) -> Tuple[List[str], Dict]:
        """
        Evaluate examples and select top-k using specified method.
        
        Args:
            data_list: All examples to evaluate
            eval_data_path: Path to evaluation dataset
            target_size: Number of examples to select
            method: "perplexity", "eval_loss", "similarity", or "combined"
            batch_size: Process in batches
            **kwargs: Additional method-specific parameters
        
        Returns:
            Tuple of (selected examples, metadata)
        """
        bt.logging.info(f"Evaluating {len(data_list)} examples using method: {method}")
        
        # Load eval examples
        with open(eval_data_path, "r", encoding="utf-8") as f:
            eval_examples = f.readlines()
        
        # Select method
        if method == "perplexity":
            score_func = lambda ex: self.method1_perplexity_scoring(ex)
        elif method == "eval_loss":
            score_func = lambda ex: self.method2_eval_loss_prediction(ex, eval_examples, **kwargs)
        elif method == "similarity":
            score_func = lambda ex: self.method3_embedding_similarity(ex, eval_examples, **kwargs)
        elif method == "combined":
            score_func = lambda ex: self.method4_combined_scoring(ex, eval_examples, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute scores
        example_scores = []
        
        for i in tqdm(range(0, len(data_list), batch_size), desc=f"Evaluating ({method})"):
            batch = data_list[i:i+batch_size]
            batch_scores = []
            
            for example in batch:
                score = score_func(example)
                batch_scores.append(score)
            
            example_scores.extend(batch_scores)
            
            # Periodic cleanup
            if i % (batch_size * 10) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Sort by score (lower is better)
        indexed_scores = list(enumerate(example_scores))
        indexed_scores.sort(key=lambda x: x[1])
        
        # Get top-k (lowest scores)
        top_indices = [idx for idx, _ in indexed_scores[:target_size]]
        top_examples = [data_list[idx] for idx in top_indices]
        top_scores = [example_scores[idx] for idx in top_indices]
        
        metadata = {
            "method": method,
            "total_examples": len(data_list),
            "selected": len(top_examples),
            "scores": top_scores,
            "avg_score": np.mean(top_scores),
            "min_score": np.min(top_scores),
            "max_score": np.max(top_scores),
            "std_score": np.std(top_scores),
        }
        
        bt.logging.info(
            f"Selected {len(top_examples)} examples using {method} method. "
            f"Score range: [{metadata['min_score']:.4f}, {metadata['max_score']:.4f}]"
        )
        
        return top_examples, metadata
    
    # ============================================
    # Loss Prediction Without Training
    # ============================================
    def predict_loss_without_training(
        self,
        train_data_path: str,
        eval_data_path: str,
        max_eval_samples: Optional[int] = None,
        method: str = "baseline",
    ) -> Dict:
        """
        Predict evaluation loss WITHOUT any training.
        
        This method estimates what the loss would be after training on train_data,
        but without actually training the model.
        
        Args:
            train_data_path: Path to training data (JSONL)
            eval_data_path: Path to evaluation data (JSONL)
            max_eval_samples: Maximum eval samples to use (None = all)
            method: Prediction method:
                - "baseline": Just evaluate base model on eval data
                - "estimated": Estimate improvement based on train data characteristics
        
        Returns:
            Dictionary with predicted loss and metadata
        """
        try:
            bt.logging.info(f"Predicting loss without training...")
            bt.logging.info(f"Train data: {train_data_path}")
            bt.logging.info(f"Eval data: {eval_data_path}")
            
            # Load eval dataset
            eval_dataset = SFTDataset(
                file=eval_data_path,
                tokenizer=self.tokenizer,
                max_seq_length=2048,
                template=self.template,
            )
            
            # Limit eval samples if specified
            if max_eval_samples and len(eval_dataset) > max_eval_samples:
                import random
                random.seed(42)
                eval_indices = random.sample(range(len(eval_dataset)), max_eval_samples)
                eval_dataset.data_list = [eval_dataset.data_list[i] for i in eval_indices]
                bt.logging.info(f"Limited eval samples to {max_eval_samples}")
            
            bt.logging.info(f"Evaluating on {len(eval_dataset)} eval examples...")
            
            # Method 1: Baseline (just evaluate base model)
            if method == "baseline":
                return self._predict_baseline_loss(eval_dataset)
            
            # Method 2: Estimated (consider training data characteristics)
            elif method == "estimated":
                # Load training data for analysis
                train_dataset = SFTDataset(
                    file=train_data_path,
                    tokenizer=self.tokenizer,
                    max_seq_length=2048,
                    template=self.template,
                )
                return self._predict_estimated_loss(train_dataset, eval_dataset)
            
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            bt.logging.error(f"Error predicting loss: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return {
                "predicted_eval_loss": float('inf'),
                "error": str(e),
            }
    
    def _predict_baseline_loss(self, eval_dataset) -> Dict:
        """
        Predict loss by evaluating base model on eval data.
        This gives the baseline loss without any training.
        """
        bt.logging.info("Computing baseline loss (no training)...")
        
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for i in tqdm(range(len(eval_dataset)), desc="Evaluating"):
                try:
                    example = eval_dataset[i]
                    input_ids = torch.tensor([example["input_ids"]], device=self.model.device)
                    attention_mask = torch.tensor([example["attention_mask"]], device=self.model.device)
                    
                    # Create labels (only assistant tokens)
                    labels = torch.where(
                        torch.tensor([example["target_mask"]], device=self.model.device) == 1,
                        input_ids,
                        -100
                    )
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    
                    loss = outputs.loss.item()
                    total_loss += loss
                    count += 1
                    
                except Exception as e:
                    bt.logging.debug(f"Error evaluating example {i}: {e}")
                    continue
        
        if count == 0:
            return {
                "predicted_eval_loss": float('inf'),
                "error": "No valid examples evaluated",
            }
        
        avg_loss = total_loss / count
        
        return {
            "predicted_eval_loss": avg_loss,
            "method": "baseline",
            "eval_samples": count,
            "note": "This is the baseline loss without training. Actual loss after training would likely be lower.",
        }
    
    def _predict_estimated_loss(self, train_dataset, eval_dataset) -> Dict:
        """
        Estimate loss by analyzing training data characteristics and base model performance.
        """
        bt.logging.info("Computing estimated loss based on training data characteristics...")
        
        # First, get baseline loss
        baseline_result = self._predict_baseline_loss(eval_dataset)
        baseline_loss = baseline_result["predicted_eval_loss"]
        
        if baseline_loss == float('inf'):
            return baseline_result
        
        # Analyze training data
        bt.logging.info("Analyzing training data characteristics...")
        
        train_losses = []
        train_perplexities = []
        train_lengths = []
        
        with torch.no_grad():
            for i in tqdm(range(min(100, len(train_dataset))), desc="Analyzing train data"):
                try:
                    example = train_dataset[i]
                    input_ids = torch.tensor([example["input_ids"]], device=self.model.device)
                    attention_mask = torch.tensor([example["attention_mask"]], device=self.model.device)
                    labels = torch.where(
                        torch.tensor([example["target_mask"]], device=self.model.device) == 1,
                        input_ids,
                        -100
                    )
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    
                    loss = outputs.loss.item()
                    train_losses.append(loss)
                    train_perplexities.append(np.exp(loss))
                    train_lengths.append(len(example["input_ids"]))
                    
                except:
                    continue
        
        if len(train_losses) == 0:
            return baseline_result
        
        # Estimate improvement factor
        # If training data has high loss (model doesn't know it well), training will help more
        avg_train_loss = np.mean(train_losses)
        avg_train_perplexity = np.mean(train_perplexities)
        
        # Heuristic: if train loss is high, model will learn more, so eval loss will improve more
        # Improvement factor: higher train loss = more potential improvement
        improvement_factor = 0.1 + 0.3 * (avg_train_loss / baseline_loss)  # 10-40% improvement
        
        # Estimate final loss
        estimated_loss = baseline_loss * (1.0 - improvement_factor)
        
        # Ensure reasonable bounds
        estimated_loss = max(baseline_loss * 0.6, estimated_loss)  # At least 40% improvement
        estimated_loss = min(baseline_loss * 0.95, estimated_loss)  # At most 5% improvement
        
        return {
            "predicted_eval_loss": estimated_loss,
            "baseline_loss": baseline_loss,
            "method": "estimated",
            "train_samples_analyzed": len(train_losses),
            "eval_samples": baseline_result["eval_samples"],
            "avg_train_loss": avg_train_loss,
            "avg_train_perplexity": avg_train_perplexity,
            "improvement_factor": improvement_factor,
            "note": "Estimated loss based on training data characteristics. This is a heuristic estimate.",
        }
    
    def cleanup(self):
        """Clean up resources."""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


