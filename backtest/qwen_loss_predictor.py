from flockoff.validator.qwen_evaluator import QwenExampleEvaluator
import json
import os

# Your datasets
train_data_paths = [
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

eval_data_path = "./backtest/eval_data.jsonl"

# Evaluate all datasets
evaluator = QwenExampleEvaluator()
results = []
skipped = []

try:
    for train_path in train_data_paths:
        # Check if file exists
        if not os.path.exists(train_path):
            print(f"⚠️  Skipping (file not found): {train_path}")
            skipped.append(train_path)
            continue
        
        print(f"Evaluating: {train_path}")
        
        try:
            result = evaluator.predict_loss_without_training(
                train_data_path=train_path,
                eval_data_path=eval_data_path,
                method="estimated",
                max_eval_samples=100,
            )
            result["train_data_path"] = train_path
            result["eval_data_path"] = eval_data_path
            results.append(result)
            print(f"  ✅ Loss: {result['predicted_eval_loss']:.4f}")
        except Exception as e:
            print(f"  ❌ Error evaluating {train_path}: {e}")
            skipped.append(train_path)
            continue

finally:
    evaluator.cleanup()

# Save results
with open("loss_prediction_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Summary: Evaluated {len(results)} datasets, skipped {len(skipped)}")
print(f"Results saved to loss_prediction_results.json")
if skipped:
    print(f"\nSkipped files:")
    for path in skipped:
        print(f"  - {path}")
