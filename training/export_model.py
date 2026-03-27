#!/usr/bin/env python3
"""Export the fine-tuned LoRA model by merging weights with the base model.

Usage:
    python training/export_model.py
    python training/export_model.py --to-gguf   # For Ollama
    python training/export_model.py --push-hub reasoning-box/ReasoningBox-7B
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model")
    parser.add_argument("--checkpoint", default=str(Path(__file__).parent / "checkpoints" / "final"),
                        help="Path to LoRA checkpoint")
    parser.add_argument("--output", default=str(Path(__file__).parent / "models" / "reasoning-box"),
                        help="Output path for merged model")
    parser.add_argument("--to-gguf", action="store_true",
                        help="Also convert to GGUF format for Ollama")
    parser.add_argument("--gguf-quant", default="Q4_K_M",
                        help="GGUF quantization type (default: Q4_K_M)")
    parser.add_argument("--push-hub", type=str, default=None,
                        help="Push to HuggingFace Hub (e.g., 'username/ReasoningBox-7B')")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    checkpoint_dir = Path(args.checkpoint)
    output_dir = Path(args.output)

    # Load training config to get base model
    config_path = checkpoint_dir / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        base_model_name = train_config["base_model"]
    else:
        # Fallback: try to read from adapter_config.json
        adapter_config = checkpoint_dir / "adapter_config.json"
        with open(adapter_config) as f:
            base_model_name = json.load(f)["base_model_name_or_path"]

    print(f"Base model: {base_model_name}")
    print(f"LoRA checkpoint: {checkpoint_dir}")

    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load and merge LoRA
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    # Save merged model
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving merged model to {output_dir}")
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save model card
    model_card = f"""---
tags:
  - reasoning-box
  - ensemble-trained
  - fine-tuned
base_model: {base_model_name}
---

# ReasoningBox

Fine-tuned on synthesized answers from multi-model AI ensemble reasoning.

## Training

This model was trained on high-quality answers produced by the AI Reasoning Box platform,
which queries multiple frontier AI models (Claude, GPT-4o, Gemini) simultaneously,
scores their responses, and synthesizes the best elements into a single answer.

## Base Model

{base_model_name}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("reasoning-box/ReasoningBox-7B")
tokenizer = AutoTokenizer.from_pretrained("reasoning-box/ReasoningBox-7B")
```
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(model_card)

    print(f"Merged model saved to {output_dir}")

    # Convert to GGUF for Ollama
    if args.to_gguf:
        print(f"\nConverting to GGUF ({args.gguf_quant})...")
        gguf_dir = output_dir.parent / f"reasoning-box-{args.gguf_quant.lower()}.gguf"

        try:
            import subprocess
            # Try using llama.cpp convert script
            result = subprocess.run([
                "python", "-m", "llama_cpp.convert",
                "--outfile", str(gguf_dir),
                "--outtype", args.gguf_quant.lower(),
                str(output_dir),
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(f"GGUF model saved to {gguf_dir}")
                print(f"\nTo use with Ollama:")
                print(f"  1. Create a Modelfile:")
                print(f'     echo "FROM {gguf_dir}" > Modelfile')
                print(f"  2. ollama create reasoning-box -f Modelfile")
                print(f"  3. ollama run reasoning-box")
            else:
                print("GGUF conversion requires llama-cpp-python. Install it with:")
                print("  pip install llama-cpp-python")
                print(f"\nOr manually convert using llama.cpp:")
                print(f"  python convert_hf_to_gguf.py {output_dir} --outtype {args.gguf_quant.lower()}")
        except Exception as e:
            print(f"GGUF conversion failed: {e}")
            print("You can manually convert later using llama.cpp tools.")

    # Push to Hub
    if args.push_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_hub}")
        merged_model.push_to_hub(args.push_hub)
        tokenizer.push_to_hub(args.push_hub)
        print(f"Model pushed to https://huggingface.co/{args.push_hub}")

    print("\nDone! Next steps:")
    print("  - Test: python training/test_model.py")
    print("  - For Ollama: python training/export_model.py --to-gguf")
    print("  - For HF Hub: python training/export_model.py --push-hub your-name/ReasoningBox-7B")


if __name__ == "__main__":
    main()
