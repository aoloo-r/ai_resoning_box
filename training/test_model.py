#!/usr/bin/env python3
"""Test the fine-tuned ReasoningBox model.

Usage:
    python training/test_model.py
    python training/test_model.py --model ./training/models/reasoning-box
    python training/test_model.py --interactive
"""

from __future__ import annotations
import argparse
from pathlib import Path


SYSTEM_PROMPT = (
    "You are ReasoningBox, an AI assistant trained on synthesized answers from "
    "multiple world-class AI models. You combine the best reasoning patterns from "
    "Claude, GPT-4, Gemini, and other frontier models. Provide thorough, accurate, "
    "and well-structured answers. Show your reasoning process clearly."
)

TEST_QUESTIONS = [
    "Explain the difference between TCP and UDP in simple terms.",
    "What are the pros and cons of microservices vs monolithic architecture?",
    "How does a neural network learn? Explain backpropagation step by step.",
    "Write a Python function to find the longest palindromic substring.",
    "What would happen to Earth's climate if the Moon disappeared?",
]


def main():
    parser = argparse.ArgumentParser(description="Test ReasoningBox model")
    parser.add_argument("--model", default=str(Path(__file__).parent / "models" / "reasoning-box"),
                        help="Path to merged model or LoRA checkpoint")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    def generate(question: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
                streamer=streamer,
            )

    if args.interactive:
        print("\nReasoningBox Interactive Mode")
        print("Type your questions. Type 'quit' to exit.\n")
        while True:
            try:
                question = input("\nYou: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    break
                if not question:
                    continue
                print("\nReasoningBox: ", end="")
                generate(question)
                print()
            except (EOFError, KeyboardInterrupt):
                break
        print("\nGoodbye!")
    else:
        print(f"\nRunning {len(TEST_QUESTIONS)} test questions...\n")
        for i, q in enumerate(TEST_QUESTIONS, 1):
            print(f"{'='*60}")
            print(f"Question {i}: {q}")
            print(f"{'='*60}")
            print("\nReasoningBox: ", end="")
            generate(q)
            print(f"\n")


if __name__ == "__main__":
    main()
