"""Fine-tune Qwen3 into JARVIS-X with LoRA (Unsloth), then export a GGUF for Ollama.

    python training\\train_lora.py --dry-run                 check the data and lengths, no GPU needed
    python training\\train_lora.py --base unsloth/Qwen3-4B   train on an 8 GB card (RTX 3070)
    python training\\train_lora.py --base unsloth/Qwen3-8B   train the 8B (needs ~12-16 GB: use Colab)

What it does:
1. Loads the base model in 4-bit (QLoRA) and adds small trainable LoRA adapters.
2. Renders every example with Qwen3's own chat template, including the 17 tool
   descriptions, exactly as Jarvis sees them at run time, with thinking switched off.
3. Trains ONLY on Jarvis's turns (tool calls and spoken replies); sir's messages and
   tool results are masked out, so the model learns to answer, not to imitate input.
4. Saves the adapter and a q4_k_m GGUF you can load into Ollama with
   training\\install_model.py.

Then measure it: python evals\\compare.py --models qwen3:8b jarvis-x
Keep the fine-tune only if it beats the base model on BOTH tests.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render(tokenizer, example: dict) -> str:
    return tokenizer.apply_chat_template(example["messages"], tools=example["tools"],
                                         tokenize=False, enable_thinking=False)


def dry_run(base: str, max_seq: int) -> int:
    """Everything except training: data present, renders with the real template, fits the length."""
    from transformers import AutoTokenizer

    train, valid = load(DATA / "train.jsonl"), load(DATA / "valid.jsonl")
    tok = AutoTokenizer.from_pretrained(base.replace("unsloth/", "Qwen/"))
    lengths = [len(tok(render(tok, ex))["input_ids"]) for ex in train + valid]
    too_long = sum(1 for n in lengths if n > max_seq)
    sample = render(tok, train[0])
    print(f"train {len(train)}, valid {len(valid)} examples")
    print(f"tokens per example: min {min(lengths)}, max {max(lengths)}, avg {sum(lengths) // len(lengths)}")
    print(f"longer than --max-seq {max_seq}: {too_long}")
    print("chat template has tool calls:", "<tool_call>" in "".join(render(tok, e) for e in train[:20]))
    print("thinking switched off:", "<think>\n\n</think>" in sample)
    return 1 if too_long else 0


def train(args) -> int:
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_seq, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    def to_text(rows):
        return Dataset.from_list([{"text": render(tokenizer, ex)} for ex in rows])

    train_ds, valid_ds = to_text(load(DATA / "train.jsonl")), to_text(load(DATA / "valid.jsonl"))
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=valid_ds,
        args=SFTConfig(
            dataset_text_field="text", per_device_train_batch_size=1, gradient_accumulation_steps=8,
            num_train_epochs=args.epochs, learning_rate=args.lr, warmup_steps=5, lr_scheduler_type="linear",
            optim="adamw_8bit", weight_decay=0.01, logging_steps=5, eval_strategy="epoch",
            output_dir=str(OUT / "checkpoints"), seed=3407, report_to="none",
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported()))
    # Learn only Jarvis's turns: everything inside sir's turns (including tool results,
    # which Qwen3 places in user turns) is masked from the loss.
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    stats = trainer.train()
    print(f"Training finished in {stats.metrics.get('train_runtime', 0) / 60:.1f} minutes, "
          f"final loss {stats.metrics.get('train_loss', 0):.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT / "lora"))
    tokenizer.save_pretrained(str(OUT / "lora"))
    model.save_pretrained_gguf(str(OUT / "gguf"), tokenizer, quantization_method="q4_k_m")
    print(f"Saved the adapter to {OUT / 'lora'} and the GGUF to {OUT / 'gguf'}")
    print("Next: python training\\install_model.py, then python evals\\compare.py --models qwen3:8b jarvis-x")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="unsloth/Qwen3-4B",
                        help="unsloth/Qwen3-4B fits an 8 GB card; unsloth/Qwen3-8B needs ~12-16 GB (Colab)")
    parser.add_argument("--max-seq", type=int, default=3584, help="tokens per example (ours are ~3,000)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="check data and lengths only, no GPU needed")
    args = parser.parse_args()
    if not (DATA / "train.jsonl").exists():
        print("No dataset yet. Run: python training\\build_dataset.py")
        return 1
    return dry_run(args.base, args.max_seq) if args.dry_run else train(args)


if __name__ == "__main__":
    sys.exit(main())
