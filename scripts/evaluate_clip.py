"""
Measure the style transfer through CLIP similarity between generated images and a reference dataset
"""

import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def list_images(directory):
    paths = sorted(p for p in directory.glob("*.png"))
    if not paths:
        raise SystemExit(f"No .png images found in {directory}. Execute generate_comparison.py first")
    return paths


@torch.no_grad()
def embed_images(paths, model, processor, device):
    """
    Return CLIP embeddings
    """
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    features = model.get_image_features(**inputs)
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    return torch.nn.functional.normalize(features, dim=-1)


def score_against_reference(gen_embeds, ref_embeds):
    """
    For each generated image, compute the average similarity against all reference images
    """
    return (gen_embeds @ ref_embeds.T).mean(dim=1)


def discover_runs(checkpoints_dir, requested):
    """
    Return the pairs (name, run.json card) of runs to evaluate
    """
    available = sorted(d.name for d in checkpoints_dir.iterdir() if (d / "pytorch_lora_weights.safetensors").exists())
    names = requested or available
    if not names:
        raise SystemExit(f"No trained LoRA weights in {checkpoints_dir}")

    runs = []
    for name in names:
        card_path = checkpoints_dir / name / "run.json"
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        runs.append((name, card))
    return runs


def reference_dir_for(name, card, override):
    if override:
        return override
    if card.get("dataset_dir"):
        return Path(card["dataset_dir"])
    print(f"Warning: run '{name}' without dataset_dir in run.json, using data/processed/{name}")
    return Path("data/processed") / name


def evaluate_run(name, ref_dir, samples_dir, base_embeds, model, processor, device):
    """
    Calculate base, lora and delta for a single run and print the table per image
    """
    lora_paths = list_images(samples_dir / name)
    ref_embeds = embed_images(list_images(ref_dir), model, processor, device)

    base_scores = score_against_reference(base_embeds, ref_embeds)
    lora_scores = score_against_reference(embed_images(lora_paths, model, processor, device), ref_embeds)

    per_image = []
    print(f"\n--- {name} (riferimento: {ref_dir}) ---")
    print(f"{'immagine':<14}{'base':>8}{'lora':>8}{'delta':>9}")
    for path, base_s, lora_s in zip(lora_paths, base_scores.tolist(), lora_scores.tolist()):
        per_image.append({"image": path.name, "base": base_s, "lora": lora_s, "delta": lora_s - base_s})
        print(f"{path.name:<14}{base_s:>8.4f}{lora_s:>8.4f}{lora_s - base_s:>+9.4f}")

    return {
        "run_name": name,
        "reference_dir": str(ref_dir),
        "mean_clip_sim_base": base_scores.mean().item(),
        "mean_clip_sim_lora": lora_scores.mean().item(),
        "std_clip_sim_lora": lora_scores.std().item(),
        "delta": (lora_scores.mean() - base_scores.mean()).item(),
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=None, help="run da valutare; default: tutti quelli addestrati")
    parser.add_argument("--reference-dir", type=Path, default=None, help="forza il dataset di riferimento per tutti i run")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output", type=Path, default=Path("outputs/clip_scores.json"))
    args = parser.parse_args()

    samples_dir = Path("outputs/samples")
    runs = discover_runs(Path("outputs/checkpoints"), args.runs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carico {args.clip_model} su {device}")
    model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    base_embeds = embed_images(list_images(samples_dir / "base"), model, processor, device)

    results = []
    for name, card in runs:
        ref_dir = reference_dir_for(name, card, args.reference_dir)
        results.append(evaluate_run(name, ref_dir, samples_dir, base_embeds, model, processor, device))

    print(f"\n{'run':<14}{'base':>8}{'lora':>8}{'delta':>9}")
    for r in results:
        print(f"{r['run_name']:<14}{r['mean_clip_sim_base']:>8.4f}{r['mean_clip_sim_lora']:>8.4f}{r['delta']:>+9.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"clip_model": args.clip_model, "runs": results}, indent=2), encoding="utf-8"
    )
    print(f"\nRisultati salvati in {args.output}")


if __name__ == "__main__":
    main()
