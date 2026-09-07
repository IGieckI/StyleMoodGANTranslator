"""
Generate a comparison grid of images generated with the base model and each LoRA
"""
import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

# Grid layout
THUMB = 320
GUTTER = 240
HEADER = 28
PAD = 8


def read_prompts(path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line and not line.startswith("#")]
    if not prompts:
        raise SystemExit(f"Nessun prompt valido in {path}")
    return prompts


def discover_runs(checkpoints_dir, requested):
    available = sorted(d.name for d in checkpoints_dir.iterdir() if (d / "pytorch_lora_weights.safetensors").exists())
    names = requested or available
    if not names:
        raise SystemExit(f"Nessun LoRA addestrato in {checkpoints_dir}. Esegui prima train_lora.py.")

    runs = []
    for name in names:
        run_dir = checkpoints_dir / name
        if not (run_dir / "pytorch_lora_weights.safetensors").exists():
            raise SystemExit(f"Pesi LoRA non trovati in {run_dir}. Run disponibili: {', '.join(available) or 'nessuno'}")
        card_path = run_dir / "run.json"
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        runs.append((name, card))
    return runs


def style_suffix_for(name, card, override):
    if override:
        return override
    if card.get("style_suffix"):
        return card["style_suffix"]
    print(f"ATTENZIONE: run '{name}' senza style_suffix in run.json, uso la convenzione <cjh-{name}>.")
    return f"in <cjh-{name}> style"


def load_pipeline(base_model):
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_set(pipe, prompts, out_dir, args, suffix="", label=""):
    """
    Generate an image for each prompt in `out_dir` and return the list of images
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for p_idx, prompt in enumerate(prompts):
        full_prompt = f"{prompt}, {suffix}" if suffix else prompt
        name = f"{p_idx:02d}.png"
        path = out_dir / name

        if args.skip_existing and path.exists():
            images.append(Image.open(path).convert("RGB"))
            print(f"  [{label}] {name}: gia' presente, salto")
            continue

        generator = torch.Generator(device="cuda").manual_seed(args.seed + 1000 * p_idx)
        image = pipe(
            full_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        image.save(path)
        images.append(image)
        print(f"  [{label}] {name}: {full_prompt}")
    return images


def load_font(size):
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build_comparison_grid(columns, prompts, output_path):
    n_cols, n_rows = len(columns), len(prompts)
    width = GUTTER + n_cols * THUMB
    height = HEADER + n_rows * THUMB
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    header_font, label_font = load_font(18), load_font(15)

    for col, (label, _) in enumerate(columns):
        x = GUTTER + col * THUMB + THUMB // 2 - draw.textlength(label, font=header_font) // 2
        draw.text((x, PAD // 2), label, fill="black", font=header_font)

    for row, prompt in enumerate(prompts):
        y = HEADER + row * THUMB
        lines = wrap_text(draw, prompt, label_font, GUTTER - 2 * PAD)
        text_y = y + max(PAD, (THUMB - len(lines) * 18) // 2)
        for i, line in enumerate(lines):
            draw.text((PAD, text_y + i * 18), line, fill="black", font=label_font)
        for col, (_, images) in enumerate(columns):
            grid.paste(images[row].resize((THUMB, THUMB), Image.LANCZOS), (GUTTER + col * THUMB, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"\nGriglia di confronto: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--runs", nargs="*", default=None, help="run da confrontare; default: tutti quelli addestrati")
    parser.add_argument("--prompts-file", type=Path, default=Path("configs/prompts.txt"))
    parser.add_argument(
        "--style-suffix",
        default=None,
        help="forza il suffisso di attivazione per tutti i run; default: quello salvato in run.json",
    )
    parser.add_argument(
        "--lora-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="intensita' dell'adapter; piu' valori generano piu' colonne per run (>1 accentua lo stile)",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison.png"))
    parser.add_argument("--skip-existing", action="store_true", help="non rigenera le immagini gia' presenti")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    args = parser.parse_args()

    prompts = read_prompts(args.prompts_file)
    runs = discover_runs(Path("outputs/checkpoints"), args.runs)
    samples_dir = Path("outputs/samples")
    print(f"Prompt: {len(prompts)} | run: {', '.join(name for name, _ in runs)} | scale: {args.lora_scales}")

    pipe = load_pipeline(args.base_model)

    print(f"\nGenerazione BASE ({len(prompts)} prompt)")
    columns = [("BASE", generate_set(pipe, prompts, samples_dir / "base", args, label="base"))]

    for name, card in runs:
        suffix = style_suffix_for(name, card, args.style_suffix)
        lora_path = Path("outputs/checkpoints") / name
        print(f"\nCarico i pesi LoRA di '{name}' da {lora_path}")
        pipe.load_lora_weights(str(lora_path))
        for scale in args.lora_scales:
            variant = name if scale == 1.0 else f"{name}-s{scale:g}".replace(".", "")
            label = name.upper() if scale == 1.0 else f"{name.upper()} x{scale:g}"
            pipe.set_adapters(pipe.get_active_adapters(), adapter_weights=[scale])
            print(f"\nGenerazione {label} (suffisso: '{suffix}', scala: {scale})")
            columns.append((label, generate_set(pipe, prompts, samples_dir / variant, args, suffix=suffix, label=variant)))
        pipe.unload_lora_weights()

    build_comparison_grid(columns, prompts, args.output)


if __name__ == "__main__":
    main()
