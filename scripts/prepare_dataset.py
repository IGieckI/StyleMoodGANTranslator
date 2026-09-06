"""
Prepare the dataset for a given mood (resize/crop + caption)
"""
import argparse
import json
import shutil
from pathlib import Path
from transformers import pipeline

from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_and_validate_images(input_dir, min_size):
    """
    Discard images that are not readable or too small
    """
    images = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as exc:
            print(f"  scartata {path.name}: non leggibile ({exc})")
            continue
        
        if min(img.size) < min_size:
            print(f"  scartata {path.name}: {img.size[0]}x{img.size[1]} sotto il minimo di {min_size}px")
            continue
        images.append((path.name, img))
    return images


def resize_and_center_crop(img, resolution):
    """
    Set the image to resolution x resolution without deforming it
    """
    w, h = img.size
    scale = resolution / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    w, h = img.size
    left, top = (w - resolution) // 2, (h - resolution) // 2
    return img.crop((left, top, left + resolution, top + resolution))


def build_caption(trigger, style_description, content=None):
    """
    Build a training caption: "<content>, <style description>, in <trigger> style"
    """
    parts = [p for p in (content, style_description) if p]
    parts.append(f"in {trigger} style")
    return ", ".join(parts)


def build_blip_captioner():
    """
    Return a function that takes an image and returns a textual description, based on BLIP.
    """

    print("Carico BLIP per il captioning automatico...")
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    def caption(img):
        return captioner(img)[0]["generated_text"].strip()

    return caption


def resolve_paths(args):
    """
    Return input_dir, output_dir, trigger based on args.mood or explicit args.input_dir/output_dir/trigger
    """
    if args.mood is None:
        if args.input_dir is None or args.output_dir is None:
            raise SystemExit("Specifica --mood <nome>, oppure --input-dir e --output-dir espliciti.")
        if args.trigger is None:
            raise SystemExit("Senza --mood serve un --trigger esplicito, es. --trigger <cjh-scary>")
        return args.input_dir, args.output_dir, args.trigger

    input_dir = args.input_dir or Path("data/raw") / args.mood
    output_dir = args.output_dir or Path("data/processed") / args.mood
    trigger = args.trigger or f"<cjh-{args.mood}>"
    return input_dir, output_dir, trigger


def write_dataset_card(output_dir, trigger, args, num_images):
    """
    Write a dataset card (dataset.json) with metadata about the dataset
    """
    card = {
        "trigger": trigger,
        "caption_suffix": build_caption(trigger, args.style_description),
        "style_description": args.style_description,
        "resolution": args.resolution,
        "caption_mode": args.caption_mode,
        "num_images": num_images,
    }
    (output_dir / "dataset.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return card


def main():
    """
Prepare the dataset for a given mood (resize/crop + caption)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mood", default=None, help="nome del mood: deriva input, output e trigger")
    parser.add_argument("--input-dir", type=Path, default=None, help="override di data/raw/<mood>")
    parser.add_argument("--output-dir", type=Path, default=None, help="override di data/processed/<mood>")
    parser.add_argument("--trigger", default=None, help="override del trigger <cjh-<mood>>")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--min-size", type=int, default=384)
    parser.add_argument("--caption-mode", choices=["template", "blip"], default="template")
    parser.add_argument("--style-description", default="")
    parser.add_argument("--augment-flip", action="store_true", help="aggiunge il flip orizzontale di ogni immagine")
    args = parser.parse_args()

    input_dir, output_dir, trigger = resolve_paths(args)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    print(f"Reading images from {input_dir}")
    images = load_and_validate_images(input_dir, args.min_size)
    if len(images) < 5:
        raise SystemExit(
            f"Warning: {len(images)} images. At least 5 images are recommended for training a LoRA model."
        )
    if len(images) < 15:
        print(f"Warning: {len(images)} images. Under 15 images, the LoRA model tends to memorize instead of generalize the style.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    blip = build_blip_captioner() if args.caption_mode == "blip" else None

    records = []
    for name, img in images:
        img = resize_and_center_crop(img, args.resolution)
        variants = [img, ImageOps.mirror(img)] if args.augment_flip else [img]
        for variant in variants:
            file_name = f"{len(records) + 1:04d}.png"
            variant.save(output_dir / file_name)
            caption = build_caption(trigger, args.style_description, blip(variant) if blip else None)
            records.append({"file_name": file_name, "text": caption})
        print(f"  {name} -> {records[-1]['file_name']}: {records[-1]['text']}")

    with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_dataset_card(output_dir, trigger, args, len(records))

    print(f"\n{len(records)} images {args.resolution}x{args.resolution} written to {output_dir}")
    print(f"Trigger token to use in inference: 'in {trigger} style'")


if __name__ == "__main__":
    main()
