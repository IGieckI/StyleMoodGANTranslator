# Style/Mood LoRA on Stable Diffusion 1.5

**Parameter-Efficient Fine-Tuning** experiment: a LoRA adapter is trained over the UNet of Stable Diffusion 1.5 such that it learns a visual "mood" from a small custom dataset.

The style is triggered by a special token (`<cjh-<mood>>`), and training captions end with
`in <cjh-scary> style`, the same sentence that activates that style at inference time.

I've used two datasets: `bright` and `scary`.

## Setup

Install `torch`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

> Default settings like used VRAM are configured for a GTX 1060 6GB.

## Pipeline

**1. Put 15-30 images per mood in `data/raw/<mood-name>/`**, they must share a common mood/style, with subjects as varied as possible (so the model learns the style and not the content). At least 384px per side.

**2. Prepare the dataset of each mood** (resize + center crop to 512x512, caption with a dedicated trigger):

```bash
python scripts/prepare_dataset.py --mood bright
python scripts/prepare_dataset.py --mood scary
```

Add `--style-description "..."` if you want to anchor the caption to semantic words beyond the trigger (it helps generalization but introduces some of the base model's bias on those words).
With a thematically varied dataset `--caption-mode blip` may be preferable: BLIP describes the *content* of each image and the trigger stays separate, helping the LoRA decouple style from subject.

**3. Train one LoRA per mood** (1200 steps by default, the parameters live in `configs/lora_config.yaml`):

```bash
python scripts/train_lora.py --mood bright
python scripts/train_lora.py --mood scary
```

**4. Generate the comparison** over all the prompts in `configs/prompts.txt`.
```bash
python scripts/generate_comparison.py
```

The seed only depends on the prompt index, so every column starts from the same noise and the differences are attributable to the adapter alone. Output: images in `outputs/samples/<variant>/` and a grid in `outputs/comparison.png`.

Useful variants:

```bash
# compare different adapter intensities: one column per scale
python scripts/generate_comparison.py --lora-scales 1.0 1.5

# add a run without regenerating the columns already done
python scripts/generate_comparison.py --skip-existing
```

`--lora-scales` tunes the intensity of the adapter: 1.5 makes the style obvious, above 2.0 it
degrades the image.

**5. Evaluate with CLIP**:

```bash
python scripts/evaluate_clip.py
```

It prints and saves to `outputs/clip_scores.json` the average CLIP similarity between generated images and the reference dataset of each mood. The key metric is the **delta** (lora − base).

> The main limitation of this metric is that CLIP similarity is dominated by content, not style.
> The delta can stay small while visual inspection shows a clear style transfer.