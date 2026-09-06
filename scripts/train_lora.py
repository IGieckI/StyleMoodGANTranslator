"""
Train a LoRA adapter on the UNet of Stable Diffusion 1.5 to learn a visual style
"""

import argparse
import json
from pathlib import Path
import bitsandbytes as bnb
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Chiavi di configs/lora_config.yaml sovrascrivibili da riga di comando, con il tipo da usare
# per il parsing. Restano None se non passate, cosi' il valore del YAML sopravvive.
OVERRIDABLE = [
    ("--pretrained-model-name-or-path", str),
    ("--dataset-dir", str),
    ("--output-dir", str),
    ("--run-name", str),
    ("--rank", int),
    ("--lora-alpha", int),
    ("--learning-rate", float),
    ("--lr-warmup-steps", int),
    ("--max-train-steps", int),
    ("--checkpointing-steps", int),
    ("--train-batch-size", int),
    ("--gradient-accumulation-steps", int),
    ("--seed", int),
]


def load_config():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lora_config.yaml"))
    parser.add_argument("--mood", default=None, help="scorciatoia: dataset data/processed/<mood> e run-name <mood>")
    for flag, flag_type in OVERRIDABLE:
        parser.add_argument(flag, type=flag_type, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if args.mood:
        cfg["dataset_dir"] = f"data/processed/{args.mood}"
        cfg["run_name"] = args.mood

    for flag, _ in OVERRIDABLE:
        key = flag.lstrip("-").replace("-", "_")
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    if not cfg.get("dataset_dir") or not cfg.get("run_name"):
        raise SystemExit("Servono --mood <nome> oppure --dataset-dir e --run-name espliciti.")
    return cfg


def load_dataset_card(dataset_dir):
    card_path = Path(dataset_dir) / "dataset.json"
    if not card_path.exists():
        print(f"ATTENZIONE: {card_path} non trovato, run.json restera' senza trigger.")
        return {}
    return json.loads(card_path.read_text(encoding="utf-8"))


def load_sd_components(model_path):
    """
    Load the four components of SD 1.5 plus the noise scheduler, still on CPU and in fp32
    """
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def apply_lora(unet, cfg):
    """
    Inject LoRA adapters into the UNet and return the list of trainable parameters.
    """
    unet.add_adapter(
        LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0.0),
            target_modules=cfg["target_modules"],
            init_lora_weights="gaussian",
        )
    )
    cast_training_params(unet, dtype=torch.float32)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"Parametri addestrabili: {trainable:,} su {total:,} ({100 * trainable / total:.3f}%)")
    return [p for p in unet.parameters() if p.requires_grad]


class StyleDataset(torch.utils.data.Dataset):
    """
    Read the preprocessed images and their captions from metadata.jsonl
    """

    def __init__(self, dataset_dir, tokenizer):
        self.dir = Path(dataset_dir)
        metadata = self.dir / "metadata.jsonl"
        if not metadata.exists():
            raise SystemExit(f"{metadata} not found. Execute scripts/prepare_dataset.py first.")
        self.records = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        """
        Number of images in the dataset
        """
        return len(self.records)

    def __getitem__(self, idx):
        """
        Returns a dictionary with the image and its corresponding caption
        """
        record = self.records[idx]
        image = Image.open(self.dir / record["file_name"]).convert("RGB")
        input_ids = self.tokenizer(
            record["text"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": self.transform(image), "input_ids": input_ids}


def build_dataloader(cfg, tokenizer):
    """
    Build the training DataLoader from the preprocessed dataset
    """
    dataset = StyleDataset(cfg["dataset_dir"], tokenizer)
    print(f"Dataset: {len(dataset)} immagini da {cfg['dataset_dir']}")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["train_batch_size"],
        shuffle=True,
        num_workers=0,
    )


def save_lora_checkpoint(accelerator, unet, output_dir):
    """
    Extract the LoRA weights from the UNet and save them as pytorch_lora_weights.safetensors
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(accelerator.unwrap_model(unet)))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=state_dict,
        safe_serialization=True,
    )
    print(f"  checkpoint salvato in {output_dir}")


def save_run_card(cfg, card, ckpt_dir):
    """
    Write a `run.json` file next to the weights: how to activate the style and with which parameters it was created
    """
    run_card = {
        "run_name": cfg["run_name"],
        "dataset_dir": cfg["dataset_dir"],
        "trigger": card.get("trigger"),
        "style_suffix": card.get("caption_suffix"),
        "base_model": cfg["pretrained_model_name_or_path"],
        "rank": cfg["rank"],
        "lora_alpha": cfg["lora_alpha"],
        "learning_rate": cfg["learning_rate"],
        "max_train_steps": cfg["max_train_steps"],
        "resolution": card.get("resolution"),
        "num_images": card.get("num_images"),
    }
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "run.json").write_text(json.dumps(run_card, indent=2, ensure_ascii=False), encoding="utf-8")


def training_loop(cfg, accelerator, unet, vae, text_encoder, noise_scheduler, dataloader, optimizer, lr_scheduler, ckpt_dir):
    """
    Denoising diffusion cycle: noise -> prediction -> MSE, per "max_train_steps" steps
    """
    weight_dtype = torch.float16
    vae_dtype = torch.float32 if cfg.get("vae_fp32", True) else weight_dtype
    checkpointing_steps = cfg.get("checkpointing_steps", 0)
    global_step = 0
    progress = tqdm(total=cfg["max_train_steps"], desc="training")

    while global_step < cfg["max_train_steps"]:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Image -> latent 64x64x4, scaled by the VAE's scaling factor
                latents = vae.encode(batch["pixel_values"].to(dtype=vae_dtype)).latent_dist.sample()
                latents = (latents * vae.config.scaling_factor).to(dtype=weight_dtype)

                # Add random noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # UNet + LoRA predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                
                loss = F.mse_loss(model_pred.float(), target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in unet.parameters() if p.requires_grad], cfg["max_grad_norm"]
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.detach().item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                if checkpointing_steps and global_step % checkpointing_steps == 0:
                    save_lora_checkpoint(accelerator, unet, ckpt_dir / f"step-{global_step}")
                if global_step >= cfg["max_train_steps"]:
                    break
    progress.close()


def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    )
    weight_dtype = torch.float16

    card = load_dataset_card(cfg["dataset_dir"])
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_sd_components(cfg["pretrained_model_name_or_path"])

    # Freeze everything besides the LoRA adapters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32 if cfg.get("vae_fp32", True) else weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    lora_params = apply_lora(unet, cfg)
    # Recalculate activations in the backward pass instead of keeping them in VRAM (to keep everything in 6GB memory)
    unet.enable_gradient_checkpointing()

    if cfg.get("use_8bit_adam", False):
        optimizer = bnb.optim.AdamW8bit(lora_params, lr=cfg["learning_rate"])
    else:
        optimizer = torch.optim.AdamW(lora_params, lr=cfg["learning_rate"])

    dataloader = build_dataloader(cfg, tokenizer)
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=cfg["lr_warmup_steps"] * cfg["gradient_accumulation_steps"],
        num_training_steps=cfg["max_train_steps"] * cfg["gradient_accumulation_steps"],
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, dataloader, lr_scheduler)

    ckpt_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    training_loop(cfg, accelerator, unet, vae, text_encoder, noise_scheduler, dataloader, optimizer, lr_scheduler, ckpt_dir)

    save_lora_checkpoint(accelerator, unet, ckpt_dir)
    save_run_card(cfg, card, ckpt_dir)
    print(f"\nTraining completato. Pesi LoRA in {ckpt_dir}")


if __name__ == "__main__":
    main()
