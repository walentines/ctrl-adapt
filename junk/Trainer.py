import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange
import sys
import torchvision

import datetime
import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from utils.utils import *
sys.path.append('..')

from torch.utils.data import Dataset

check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

class TrainControlStableVideoDiffusion:
    def __init__(self, config):
        self.config = config
        self.accelerator, self.generator = setup_accelerator(config, logger)
        self.vae, self.unet, self.text_encoder, self.controlnet, self.noise_scheduler, self.weight_dtype = instantiate_and_configure_models(config, self.accelerator, logger)
        self.unet = setup_xformers(config, logger, self.unet)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

        if config.trainer.gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()
        
        if config.trainer.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        
        if config.trainer.optimizer.scale_lr:
            config.trainer.optimizer.learning_rate = (
                config.trainer.optimizer.learning_rate * config.trainer.optimizer.gradient_accumulation_steps *
                config.dataloader.per_gpu_batch_size * self.accelerator.num_processes
            )

        self.controlnet.requires_grad_(True)

        # sam = ConfigureSAM(config.dataloader.sam_checkpoint).get_sam_inference()

        train_dataset = make_dataset(config, 'training')
        val_dataset = make_dataset(config, 'validation')

        sampler = RandomSampler(train_dataset)
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=config.dataloader.per_gpu_batch_size,
            num_workers=config.dataloader.num_of_workers,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.dataloader.per_gpu_batch_size,
            num_workers=config.dataloader.num_of_workers,
        )

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / config.trainer.gradient_accumulation_steps)
        if config.trainer.max_train_steps is None:
            config.trainer.max_train_steps = config.trainer.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        
        self.__configure_optimizers()
        
        self.unet, self.optimizer, self.lr_scheduler, self.train_dataloader, self.controlnet = self.accelerator.prepare(
            self.unet, self.optimizer, self.lr_scheduler, self.train_dataloader, self.controlnet
        )   

        num_update_steps_per_epoch = math.ceil(
        len(self.train_dataloader) / self.config.trainer.gradient_accumulation_steps)
        if overrode_max_train_steps:
            config.trainer.max_train_steps = config.trainer.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        config.trainer.num_train_epochs = math.ceil(int(config.trainer.max_train_steps) / num_update_steps_per_epoch)

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("SVDXtend", config=vars(config))
        
        total_batch_size = config.dataloader.per_gpu_batch_size * self.accelerator.num_processes * config.trainer.gradient_accumulation_steps
        self.global_step = 0
        self.first_epoch = 0

        self.progress_bar = tqdm(range(self.global_step, config.trainer.max_train_steps),
                        disable=not self.accelerator.is_local_main_process)
        self.progress_bar.set_description("Steps")

        self.min_value = 0.002
        self.max_value = 700
        self.image_d = 64
        self.noise_d_low = 32
        self.noise_d_high = 64
        self.sigma_data = 0.5
    
    def __configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.controlnet.parameters(),
            lr=self.config.trainer.optimizer.learning_rate,
            betas=(self.config.trainer.optimizer.adam_beta1, self.config.trainer.optimizer.adam_beta2),
            weight_decay=self.config.trainer.optimizer.adam_weight_decay,
            eps=self.config.trainer.optimizer.adam_epsilon,
            )
        
        self.lr_scheduler = get_scheduler(
            self.config.trainer.optimizer.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.trainer.optimizer.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.trainer.max_train_steps * self.accelerator.num_processes,
        )
    
    def training_step(self, step, batch):
        with self.accelerator.accumulate(self.controlnet):
            # Convert images to latent space
            latents = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
            latents = latents * self.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(
                dtype=self.weight_dtype
            )

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"], return_dict=False)[0]

            controlnet_image = batch["conditioning_pixel_values"].to(dtype=self.weight_dtype)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

            # Predict the noise residual
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                return_dict=False,
            )[0]

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = self.controlnet.parameters()
                self.accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
    
    def on_train_epoch_end(self):
        logs = {"step_loss": self.loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
        self.progress_bar.set_postfix(**logs)

    def on_train_step_end(self):
        self.progress_bar.update(1)
        self.global_step += 1
        self.accelerator.log({"train_loss": self.train_loss}, step=self.global_step)
        self.train_loss = 0.0

        if self.accelerator.is_main_process:
            # save checkpoints!
            if self.global_step % self.config.trainer.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if self.config.trainer.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(self.config.trainer.output_dir)
                    checkpoints = [
                        d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(
                        checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= self.config.trainer.checkpoints_total_limit:
                        num_to_remove = len(
                            checkpoints) - self.config.trainer.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(
                            f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(
                                self.config.trainer.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(
                    self.config.trainer.output_dir, f"checkpoint-{self.global_step}")
                self.accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
    def on_validation_epoch_end(self):
        del self.pipeline
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self):
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            safety_checker=None,
            revision=self.config.revision,
            variant=self.config.variant,
            torch_dtype=self.weight_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)


    def validation_step(self, step, batch):
        val_save_dir = os.path.join(self.config.trainer.output_dir, f"validation_images_{step}")

        if not os.path.exists(val_save_dir):
            os.makedirs(val_save_dir)

        with torch.autocast(
            str(self.accelerator.device).replace(":0", ""), enabled=self.accelerator.mixed_precision == "fp16"
        ):
            # for val_img_idx in range(args.num_validation_images):
            validation_images = [validation_image.squeeze(0) for validation_image in batch['pixel_values']]
            validation_control_images = [validation_image.squeeze(0) for validation_image in batch['depth_pixel_values']]
            del batch['pixel_values']
            del batch['depth_pixel_values']
            num_frames = self.config.dataloader.validation.sample_n_frames
            with torch.no_grad():
                video_frames = self.pipeline(
                    validation_images[0], 
                    validation_control_images[:14],
                    height=self.config.dataloader.validation.height,
                    width=self.config.dataloader.validation.width,
                    num_frames=num_frames,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=1,
                    noise_aug_strength=0.02,
                ).frames

            save_combined_frames(video_frames, validation_images, validation_control_images, val_save_dir)
            del validation_images
            del validation_control_images
        
    def on_train_end(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.controlnet = self.accelerator.unwrap_model(self.controlnet)

            pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
                self.config.model.pretrained_model_name_or_path,
                image_encoder=self.accelerator.unwrap_model(self.image_encoder),
                vae=self.accelerator.unwrap_model(self.vae),
                unet=self.unet,
                controlnet=self.controlnet,
            )
            pipeline.save_pretrained(self.config.trainer.output_dir)

        self.accelerator.end_training()

    def train(self):
        for _ in range(self.first_epoch, self.config.trainer.num_train_epochs):
            self.controlnet.train()
            self.train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                self.training_step(step, batch)
                if self.accelerator.sync_gradients:
                    self.on_train_step_end()
                    if self.global_step % self.config.dataloader.validation.steps == 0 and self.global_step != 0:
                        self.on_validation_epoch_start()
                        for validation_step, validation_batch in tqdm(enumerate(self.val_dataloader)):
                            self.validation_step(validation_step, validation_batch)
                        self.on_validation_epoch_end()
            self.on_train_epoch_end()
            if self.global_step >= self.config.trainer.max_train_steps:
                break
        self.on_train_end()
                


        