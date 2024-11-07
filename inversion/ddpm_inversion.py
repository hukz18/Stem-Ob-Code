import os
import sys
from pathlib import Path
# Set the './../' from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print('WARN: __file__ not found, trying local')
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f'{Path(dir_name).parent.parent}')
# Add to path
if lib_path not in sys.path:
    print(f'Adding library path: {lib_path} to PYTHONPATH')
    sys.path.append(lib_path)
else:
    print(f'Library path {lib_path} already in PYTHONPATH')

from typing import Union, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as tvt
from diffusers import AutoencoderKL, DDIMScheduler
from inversion.partial_pipeline import DDPMPartialInversionPipeline
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch import autocast, inference_mode
import tqdm

class DDPMInversionModel:
    def __init__(self, num_steps: int = 50, inversion_steps: int = 3, img_size: int = 512, config = None, version = '2.1'):
        print("DDPMInversionModel init")
        self.dtype = torch.float32
        self.pipeline = DDPMPartialInversionPipeline
        if version == '2.1':
            self.model_id = "stabilityai/stable-diffusion-2-1"
        elif version == '1.4': 
            self.model_id = "CompVis/stable-diffusion-v1-4"
        else:
            raise ValueError(f"Invalid version {version}")
    
        self.pipe = self.pipeline.from_pretrained(self.model_id)
        self.scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        self.pipe.schedler = self.scheduler
        self.pipe.to('cuda')

        self.divided = False
        self.transposed = False
        

        if config == None:
            print("Creating inversion model from default configuration")
            self.num_steps = num_steps
            self.inversion_steps = inversion_steps
            self.do_reverse = False
            self.reverse_steps = 0
            self.reverse_skip_size = 0
            self.etas = 1.0
            self.prompts = ""
            self.prompt_src = ""
            self.prompt_tar = ""
            self.save_inversed_image = False
            self.skip_error_reduction = True
            self.guidance_scale = [3.5, 15]
        else:
            print("Creating inversion model from configuration file")
            self.num_steps = config["Diffusion"]["inference_steps"]
            self.inversion_steps = config["Diffusion"]["inversion_steps"]
            self.skip_error_reduction = config["Diffusion"]["skip_error_reduction"]
            self.do_reverse = config["Diffusion"]["do_reverse"]
            self.reverse_steps = config["Diffusion"]["reverse_steps"]
            self.reverse_skip_size = config["Diffusion"]["reverse_skip_size"]
            self.etas = config["Diffusion"]["etas"]
            self.prompt_src = config["prompts"]["src"]
            self.prompt_tar = config["prompts"]["tar"]
            self.prompts = [self.prompt_src, self.prompt_tar]
            self.save_inversed_image = config["save_inversed_image"]
            self.guidance_scale = config["guidance_scale"]
        
        height = self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width =self.pipe.unet.sample_size * self.pipe.vae_scale_factor
        self.img_size = [height, width]


    def load_image(self, image_path, left=0, right=0, top=0, bottom=0, target_size: Optional[Union[int, Tuple[int, int]]] = None,device=None):
        if type(image_path) is str:
            image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
        else:
            image = image_path
        h, w, c = image.shape
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        image = np.array(Image.fromarray(image).resize((target_size)))
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        return image
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.shape[-1] == 3:
            if len(image.shape) == 3:
                image = image.transpose(2, 0, 1)[None, ...]
            elif len(image.shape) == 4:
                image = image.transpose(0, 3, 1, 2)
            self.transposed = True
        else:
            self.transposed = False
        image = torch.from_numpy(image).contiguous()
        if isinstance(image, torch.ByteTensor):
            self.divided = True
            image = image.div(255)
        else:
            self.divided = False
        return image.to(dtype=self.pipe.dtype)
    
    def to_numpy(self, image: torch.Tensor) -> np.ndarray:
        image = image.cpu().numpy()
        if self.transposed:
            if len(image.shape) == 3:
                image = image.transpose(1, 2, 0)
            elif len(image.shape) == 4:
                image = image.transpose(0, 2, 3, 1)
        if self.divided:
            image = (image * 255).astype(np.uint8)
        return image

    def batch_to_tensor(self, images) -> torch.Tensor:
        self.is_batched = True
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                images = images[None]
                self.is_batched = False
            if images.dtype == np.uint8:
                self.divided = True
            else:
                self.divided = False
                # force to uint8
                images = (images*255).astype(np.uint8)

            images = torch.from_numpy(images).float() / 127.5 - 1
            images = images.contiguous()
            if images.shape[-1] == 3:
                # (B, W, H, C) -> (B, C, W, H)
                images = images.permute(0, 3, 1, 2).to(self.pipe.device, dtype=self.pipe.dtype) 
                self.transposed = True
            else:
                images = images.to(self.pipe.device, dtype=self.pipe.dtype)                 
                self.transposed = False
            return images
        elif isinstance(images, list):
            if isinstance(images[0], np.ndarray):
                batch_img_numpy = np.stack(images, axis=0)
                images = self.batch_to_tensor(batch_img_numpy)
                return images
            else:
                raise ValueError(f"Does not support input type {type(images[0])}")
        else:
            raise ValueError(f"Does not support input type {type(images)}")

    def batch_to_numpy(self, images) -> np.ndarray: 
        if self.transposed:
            # (B, C, W, H) -> (B, W, H, C) 
            images = images.permute(0, 2, 3, 1)
        images = images.cpu().numpy()
        if self.divided:
            images = (images * 255).astype(np.uint8)
        if not self.is_batched:
            images = images[0]
        return images
        
    def save_inversion(self, save_inversed_image: bool):
        self.save_inversed_image = save_inversed_image

    # td: invert_tensor in batch to augument training
    @torch.no_grad()
    def invert(self, images, step: int=None) -> np.ndarray:
        """
        Invert an image using the DDPM inversion pipeline.
        """
        batch_tensor = self.batch_to_tensor(images)
        inv_imgs = self.invert_tensor(batch_tensor, step=step)
        return self.batch_to_numpy(inv_imgs)
    
    @torch.no_grad()
    def invert_tensor(self, image: torch.Tensor, step: int=None) -> np.ndarray:
        """
        Invert a a batch of images using the DDPM inversion pipeline.
        """
        if isinstance(image, torch.Tensor):
            image = image.to(self.pipe.device, dtype=self.pipe.dtype)
        else:
            raise ValueError(f"Invalid image type {type(image)}")
        
        original_size, batch_size = image.shape[-2:], image.shape[0]
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
        with autocast("cuda"), inference_mode():
            w0 = (self.pipe.vae.encode(image).latent_dist.mode() * 0.18215).float()
        inv_img = self.pipe(prompt=[[self.prompt_src] * batch_size, [self.prompt_tar] * batch_size], negative_prompt=[""] * batch_size, guidance_scale=self.guidance_scale,
                            width=image.shape[-1], height=image.shape[-2],
                            output_type='latent', return_dict=False, latents=w0,
                            num_inference_steps=self.num_steps, num_inversion_steps=self.inversion_steps if step is None else step,
                            etas=self.etas, 
                            do_reverse=self.do_reverse, skip=self.reverse_skip_size, reverse_steps=self.reverse_steps,
                            save_inversed_image=self.save_inversed_image,
                            skip_error_reduction=self.skip_error_reduction)
        inv_img = TF.resize(inv_img, original_size, interpolation=TF.InterpolationMode.NEAREST)
        return inv_img.to(dtype=image.dtype)
