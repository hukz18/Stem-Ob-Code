from typing import Union, Tuple, Optional
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tvt
from diffusers import DDIMInverseScheduler, AutoencoderKL
from inversion.partial_pipeline import PartialInversionPipeline, PartialRenoisePipeline
import torchvision.transforms.functional as TF

def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * vae.config.scaling_factor
    return latents

class DDIMInversionModel:
    def __init__(self, use_renoise: bool = False, num_steps: int = 50, inversion_steps: int = 3, img_size: int = 256, dtype=torch.float16):
        self.dtype = dtype
        self.device = 'cuda'
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')
        self.use_renoise = False
        self.image_suffix = 'renoised' if self.use_renoise else 'inverted'
        self.pipeline = PartialRenoisePipeline if self.use_renoise else PartialInversionPipeline
        self.pipe = self.pipeline.from_pretrained('stabilityai/stable-diffusion-2-1',
                                                       scheduler=self.inverse_scheduler,
                                                       safety_checker=None,
                                                       torch_dtype=self.dtype,)
        self.pipe.to(self.device)
        self.num_steps = num_steps
        self.inversion_steps = inversion_steps
        self.img_size = img_size
        self.divided = False
        self.transposed = False
    
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
        return image.to(dtype=self.dtype)

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
    
    # td: invert_tensor in batch to augument training
    @torch.no_grad()
    def invert(self, image: np.ndarray, steps: int=None) -> np.ndarray:
        """
        Invert an image using the DDIM inversion pipeline.
        Input: image (np.ndarray): image to invert, shape (1, H, W, C), ranges 0-255
        Output: inv_img (np.ndarray): inverted image, shape (1, H, W, C), ranges 0-255
        """
        is_batched = len(image.shape) == 4
        image = self.to_tensor(image) # (B, C, H, W), torch.float32, 0-1 after to_tensor
        original_size = image.shape[-2:]
        if len(image.shape) == 3:
            image = image[None, ...]
        image = image.to(self.device, dtype=self.dtype)
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
        latents = img_to_latents(image, self.pipe.vae)
        inversion_steps = steps or self.inversion_steps
        if type(inversion_steps) is tuple:
            inversion_steps = np.random.randint(inversion_steps[0], inversion_steps[1] + 1)
        if len(latents.shape) == 4:
            batch_size = latents.shape[0]
            prompt = [""] * batch_size
        else:
            prompt = ""
        _, inv_img = self.pipe(prompt=prompt, negative_prompt=prompt, guidance_scale=1.,
                            width=image.shape[-1], height=image.shape[-2],
                            output_type='latent', return_dict=False, latents=latents,
                            num_inference_steps=self.num_steps, num_inversion_steps=inversion_steps)
        inv_img = TF.resize(inv_img, original_size, interpolation=TF.InterpolationMode.NEAREST)
        if inv_img.shape[0] == 1 and not is_batched:
            inv_img = inv_img[0]
        return self.to_numpy(inv_img)
    
    @torch.no_grad()
    def invert_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """
        Invert an image using the DDIM inversion pipeline.
        Input: image (torch.Tensor): image to invert, shape (B, C, H, W), ranges 0-1
        Output: inv_img (torch.Tensor): inverted image, shape (B, C, H, W), ranges 0-1
        """
        image_dtype = image.dtype
        image = image.to(self.pipe.device, dtype=self.pipe.dtype).permute(0, 3, 1, 2)
        original_size, batch_size = image.shape[-2:], image.shape[0]
        self.img_size = (128, 128)
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
        latents = img_to_latents(image, self.pipe.vae)
        inv_img = self.pipe(prompt=[""] * batch_size, negative_prompt=[""] * batch_size, guidance_scale=1.,
                            width=image.shape[-1], height=image.shape[-2],
                            output_type='latent', return_dict=False, latents=latents,
                            num_inference_steps=self.num_steps, num_inversion_steps=self.inversion_steps)
        inv_img = TF.resize(inv_img, original_size, interpolation=TF.InterpolationMode.NEAREST)
        return inv_img.permute(0, 2, 3, 1).to(dtype=image_dtype)
