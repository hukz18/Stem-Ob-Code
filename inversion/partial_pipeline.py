import torch
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from typing import Union, List, Optional, Dict, Any, Callable
from inversion.prompt_to_prompt.ptp_classes import AttentionStore
from inversion.prompt_to_prompt.ptp_utils import register_attention_control
from tqdm import tqdm
import os
from torchvision import transforms as tvt
from functools import lru_cache

@lru_cache(maxsize=1)
def retrieve_timesteps_cached(scheduler, num_inference_steps, device, timesteps):
    return retrieve_timesteps(scheduler, num_inference_steps, device, timesteps)

class PartialInversionPipeline(StableDiffusionPipeline):
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        num_inversion_steps: int = 3,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        denoising=False,
        **kwargs,
    ):
        

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)


        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     height,
        #     width,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     ip_adapter_image,
        #     ip_adapter_image_embeds,
        #     callback_on_step_end_tensor_inputs,
        # )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        assert prompt == '' or all([p == '' for p in prompt]), "TODO: Add a prompt embed buffer"
        if getattr(self, 'null_prompt_embeds', None) is None:
            self.null_prompt_embeds, _ = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
            self.null_prompt_embeds = self.null_prompt_embeds[0]
        prompt_embeds, negative_prompt_embeds = self.null_prompt_embeds.repeat(batch_size, 1, 1), None
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps_cached(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        timesteps = timesteps[-num_inversion_steps:] if denoising else timesteps[:num_inversion_steps]
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

           
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]

        # do_denormalize = [True] * image.shape[0]

        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        image = (image / 2 + 0.5).clamp(0, 1)
        # Offload all models
        self.maybe_free_model_hooks()

        return latents, image


class PartialRenoisePipeline(StableDiffusionPipeline):
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            num_inversion_steps: int = 3,
            timesteps: List[int] = None,
            renoise_steps: int = 2,
            renoise_weights: Optional[List[float]] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
        ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        self.renoise_steps = renoise_steps
        self.renoise_weights = renoise_weights or [1.0 / renoise_steps] * renoise_steps
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps[:num_inversion_steps]):
            if self.interrupt:
                continue
            renoise_latents, prev_latents = None, latents
            for j in range(self.renoise_steps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, prev_latents, **extra_step_kwargs, return_dict=False)[0]
                if renoise_latents is None:
                    renoise_latents = latents * self.renoise_weights[j]
                else:
                    renoise_latents += latents * self.renoise_weights[j]


            latents = renoise_latents

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        # Offload all models
        self.maybe_free_model_hooks()

        return latents, image

class DDPMPartialInversionPipeline(StableDiffusionPipeline):
    def encode_text(self, prompts):
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding
    
    def sample_xts_from_x0(self, x0, num_inference_steps=50, num_inversion_steps=None):
        """
        Samples from P(x_1:T|x_0)
        """
        # torch.manual_seed(43256465436)
        alpha_bar = self.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
        batch_size = x0.shape[0]

        timesteps = self.scheduler.timesteps.to(self.device)
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
        # add batch input
        xts = torch.zeros((num_inference_steps+1, batch_size, self.unet.config.in_channels, self.unet.sample_size, self.unet.sample_size)).to(x0.device)
        xts[0] = x0
        for t in reversed(timesteps):
            idx = num_inference_steps-t_to_idx[int(t)]
            xts[idx] = x0 * (alpha_bar[t] ** 0.5) +  torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
            # fomula (6) in the ddpm_inversion paper
            if num_inversion_steps is not None and idx == num_inversion_steps:
                break
            # save_path = os.path.join(f'./results/')
            # os.makedirs(save_path, exist_ok=True)
            # with torch.autocast("cuda"), torch.inference_mode():
            #     save_image = self.vae.decode(1 / 0.18215 * xts[idx][None], return_dict=False, generator=None)[0]
            # save_image = (save_image / 2 + 0.5).clamp(0, 1)
            # save_image = save_image[0]
            # image_name_png = f'xt_idx{idx}_t{t}.png'
            # tvt.ToPILImage()(save_image.cpu()).save('./results/'+image_name_png)

        return xts
    
    def forward_step(self, model_output, timestep, sample):
        next_timestep = min(self.scheduler.config.num_train_timesteps - 2,
                            timestep + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)

        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 5. TODO: simple noising implementatiom
        next_sample = self.scheduler.add_noise(pred_original_sample,
                                        model_output,
                                        torch.LongTensor([next_timestep]))
        return next_sample


    def get_variance(self, timestep): #, prev_timestep):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
    def inversion_forward_process(self, x0, 
                            etas = None,    
                            prog_bar = False,
                            prompt = "",
                            cfg_scale = 3.5,
                            num_inference_steps=50, eps = None, num_inversion_step = None):
        condition_embedding = False
        if not prompt in ["", [""]]:
            condition_embedding = True
            text_embeddings = self.encode_text(prompt)
        # uncond_embedding = self.encode_text("")
        timesteps = self.scheduler.timesteps.to(self.device)
        batch_size = x0.shape[0]
        uncond_embedding = self.encode_text([""] * batch_size)
        variance_noise_shape = (
            num_inference_steps,
            batch_size,
            self.unet.config.in_channels, 
            self.unet.sample_size,
            self.unet.sample_size)

        if etas is None or (type(etas) in [int, float] and etas == 0):
            eta_is_zero = True
            zs = None
        else:
            eta_is_zero = False
            if type(etas) in [int, float]: etas = [etas]*self.scheduler.num_inference_steps
            xts = self.sample_xts_from_x0(x0, num_inference_steps=num_inference_steps)
            alpha_bar = self.scheduler.alphas_cumprod
            zs = torch.zeros(size=variance_noise_shape, device=self.device)
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
        xt = x0

        op = tqdm(timesteps) if prog_bar else timesteps

        for t in op:
            # idx = t_to_idx[int(t)]
            idx = num_inference_steps-t_to_idx[int(t)]-1
            # 1. predict noise residual
            if not eta_is_zero:
                # xt = xts[idx+1][None]
                xt = xts[idx+1]
                # xt = xts_cycle[idx+1][None]
            with torch.no_grad(): #predice noise from unet
                out = self.unet.forward(xt, timestep = t, encoder_hidden_states = uncond_embedding)
                # if not prompt=="":
                if condition_embedding:
                    cond_out = self.unet.forward(xt, timestep = t, encoder_hidden_states = text_embeddings)

            # if not prompt=="":
            if condition_embedding:
                ## classifier free guidance
                noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
            else:
                noise_pred = out.sample
            if eta_is_zero:
                # 2. compute more noisy image and set x_t -> x_t+1
                xt = self.forward_step(noise_pred, t, xt)

            else: # algorithm 1 in the main paper
                # xtm1 =  xts[idx][None]
                xtm1 =  xts[idx]
                # pred of x0, ontained from formula (2), the first term of DDIM equation (12)
                pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * noise_pred ) / alpha_bar[t] ** 0.5
                
                # direction to xt
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                # alpha_(t-1)
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                
                variance = self.get_variance(t)
                # DDIM equation (12), second term
                pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance ) ** (0.5) * noise_pred
                # DDIM equation
                mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

                z = (xtm1 - mu_xt ) / ( etas[idx] * variance ** 0.5 ) # formula (5)
                zs[idx] = z

                # correction to avoid error accumulation
                xtm1 = mu_xt + ( etas[idx] * variance ** 0.5 )*z

                xts[idx] = xtm1

        if not zs is None: 
            zs[0] = torch.zeros_like(zs[0]) 

        return xt, zs, xts
    
    def reverse_step(self, model_output, timestep, sample, eta = 0, variance_noise=None):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)    
        # variance = self.scheduler._get_variance(timestep, prev_timestep)
        variance = self.get_variance(timestep) #, prev_timestep)
        std_dev_t = eta * variance ** (0.5)
        # Take care of asymetric reverse process (asyrp)
        model_output_direction = model_output
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        # 8. Add noice if eta > 0
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn(model_output.shape, device=self.device)
            sigma_z =  eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z

        return prev_sample
    
    def inversion_reverse_process(self,
                        xT, 
                        etas = 0,
                        prompts = "",
                        cfg_scales = None,
                        prog_bar = False,
                        zs = None,
                        controller=None,
                        asyrp = False,
                        inver_steps = None):
        '''reverse xT to x0'''
        batch_size = len(prompts)

        cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(self.device)
        text_embeddings = self.encode_text(prompts)
        uncond_embedding = self.encode_text([""] * batch_size)

        if etas is None: etas = 0
        if type(etas) in [int, float]: etas = [etas]*self.scheduler.num_inference_steps
        assert len(etas) == self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps.to(self.device)

        xt = xT.expand(batch_size, -1, -1, -1)
        op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:] 

        t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}

        inverse_step_count = 0
        for t in op:
            idx = self.scheduler.num_inference_steps-t_to_idx[int(t)]-(self.scheduler.num_inference_steps-zs.shape[0]+1)    
            ## Unconditional embedding
            with torch.no_grad():
                uncond_out = self.unet.forward(xt, timestep =  t, 
                                                encoder_hidden_states = uncond_embedding)

                ## Conditional embedding  
            if prompts:  
                with torch.no_grad():
                    cond_out = self.unet.forward(xt, timestep =  t, 
                                                    encoder_hidden_states = text_embeddings)
                
            
            z = zs[idx] if not zs is None else None
            z = z.expand(batch_size, -1, -1, -1)
            if prompts:
                ## classifier free guidance
                noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
            else: 
                noise_pred = uncond_out.sample
            # 2. compute less noisy image and set x_t -> x_t-1  
            xt = self.reverse_step(noise_pred, t, xt, eta = etas[idx], variance_noise = z) 
            # xt = self.reverse_step(noise_pred, t, xt, eta = 0.95, variance_noise = torch.randn(noise_pred.shape, device=self.device)) 
            # xt = self.reverse_step(noise_pred, t, xt, eta = 0.15, variance_noise = z) 
            if controller is not None:
                xt = controller.step_callback(xt)  
            if inver_steps is not None:
                inverse_step_count += 1
                if inverse_step_count >= inver_steps:
                    break      
        return xt, zs
    
    def inverse_and_denoise(
            self, 
            prompt: Union[str, List[str]] = None,
            num_inference_steps: int = 50,
            num_inversion_steps: int = 3,
            skip: int = 16,
            guidance_scale: Union[float, List[float]] = 7.5,
            etas: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prog_bar = True,
            reverse_steps = 0,
            save_inversed_image = False,
            skip_error_reduction = False,
            num_loops = 1) -> torch.Tensor:
        '''
        do the following loop:
            inverse the tensor
            denoise
            inverse
            ...
        '''
        # reimplementation of the inversion_forward_process from DDPM paper
        batch_size = latents.shape[0]
        if prompt is not None and isinstance(prompt, str):
            prompt_src = [prompt] * batch_size
            prompt_tar = [prompt] * batch_size
        elif prompt is not None and isinstance(prompt, list):
            prompt_src = prompt[0]
            prompt_tar = prompt[1]
            if len(prompt_src) == 1:
                prompt_src = prompt_src * batch_size
                prompt_tar = prompt_tar * batch_size
            elif len(prompt_src) != batch_size:
                raise ValueError(f"prompt lengh {len(prompt_src)} does not match batch size {batch_size}")
        else:
            prompt_src = [""] * batch_size
            prompt_tar = [""] * batch_size
        if isinstance(guidance_scale, float):
            guidance_scale_src = guidance_scale
            guidance_scale_tar = guidance_scale
        elif isinstance(guidance_scale, list):
            guidance_scale_src = guidance_scale[0]
            guidance_scale_tar = guidance_scale[1]
        
        self.scheduler.set_timesteps(num_inference_steps)

        loop = 0
        intermediate_inversed_latents = torch.zeros((num_loops, batch_size, self.unet.config.in_channels, self.unet.sample_size, self.unet.sample_size)).to(latents.device)
        intermediate_denoised_latents = torch.zeros((num_loops, batch_size, self.unet.config.in_channels, self.unet.sample_size, self.unet.sample_size)).to(latents.device)
        while loop < num_loops:
            print(f"Doing {loop + 1} diffusion loop")
            if skip_error_reduction:
                # do not support currently
                raise ValueError(f"skip_error_reduction is not supported in multiple loops of diffusion!")
            else:
                wt, zs, wts = self.inversion_forward_process(latents, 
                                                        etas=etas, 
                                                        prompt=prompt_src, 
                                                        cfg_scale=guidance_scale_src, 
                                                        prog_bar=prog_bar, 
                                                        num_inference_steps=num_inference_steps)

                intermediate_inversed_latents[loop] = wts[num_inversion_steps][None]
            # do_reverse
                controller = AttentionStore()
                register_attention_control(self, controller)
                w0, _ = self.inversion_reverse_process(xT=wts[num_inference_steps-skip], 
                                                        etas=etas, 
                                                        prompts=prompt_tar, 
                                                        cfg_scales=[guidance_scale_tar], 
                                                        prog_bar=prog_bar, 
                                                        zs=zs[:(num_inference_steps-skip)], 
                                                        controller=controller, 
                                                        inver_steps=reverse_steps)
                save_inversed_image = False
                if save_inversed_image:
                    save_path = os.path.join(f'./results/')
                    os.makedirs(save_path, exist_ok=True)
                    with torch.autocast("cuda"), torch.inference_mode():
                        save_image = self.vae.decode(1 / 0.18215 * w0, return_dict=False, generator=generator)[0]
                    save_image = (save_image / 2 + 0.5).clamp(0, 1)
                    save_image = save_image[0]
                    image_name_png = f'intermediate_denoise_{loop}.png'
                    tvt.ToPILImage()(save_image.cpu()).save('./results/'+image_name_png)
                
                intermediate_denoised_latents[loop] = w0
                latents = w0
            loop += 1
        
        # use the denoised image as source and do inversion
        if skip_error_reduction:
            wts = self.sample_xts_from_x0(latents, num_inference_steps=num_inference_steps, num_inversion_steps=num_inversion_steps)    
        else:
            wt, zs, wts = self.inversion_forward_process(latents, 
                                                        etas=etas, 
                                                        prompt=prompt_src, 
                                                        cfg_scale=guidance_scale_src, 
                                                        prog_bar=prog_bar, 
                                                        num_inference_steps=num_inference_steps)

        if num_inversion_steps >= 0:
            w0 = wts[num_inversion_steps]
        else:
            w0 = wts[num_inference_steps]
        
        with torch.autocast("cuda"), torch.inference_mode():
            image = self.vae.decode(1 / 0.18215 * w0, return_dict=False, generator=generator)[0]
            intermediate_inversed_images = torch.zeros(num_loops, *(image.shape))
            intermediate_denoised_images = torch.zeros(num_loops, *(image.shape))
            
            for i in range(num_loops):
                intermediate_inversed_images[i] = self.vae.decode(1 / 0.18215 * intermediate_inversed_latents[i] , return_dict=False, generator=generator)[0]
                intermediate_denoised_images[i] = self.vae.decode(1 / 0.18215 * intermediate_denoised_latents[i] , return_dict=False, generator=generator)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        intermediate_inversed_images_copy = torch.zeros(num_loops, *(image.shape))
        intermediate_denoised_images_copy = torch.zeros(num_loops, *(image.shape))
        
        for i in range(num_loops):
            intermediate_inversed_images_copy[i] = (intermediate_inversed_images[i] / 2 + 0.5).clamp(0, 1)
            intermediate_denoised_images_copy[i] = (intermediate_denoised_images[i] / 2 + 0.5).clamp(0, 1)
        # Offload all models
        self.maybe_free_model_hooks()

        save_inversed_image = False
        if save_inversed_image:
            save_path = os.path.join(f'./results/')
            os.makedirs(save_path, exist_ok=True)
            save_image = intermediate_denoised_images_copy[0][0]
            # import pdb; pdb.set_trace()
            image_name_png = f'intermediate_denoise_copy_{loop}.png'
            tvt.ToPILImage()(save_image.cpu()).save('./results/'+image_name_png)

        return intermediate_inversed_images_copy, intermediate_denoised_images_copy, image

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        num_inversion_steps: int = 3,
        skip: int = 16,
        timesteps: List[int] = None,
        guidance_scale: Union[float, List[float]] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        etas: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prog_bar = True,
        do_reverse = False,
        reverse_steps = 0,
        save_inversed_image = False,
        skip_error_reduction = False,
        **kwargs,
    ):
        # reimplementation of the inversion_forward_process from DDPM paper
        batch_size = latents.shape[0]
        if prompt is not None and isinstance(prompt, str):
            prompt_src = [prompt] * batch_size
            prompt_tar = [prompt] * batch_size
        elif prompt is not None and isinstance(prompt, list):
            prompt_src = prompt[0]
            prompt_tar = prompt[1]
            if len(prompt_src) == 1:
                prompt_src = prompt_src * batch_size
                prompt_tar = prompt_tar * batch_size
            elif len(prompt_src) != batch_size:
                raise ValueError(f"prompt lengh {len(prompt_src)} does not match batch size {batch_size}")
        else:
            prompt_src = [""] * batch_size
            prompt_tar = [""] * batch_size
        if isinstance(guidance_scale, float):
            guidance_scale_src = guidance_scale
            guidance_scale_tar = guidance_scale
        elif isinstance(guidance_scale, list):
            guidance_scale_src = guidance_scale[0]
            guidance_scale_tar = guidance_scale[1]
        
        self.scheduler.set_timesteps(num_inference_steps)
        if skip_error_reduction and not do_reverse:
            wts = self.sample_xts_from_x0(latents, num_inference_steps=num_inference_steps, num_inversion_steps=num_inversion_steps)    
        else:
            wt, zs, wts = self.inversion_forward_process(latents, 
                                                        etas=etas, 
                                                        prompt=prompt_src, 
                                                        cfg_scale=guidance_scale_src, 
                                                        prog_bar=prog_bar, 
                                                        num_inference_steps=num_inference_steps)
        if save_inversed_image:
            save_path = os.path.join(f'./results/')
            os.makedirs(save_path, exist_ok=True)
            # print("================", num_inversion_steps)
            for i in range(len(wts)):
                if (i != num_inversion_steps):
                    continue
                with torch.autocast("cuda"), torch.inference_mode():
                    save_image = self.vae.decode(1 / 0.18215 * wts[i][None], return_dict=False, generator=generator)[0]
                save_image = (save_image / 2 + 0.5).clamp(0, 1)
                save_image = save_image[0]
                image_name_png = f'square_xt_{i}.png'
                tvt.ToPILImage()(save_image.cpu()).save('./results/'+image_name_png)

        # reverse process (via Zs and wT)
        if do_reverse:
            controller = AttentionStore()
            register_attention_control(self, controller)
            w0, _ = self.inversion_reverse_process(xT=wts[num_inference_steps-skip], 
                                                   etas=etas, 
                                                   prompts=prompt_tar, 
                                                   cfg_scales=[guidance_scale_tar], 
                                                   prog_bar=prog_bar, 
                                                   zs=zs[:(num_inference_steps-skip)], 
                                                   controller=controller, 
                                                   inver_steps=reverse_steps)
        elif num_inversion_steps >= 0:
            w0 = wts[num_inversion_steps]
        else:
            w0 = wts[num_inference_steps]

        with torch.autocast("cuda"), torch.inference_mode():
            image = self.vae.decode(1 / 0.18215 * w0, return_dict=False, generator=generator)[0]

        image = (image / 2 + 0.5).clamp(0, 1)

        # Offload all models
        self.maybe_free_model_hooks()

        return image