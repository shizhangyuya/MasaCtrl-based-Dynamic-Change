"""
Util functions based on Diffuser framework.
"""


import os
import torch
import cv2
import numpy as np

import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torchvision.io import read_image

from diffusers import StableDiffusionPipeline
from masactrl.matrics_calculator import MetricsCalculator
from torchmetrics.multimodal import CLIPScore

from pytorch_lightning import seed_everything


class MasaCtrlPipeline(StableDiffusionPipeline):

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
        cos_optim=False,
        clip_optim=False,
        tgt_prompt=None
    ):
        """
        predict the sampe the next step in the denoise process.
        """

        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

        #梯度下降
        # if cos_optim:
        #     with torch.enable_grad():
        #         img1 = pred_x0[1].unsqueeze(0)
        #         img1.requires_grad_()
        #         img0 = pred_x0[0].unsqueeze(0)
        #
        #         # img1 = torch.randn((1, 3, 64, 64), requires_grad=True)  # 这里我们使用随机张量作为示例
        #         # img2 = torch.randn((1, 3, 64, 64))  # 这里我们使用另一个随机张量作为示例
        #
        #         # 定义优化器
        #
        #         optimizer = torch.optim.SGD([img1], lr=50) #10-100之间
        #
        #         # 定义余弦相似度损失
        #         cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        #
        #         for i in range(5):  # 运行100次迭代
        #             optimizer.zero_grad()  # 清除之前的梯度
        #
        #             # 计算余弦相似度
        #             similarity = cosine_similarity(img0.view(1, -1), img1.view(1, -1))
        #
        #             # 计算损失为1减去余弦相似度
        #             loss = 1 /(((1 + similarity)/2)*0.1)
        #
        #             # 计算梯度
        #             loss.backward()
        #
        #             # 更新图像
        #             optimizer.step()
        #
        #             print(f'Iteration {i}: Loss: {loss.item()}, Similarity: {similarity.item()}')

        if clip_optim:
            with torch.enable_grad():
                img1 = pred_x0[1].unsqueeze(0)
                # img1.requires_grad_()
                img0 = pred_x0[0].unsqueeze(0)

                # img1 = torch.randn((1, 3, 64, 64), requires_grad=True)  # 这里我们使用随机张量作为示例
                # img2 = torch.randn((1, 3, 64, 64))  # 这里我们使用另一个随机张量作为示例

                # 定义优化器



                # 定义余弦相似度损失
                # cosine_similarity = torch.nn.CosineSimilarity(dim=1)

                # metrics_calculator = MetricsCalculator('cuda')
                clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to('cuda')

                tgt_image = self.latent2image(img1)
                img_tensor = torch.tensor(tgt_image, dtype=torch.float).permute(2, 0, 1).to(self.device)
                img_tensor.requires_grad_()
                optimizer = torch.optim.SGD([img_tensor], lr=1000000.)
                for i in range(10):  # 运行100次迭代
                    optimizer.zero_grad()  # 清除之前的梯度

                    # 计算余弦相似度
                    # similarity = cosine_similarity(img1.view(1, -1), img0.view(1, -1))

                    # 计算损失为1减去余弦相似度
                    # loss = 1-similarity

                    loss = 1000./clip_metric_calculator(img_tensor, tgt_prompt)

                    # 计算梯度
                    loss.backward()

                    # 更新图像
                    optimizer.step()

                    # print(f'Iteration {i}: Loss: {loss.item()}, Similarity: {similarity.item()}')
                    print(f'Iteration {i}: Loss: {loss.item()}, Similarity: {loss}')

            pred_x0[0]=img0
            pred_x0[1]=img_tensor

        #以上是梯度下降代码
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        interpolate_step=40
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):

            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])



            # predict tghe noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents,tgt_prompt=prompt[1],clip_optim=True)

            # 自己写的：插值
            # if i == 15:
            #     latents[1] = 0.5 * model_inputs[0] + 0.5 * model_inputs[1]

            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        num_frames=32
        interpolate_timestep=self.scheduler.timesteps[interpolate_step]
        # latents[0]=(self.scheduler.alphas_cumprod[interpolate_timestep]**0.5)*latents[0]+(
        #         1-self.scheduler.alphas_cumprod[interpolate_timestep])**0.5*latents_list[0][0]
        # latents[1] = (self.scheduler.alphas_cumprod[interpolate_timestep] ** 0.5) * latents[1] + (
        #             1 - self.scheduler.alphas_cumprod[interpolate_timestep]) ** 0.5 * latents_list[0][0]

        type_noisy=5
        type_prompt=1

        if type_prompt==1:
            prompt_type='norm prompt'
        else:
            prompt_type='inter prompt'


        if type_noisy==1:
            noisy_type="no shared random noisy"
            noisy1 = torch.randn_like(latents[0])
            noisy2 = torch.randn_like(latents[0])
        elif type_noisy==2:
            noisy_type='shared random_noisy'
            noisy1 = torch.randn_like(latents[0])
            noisy2 = noisy1
        elif type_noisy==3:
            noisy_type='latent inter no shared random noisy'
            noisy1=(self.scheduler.alphas_cumprod[interpolate_timestep] ** 0.5) * latents_list[0][0] + (
                    1 - self.scheduler.alphas_cumprod[interpolate_timestep]) ** 0.5 * torch.randn_like(latents[0])
            noisy2 = (self.scheduler.alphas_cumprod[interpolate_timestep] ** 0.5) * latents_list[0][0] + (
                    1 - self.scheduler.alphas_cumprod[interpolate_timestep]) ** 0.5 * torch.randn_like(latents[1])
        elif type_noisy==4:
            noisy_type='no shared with latent noisy'
            noisy1 = (self.scheduler.alphas_cumprod[interpolate_timestep] ** 0.5) * torch.randn_like(latents[0])+ (
                    1 - self.scheduler.alphas_cumprod[interpolate_timestep]) ** 0.5 * latents_list[0][0]
            noisy2 = (self.scheduler.alphas_cumprod[interpolate_timestep] ** 0.5) * torch.randn_like(latents[1]) + (
                    1 - self.scheduler.alphas_cumprod[interpolate_timestep]) ** 0.5 * latents_list[0][0]
        else:
            noisy_type = 'latent noisy'
            noisy1 = latents_list[0][0]
            noisy2 = latents_list[0][1]


        latents[0] = (self.scheduler.alphas_cumprod[interpolate_timestep] ** 0.5) * latents[0] + (
                1 - self.scheduler.alphas_cumprod[interpolate_timestep]) ** 0.5 * noisy1
        latents[1] = (self.scheduler.alphas_cumprod[interpolate_timestep] ** 0.5) * latents[1] + (
                1 - self.scheduler.alphas_cumprod[interpolate_timestep]) ** 0.5 * noisy2

        image_grad_xt=False
        for frame in tqdm(range(num_frames),desc="Interpolate"):
            # frame_index=0.0+frame
            # rate=frame_index/num_frames
            # interpolate_latents=torch.stack([latents[0],(1-rate)*latents[0]+rate*latents[1]])
            interpolate_latents = torch.stack([latents[0],latents[1]])

            #梯度下降法生成中间图像
            if image_grad_xt:
                interpolate_latents=torch.stack([latents[0],latents[1]])

                img1=interpolate_latents[1].unsqueeze(0)
                img1.requires_grad_()
                img0=interpolate_latents[0].unsqueeze(0)

                # img1 = torch.randn((1, 3, 64, 64), requires_grad=True)  # 这里我们使用随机张量作为示例
                # img2 = torch.randn((1, 3, 64, 64))  # 这里我们使用另一个随机张量作为示例

                # 定义优化器
                optimizer = torch.optim.SGD([img1], lr=0.01)

                # 定义余弦相似度损失
                # cosine_similarity = torch.nn.CosineSimilarity(dim=1)

                metrics_calculator = MetricsCalculator('cuda')


                for i in range(20):  # 运行100次迭代
                    optimizer.zero_grad()  # 清除之前的梯度

                    # 计算余弦相似度
                    # similarity = cosine_similarity(img1.view(1, -1), img0.view(1, -1))

                    # 计算损失为1减去余弦相似度
                    # loss = 1-similarity
                    tgt_image=self.latent2image(img1)
                    tgt_prompt=prompt[1]
                    loss=metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)

                    # 计算梯度
                    loss.backward()

                    # 更新图像
                    optimizer.step()

                    # print(f'Iteration {i}: Loss: {loss.item()}, Similarity: {similarity.item()}')
                    print(f'Iteration {i}: Loss: {loss.item()}, Similarity: {loss}')
                    interpolate_latents[1] = img1
                    interpolate_latents[0] = img0

            #不求梯度，减少显存开支
            #with torch.no_grad():

                # if type_prompt==2:
                #     text_embeddings[1]=(1-rate)*text_embeddings[0]+rate*text_embeddings[1]
                #     text_embeddings[3] = (1 - rate) * text_embeddings[2] + rate * text_embeddings[3]

            for i, t in enumerate(tqdm(self.scheduler.timesteps[interpolate_step:], desc="DDIM Sampler")):

                if guidance_scale > 1.:
                    model_inputs = torch.cat([interpolate_latents] * 2)
                else:
                    model_inputs = interpolate_latents
                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])

                # predict tghe noise
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
                if guidance_scale > 1.:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                # compute the previous noise sample x_t -> x_t-1
                interpolate_latents, pred_x0 = self.step(noise_pred, t, interpolate_latents,cos_optim=True,clip_optim=True)

                # 自己写的：插值
                # if i == 15:
                #     latents[1] = 0.5 * model_inputs[0] + 0.5 * model_inputs[1]
            image = self.latent2image(interpolate_latents, return_type="pt")

            masa=True
            if masa:
                out_dir = os.path.join("./workdir/masactrl_exp/", 'masa',prompt[0]+'+'+prompt[1])
                os.makedirs(out_dir,exist_ok=True)

            else:
                out_dir = os.path.join("./workdir/masactrl_exp/",'without_masa',prompt[0]+prompt[1])
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            save_image(image,os.path.join(out_dir, f"frame{num_frames}_step{interpolate_step}_{noisy_type}_{prompt_type}_frame{frame}_.png"))

            #以上不求梯度的缩进范围至此

        assert num_frames==17

        image = self.latent2image(latents, return_type="pt")
        if not return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in tqdm(latents_list,desc="Latents")]
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents
        # print(latents)
        # exit()
        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
