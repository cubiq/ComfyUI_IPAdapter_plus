import torch
import os

import comfy.utils
import comfy.model_management

from torch import nn
from transformers import CLIPVisionModelWithProjection

MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

# attention_channels
SD_V12_CHANNELS = [320] * 4 + [640] * 4 + [1280] * 4 + [1280] * 6 + [640] * 6 + [320] * 6 + [1280] * 2
SD_XL_CHANNELS = [640] * 8 + [1280] * 40 + [1280] * 60 + [640] * 12 + [1280] * 20

def get_filename_list(path):
    return [f for f in os.listdir(path) if f.endswith('.bin')]

class ImageProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class To_KV(nn.Module):
    def __init__(self, cross_attention_dim):
        super().__init__()

        channels = SD_XL_CHANNELS if cross_attention_dim == 2048 else SD_V12_CHANNELS
        self.to_kvs = nn.ModuleList([torch.nn.Linear(cross_attention_dim, channel, bias=False) for channel in channels])
        
    def load_state_dict(self, state_dict):
        # input -> output -> middle
        for i, key in enumerate(state_dict.keys()):
            self.to_kvs[i].weight.data = state_dict[key]

'''
code from I'm using the code from:
https://github.com/comfyanonymous/ComfyUI/blob/a74c5dbf3764fa598b58da8c88da823aaf8364fa/comfy/model_patcher.py#L64
'''
def set_model_patch_replace(model, patch, block_name, number, index, name="attn2"):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    to["patches_replace"][name][(block_name, number, index)] = patch

class IPAdapter(nn.Module):
    def __init__(self, ipadapter_model, clip_embeddings_dim, dtype=torch.float16, device="cuda", num_tokens=4):
        super().__init__()

        self.ipadapter_model = ipadapter_model
        self.device = device
        #self.num_tokens = num_tokens
        self.dtype = dtype

        self.cross_attention_dim = self.ipadapter_model["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        self.clip_extra_context_tokens = num_tokens
        
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )

        self.image_proj_model.load_state_dict(self.ipadapter_model["image_proj"])
        self.ip_layers = To_KV(self.cross_attention_dim)
        self.ip_layers.load_state_dict(self.ipadapter_model["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, clip_image_embeds):
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

class IPAdapterModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ipadapter_file": (get_filename_list(MODELS_DIR), )}}

    RETURN_TYPES = ("IPADAPTER",)
    FUNCTION = "load_ipadapter_model"

    CATEGORY = "loaders"

    def load_ipadapter_model(self, ipadapter_file):
        ckpt_path = os.path.join(MODELS_DIR, ipadapter_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
        keys = model.keys()

        if not "ip_adapter" in keys:
            raise Exception("invalid IPAdapter model {}".format(ckpt_path))

        return (model,)

class IPAdapterApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                             "ipadapter": ("IPADAPTER", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.1 }),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "conditioning/ipadapter"

    def apply_ipadapter(self, model, ipadapter, clip_vision_output, weight):
        self.dtype = model.model.diffusion_model.dtype
        self.device = comfy.model_management.get_torch_device()
        self.weight = weight

        clip_embeddings = clip_vision_output.image_embeds.to(self.device, dtype=self.dtype)
        clip_embeddings_dim = clip_embeddings.shape[1]

        self.sdxl = clip_embeddings_dim == 1280 # is there a better way?

        self.ipadapter = IPAdapter(
            ipadapter,
            clip_embeddings_dim,
            dtype=self.dtype,
            device=self.device
        ).to(self.device, dtype=self.dtype)

        self.image_prompt_embeds, self.uncond_image_prompt_embeds = self.ipadapter.get_image_embeds(clip_embeddings)
        self.image_prompt_embeds = self.image_prompt_embeds.to(self.device, dtype=self.dtype)
        self.uncond_image_prompt_embeds = self.uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)
        self.cond_uncond_image_emb = None

        work_model = model.clone()

        if not self.sdxl:
            number = 0 # index of to_kvs
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                work_model.set_model_attn2_replace(self.patch_forward(number), "input", id)
                number += 1
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                work_model.set_model_attn2_replace(self.patch_forward(number), "output", id)
                number += 1
            work_model.set_model_attn2_replace(self.patch_forward(number), "middle", 0)
        else:
            number = 0
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, self.patch_forward(number), "input", id, index)
                    number += 1
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, self.patch_forward(number), "output", id, index)
                    number += 1
            for index in range(10):
                set_model_patch_replace(work_model, self.patch_forward(number), "middle", 0, index)
                number += 1

        return (work_model, )

    def patch_forward(self, number):
        device = self.device.type
        def forward(n, context_attn2, value_attn2, extra_options):
            org_dtype = n.dtype
            with torch.autocast(device, dtype=self.dtype):
                q = n
                k = context_attn2
                v = value_attn2
                b, _, _ = q.shape

                if self.cond_uncond_image_emb is None or self.cond_uncond_image_emb.shape[0] != b:
                    self.cond_uncond_image_emb = torch.cat([self.uncond_image_prompt_embeds.repeat(b//2, 1, 1), self.image_prompt_embeds.repeat(b//2, 1, 1)], dim=0)

                # k, v for ip_adapter
                ip_k = self.ipadapter.ip_layers.to_kvs[number*2](self.cond_uncond_image_emb)
                ip_v = self.ipadapter.ip_layers.to_kvs[number*2+1](self.cond_uncond_image_emb)

                q, k, v, ip_k, ip_v = map(
                    lambda t: t.view(b, -1, extra_options["n_heads"], extra_options["dim_head"]).transpose(1, 2),
                    (q, k, v, ip_k, ip_v),
                )

                out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
                out = out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])

                # output of ip_adapter
                ip_out = nn.functional.scaled_dot_product_attention(q, ip_k, ip_v, attn_mask=None, dropout_p=0.0, is_causal=False)
                ip_out = ip_out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])

                out = out + ip_out * self.weight

            return out.to(dtype=org_dtype)
        return forward

NODE_CLASS_MAPPINGS = {
    "IPAdapterModelLoader": IPAdapterModelLoader,
    "IPAdapterApply": IPAdapterApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterModelLoader": "Load IPAdapter Model",
    "IPAdapterApply": "Apply IPAdapter",
}