import torch
import os
import contextlib

import comfy.utils
import comfy.model_management
from comfy.clip_vision import ClipVisionModel

from torch import nn
import torch.nn.functional as F

from .resampler import Resampler

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
        self.to_kvs = nn.ModuleList([nn.Linear(cross_attention_dim, channel, bias=False) for channel in channels])
        
    def load_state_dict(self, state_dict):
        for i, key in enumerate(state_dict.keys()):
            self.to_kvs[i].weight.data = state_dict[key]

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        patch = CrossAttentionPatch(**patch_kwargs)
        to["patches_replace"]["attn2"][key] = patch
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)

def attention(q, k, v, extra_options):
    if not hasattr(F, "multi_head_attention_forward"):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=extra_options["n_heads"]), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * (extra_options["dim_head"] ** -0.5)
        sim = F.softmax(sim, dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=extra_options["n_heads"])
    else:
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.view(b, -1, extra_options["n_heads"], extra_options["dim_head"]).transpose(1, 2),
            (q, k, v),
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])
    return out

class IPAdapter(nn.Module):
    def __init__(self, ipadapter_model, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.ipadapter_model = ipadapter_model
        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = self.ipadapter_model["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        self.clip_extra_context_tokens = clip_extra_context_tokens

        self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(self.ipadapter_model["image_proj"])
        self.ip_layers = To_KV(cross_attention_dim)
        self.ip_layers.load_state_dict(self.ipadapter_model["ip_adapter"])
    
    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        image_prompt_embeds = self.image_proj_model(clip_embed)
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

class IPAdapterPlus(IPAdapter):
    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

class CrossAttentionPatch:
    # forward for patching
    def __init__(self, weight, ipadapter, dtype, number, cond, uncond, mask=None):
        self.weights = [weight]
        self.ipadapters = [ipadapter]
        self.conds = [cond]
        self.unconds = [uncond]
        self.dtype = dtype
        self.number = number
        self.masks = [mask]
    
    def set_new_condition(self, weight, ipadapter, cond, uncond, dtype, number, mask=None):
        self.weights.append(weight)
        self.ipadapters.append(ipadapter)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.masks.append(mask)
        self.dtype = dtype

    def __call__(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        with torch.autocast("cuda", dtype=self.dtype):
            q = n
            k = context_attn2
            v = value_attn2
            b, _, _ = q.shape

            out = attention(q, k, v, extra_options)

            for weight, cond, uncond, ipadapter, mask in zip(self.weights, self.conds, self.unconds, self.ipadapters, self.masks):
                uncond_cond = torch.cat([uncond.repeat(b//2, 1, 1), cond.repeat(b//2, 1, 1)], dim=0)

                # k, v for ip_adapter
                ip_k = ipadapter.ip_layers.to_kvs[self.number*2](uncond_cond)
                ip_v = ipadapter.ip_layers.to_kvs[self.number*2+1](uncond_cond)

                ip_out = attention(q, ip_k, ip_v, extra_options)
                
                if mask is not None:
                    mask_size = mask.shape[0] * mask.shape[1]
                    down_sample_rate = int((mask_size // 64 // out.shape[1]) ** (1/2))
                    mask_downsample = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), scale_factor= 1/8/down_sample_rate, mode="nearest").squeeze(0)
                    mask_downsample = mask_downsample.view(1,-1, 1).repeat(out.shape[0], 1, out.shape[2])
                    ip_out = ip_out * mask_downsample

                out = out + ip_out * weight

        return out.to(dtype=org_dtype)

class IPAdapterCLIPVisionEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "image": ("IMAGE",)
                             }}
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "encode"

    CATEGORY = "ipadapter"

    def encode(self, clip_vision, image):
        output = clip_vision.encode_image_IPAdapter(image)
        return (output,)

class IPAdapterModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ipadapter_file": (get_filename_list(MODELS_DIR), )}}

    RETURN_TYPES = ("IPADAPTER",)
    FUNCTION = "load_ipadapter_model"

    CATEGORY = "ipadapter"

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
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.1 }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter"

    def apply_ipadapter(self, model, ipadapter, clip_vision_output, weight, mask=None):
        self.dtype = model.model.diffusion_model.dtype
        self.device = comfy.model_management.get_torch_device()
        self.weight = weight

        # check if we are using the 'CLIP Vision Encode IPAdapter' node
        self.is_ipadapter_encode = type(clip_vision_output) is tuple

        if self.is_ipadapter_encode:
            clip_embed, clip_embed_zeroed = clip_vision_output
        else:
            clip_embed = clip_vision_output.image_embeds
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        self.is_plus = "latents" in ipadapter["image_proj"]
        cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]

        if self.is_plus:
            # IPAdapter Plus requires the IPAdapter Encoder
            if not self.is_ipadapter_encode:
                raise Exception("IPAdapter Plus requires 'CLIP Vision Encode (IPAdapter)' node")

            clip_extra_context_tokens = 16
            clip_embed = clip_embed.hidden_states[-2]
        else:
            # if we come from 'CLIP Vision Encode IPAdapter' we need to extract the image_embeds
            if self.is_ipadapter_encode:
                clip_embed = clip_embed.image_embeds

            clip_embed_zeroed = torch.zeros_like(clip_embed)
            clip_extra_context_tokens = 4
        
        clip_embeddings_dim = clip_embed.shape[-1]

        self.is_sdxl = cross_attention_dim == 2048

        IPA = IPAdapterPlus if self.is_plus else IPAdapter

        self.ipadapter = IPA(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens
        )
        
        self.ipadapter.to(self.device, dtype=self.dtype)

        self.image_prompt_embeds, self.uncond_image_prompt_embeds = self.ipadapter.get_image_embeds(clip_embed.to(self.device, self.dtype), clip_embed_zeroed.to(self.device, self.dtype))
        self.image_prompt_embeds = self.image_prompt_embeds.to(self.device, dtype=self.dtype)
        self.uncond_image_prompt_embeds = self.uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        work_model = model.clone()

        patch_kwargs = {
            "number": 0,
            "weight": self.weight,
            "ipadapter": self.ipadapter,
            "dtype": self.dtype,
            "cond": self.image_prompt_embeds,
            "uncond": self.uncond_image_prompt_embeds,
            "mask": mask if mask is None else mask.to(self.device, dtype=self.dtype)
        }

        if not self.is_sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("input", id))
                patch_kwargs["number"] += 1
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("output", id))
                patch_kwargs["number"] += 1
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 0))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                    patch_kwargs["number"] += 1
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                    patch_kwargs["number"] += 1
            for index in range(10):
                set_model_patch_replace(work_model, patch_kwargs, ("midlle", 0, index))
                patch_kwargs["number"] += 1

        return (work_model,)


# from: https://github.com/comfyanonymous/ComfyUI/blob/f88f7f413afbe04b42c4422e9deedbaa3269ce76/comfy/clip_vision.py#L39
def encode_image_IPAdapter(self, image):
    img = torch.clip((255. * image), 0, 255).round().int()
    img = list(map(lambda a: a, img))
    inputs = self.processor(images=img, return_tensors="pt")
    comfy.model_management.load_model_gpu(self.patcher)
    pixel_values = inputs['pixel_values'].to(self.load_device)

    if self.dtype != torch.float32:
        precision_scope = torch.autocast
    else:
        precision_scope = lambda a, b: contextlib.nullcontext(a)

    with precision_scope(comfy.model_management.get_autocast_device(self.load_device), torch.float32):
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        outputs_zeroed = self.model(torch.zeros_like(pixel_values), output_hidden_states=True).hidden_states[-2]

    return outputs, outputs_zeroed

# Bad developer, bad!
ClipVisionModel.encode_image_IPAdapter = encode_image_IPAdapter

NODE_CLASS_MAPPINGS = {
    "IPAdapterModelLoader": IPAdapterModelLoader,
    "IPAdapterApply": IPAdapterApply,
    "IPAdapterCLIPVisionEncode": IPAdapterCLIPVisionEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterModelLoader": "Load IPAdapter Model",
    "IPAdapterApply": "Apply IPAdapter",
    "IPAdapterCLIPVisionEncode": "CLIP Vision Encode (IPAdapter)"
}
