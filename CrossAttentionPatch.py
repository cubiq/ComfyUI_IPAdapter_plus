import torch
import math
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention
from .utils import tensor_to_size

class CrossAttentionPatch:
    # forward for patching
    def __init__(self, ipadapter=None, number=0, weight=1.0, cond=None, cond_alt=None, uncond=None, weight_type="linear", mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False, image_schedule=None, embeds_scaling='V only'):
        self.weights = [weight]
        self.ipadapters = [ipadapter]
        self.conds = [cond]
        self.conds_alt = [cond_alt]
        self.unconds = [uncond]
        self.weight_types = [weight_type]
        self.masks = [mask]
        self.sigma_starts = [sigma_start]
        self.sigma_ends = [sigma_end]
        self.unfold_batch = [unfold_batch]
        self.image_schedule = [image_schedule]
        self.embeds_scaling = [embeds_scaling]
        self.number = number
        self.layers = 11 if '101_to_k_ip' in ipadapter.ip_layers.to_kvs else 16 # TODO: check if this is a valid condition to detect all models

        self.k_key = str(self.number*2+1) + "_to_k_ip"
        self.v_key = str(self.number*2+1) + "_to_v_ip"

    def set_new_condition(self, ipadapter=None, number=0, weight=1.0, cond=None, cond_alt=None, uncond=None, weight_type="linear", mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False, image_schedule=None, embeds_scaling='V only'):
        self.weights.append(weight)
        self.ipadapters.append(ipadapter)
        self.conds.append(cond)
        self.conds_alt.append(cond_alt)
        self.unconds.append(uncond)
        self.weight_types.append(weight_type)
        self.masks.append(mask)
        self.sigma_starts.append(sigma_start)
        self.sigma_ends.append(sigma_end)
        self.unfold_batch.append(unfold_batch)
        self.image_schedule.append(image_schedule)
        self.embeds_scaling.append(embeds_scaling)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9
        block_type = extra_options["block"][0]
        #block_id = extra_options["block"][1]
        t_idx = extra_options["transformer_index"]

        # extra options for AnimateDiff
        ad_params = extra_options['ad_params'] if "ad_params" in extra_options else None

        b = q.shape[0]
        seq_len = q.shape[1]
        batch_prompt = b // len(cond_or_uncond)
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        _, _, oh, ow = extra_options["original_shape"]

        for weight, cond, cond_alt, uncond, ipadapter, mask, weight_type, sigma_start, sigma_end, unfold_batch, image_schedule, embeds_scaling in zip(self.weights, self.conds, self.conds_alt, self.unconds, self.ipadapters, self.masks, self.weight_types, self.sigma_starts, self.sigma_ends, self.unfold_batch, self.image_schedule, self.embeds_scaling):
            if sigma <= sigma_start and sigma >= sigma_end:
                if weight_type == 'ease in':
                    weight = weight * (0.05 + 0.95 * (1 - t_idx / self.layers))
                elif weight_type == 'ease out':
                    weight = weight * (0.05 + 0.95 * (t_idx / self.layers))
                elif weight_type == 'ease in-out':
                    weight = weight * (0.05 + 0.95 * (1 - abs(t_idx - (self.layers/2)) / (self.layers/2)))
                elif weight_type == 'reverse in-out':
                    weight = weight * (0.05 + 0.95 * (abs(t_idx - (self.layers/2)) / (self.layers/2)))
                elif weight_type == 'weak input' and block_type == 'input':
                    weight = weight * 0.2
                elif weight_type == 'weak middle' and block_type == 'middle':
                    weight = weight * 0.2
                elif weight_type == 'weak output' and block_type == 'output':
                    weight = weight * 0.2
                elif weight_type == 'strong middle' and (block_type == 'input' or block_type == 'output'):
                    weight = weight * 0.2
                elif isinstance(weight, dict):
                    if t_idx not in weight:
                        continue

                    weight = weight[t_idx]

                    if cond_alt is not None and t_idx in cond_alt:
                        cond = cond_alt[t_idx]
                        del cond_alt

                if unfold_batch:
                    # Check AnimateDiff context window
                    if ad_params is not None and ad_params["sub_idxs"] is not None:
                        if isinstance(weight, torch.Tensor):
                            weight = tensor_to_size(weight, ad_params["full_length"])
                            weight = torch.Tensor(weight[ad_params["sub_idxs"]])
                            if torch.all(weight == 0):
                                continue
                            weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
                        elif weight == 0:
                            continue

                        if image_schedule is not None:
                            # Use the image_schedule as a lookup table to get the embedded image corresponding to each sub_idx
                            # If image_schedule isn't long enough then use the last image
                            cond_idxs = [image_schedule[i if i < len(image_schedule) else -1] for i in ad_params["sub_idxs"]]
                            cond = torch.Tensor(cond[cond_idxs])
                            uncond = torch.Tensor(uncond[cond_idxs])
                        else:
                            # if image length matches or exceeds full_length get sub_idx images
                            if cond.shape[0] >= ad_params["full_length"]:
                                cond = torch.Tensor(cond[ad_params["sub_idxs"]])
                                uncond = torch.Tensor(uncond[ad_params["sub_idxs"]])
                            # otherwise get sub_idxs images
                            else:
                                cond = tensor_to_size(cond, ad_params["full_length"])
                                uncond = tensor_to_size(uncond, ad_params["full_length"])
                                cond = cond[ad_params["sub_idxs"]]
                                uncond = uncond[ad_params["sub_idxs"]]
                    else:
                        if isinstance(weight, torch.Tensor):
                            weight = tensor_to_size(weight, batch_prompt)
                            if torch.all(weight == 0):
                                continue
                            weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
                        elif weight == 0:
                            continue

                        cond = tensor_to_size(cond, batch_prompt)
                        uncond = tensor_to_size(uncond, batch_prompt)

                    k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond)
                    k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond)
                    v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond)
                    v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond)
                else:
                    # TODO: should we always convert the weights to a tensor?
                    if isinstance(weight, torch.Tensor):
                        weight = tensor_to_size(weight, batch_prompt)
                        if torch.all(weight == 0):
                            continue
                        weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
                    elif weight == 0:
                        continue

                    k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond).repeat(batch_prompt, 1, 1)
                    k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond).repeat(batch_prompt, 1, 1)
                    v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond).repeat(batch_prompt, 1, 1)
                    v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond).repeat(batch_prompt, 1, 1)

                ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
                ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)
                
                if embeds_scaling == 'K+mean(V) w/ C penalty':
                    scaling = float(ip_k.shape[2]) / 1280.0
                    weight = weight * scaling
                    ip_k = ip_k * weight
                    ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
                    ip_v = (ip_v - ip_v_mean) + ip_v_mean * weight
                    out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
                    del ip_v_mean
                elif embeds_scaling == 'K+V w/ C penalty':
                    scaling = float(ip_k.shape[2]) / 1280.0
                    weight = weight * scaling
                    ip_k = ip_k * weight
                    ip_v = ip_v * weight
                    out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
                elif embeds_scaling == 'K+V':
                    ip_k = ip_k * weight
                    ip_v = ip_v * weight
                    out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
                else:
                    #ip_v = ip_v * weight
                    out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
                    out_ip = out_ip * weight # I'm doing this to get the same results as before

                if mask is not None:
                    mask_h = oh / math.sqrt(oh * ow / seq_len)
                    mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
                    mask_w = seq_len // mask_h

                    # check if using AnimateDiff and sliding context window
                    if (mask.shape[0] > 1 and ad_params is not None and ad_params["sub_idxs"] is not None):
                        # if mask length matches or exceeds full_length, get sub_idx masks
                        if mask.shape[0] >= ad_params["full_length"]:
                            mask = torch.Tensor(mask[ad_params["sub_idxs"]])
                            mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
                        else:
                            mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
                            mask = tensor_to_size(mask, ad_params["full_length"])
                            mask = mask[ad_params["sub_idxs"]]
                    else:
                        mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
                        mask = tensor_to_size(mask, batch_prompt)

                    mask = mask.repeat(len(cond_or_uncond), 1, 1)
                    mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

                    # covers cases where extreme aspect ratios can cause the mask to have a wrong size
                    mask_len = mask_h * mask_w
                    if mask_len < seq_len:
                        pad_len = seq_len - mask_len
                        pad1 = pad_len // 2
                        pad2 = pad_len - pad1
                        mask = F.pad(mask, (0, 0, pad1, pad2), value=0.0)
                    elif mask_len > seq_len:
                        crop_start = (mask_len - seq_len) // 2
                        mask = mask[:, crop_start:crop_start+seq_len, :]

                    out_ip = out_ip * mask

                out = out + out_ip

        return out.to(dtype=dtype)
