# User Guide

Quick introduction lorem ipsum dolor sit amet, consectetur adipiscing elit. In porttitor condimentum ullamcorper. Nullam mattis tellus ac quam commodo semper. Nullam posuere diam est, sit amet lacinia velit semper et. Proin eget lacinia dui. Nam aliquet porttitor mauris et aliquam. Proin vel egestas diam. Curabitur posuere aliquam porta.

## 1. Quick Start

Lorem ipsum dolor sit amet, consectetur adipiscing elit. In porttitor condimentum ullamcorper. Nullam mattis tellus ac quam commodo semper. Nullam posuere diam est, sit amet lacinia velit semper et. Proin eget lacinia dui. Nam aliquet porttitor mauris et aliquam. Proin vel egestas diam. Curabitur posuere aliquam porta.

## 2. IP Adapter Simple

Cras convallis maximus euismod. Nullam id rutrum erat, id elementum purus. Pellentesque condimentum arcu id arcu sagittis blandit. Pellentesque euismod semper enim, nec faucibus velit hendrerit eu. Nam feugiat, tellus dignissim mollis vulputate, libero urna fermentum tortor, id ornare felis quam quis lectus. Aenean feugiat diam nisl, quis mattis turpis volutpat vel. Donec sagittis nunc tincidunt, varius elit vitae, venenatis leo. Aenean ac velit ut lectus dapibus cursus vel eget enim. Proin vel neque sit amet metus pellentesque efficitur. Mauris a lacinia elit, aliquet elementum ipsum. Nam vulputate, diam nec semper varius, neque quam interdum sapien, et commodo lorem tortor nec leo. Sed condimentum at lacus non interdum. Maecenas nec enim congue, molestie diam at, condimentum lorem. Sed porta erat vitae viverra vulputate. Curabitur sed mauris nec neque ullamcorper tempor.

![IP Adapter Simple Workflow](images/workflows/matteo/ipadapter_simple_wflow_thumb.png)

## 3. IP Adapter Advanced

Nullam sagittis convallis scelerisque. Donec dui erat, tristique nec iaculis et, hendrerit a turpis. Suspendisse velit ipsum, varius in augue a, porttitor accumsan tellus. Suspendisse erat tellus, tincidunt id ullamcorper pretium, feugiat sed quam. Nunc rutrum eros neque, vel suscipit erat tempus at. Phasellus eu hendrerit nunc, a lobortis diam. Proin a ex massa. Pellentesque quis ex lacinia nibh blandit sagittis at eget elit.

![IP Adapter Advanced](images/workflows/matteo/ipadapter_advanced_thumb.png)

## 4. IP Adapter Portrait (Style Transfer)

Duis dapibus, enim vitae elementum egestas, libero ex gravida mi, at luctus tellus mauris vel lorem. Nulla tristique consectetur arcu, at sagittis diam viverra vitae. Suspendisse potenti. Nulla id lacus fermentum felis maximus lobortis. Mauris egestas diam mi, eget interdum mauris varius eu. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Aliquam erat volutpat. Vestibulum quis ex feugiat, cursus purus eget, commodo ligula. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Vestibulum at diam sit amet tellus dignissim viverra. Nulla placerat, sem sed tincidunt lobortis, quam turpis fermentum ligula, eu tincidunt ligula dolor non felis. Pellentesque erat quam, egestas sed lacinia venenatis, pulvinar quis justo. Etiam id pharetra urna. Curabitur tristique facilisis iaculis.

![ip_adapter_portrait](ipadapter_portrait_wflow_thumb.png)

## 5. IP Adapter: Models and Transformers

### 5.1 IP Adapter with SD1.5

| IP Adapter                            | Matching Visual Transformer (ViT)              | Alias\*           |
| ------------------------------------- | ---------------------------------------------- | ----------------- |
| ip-adapter_sd15.safetensors           | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors    | STANDARD          |
| ip-adapter_sd15_light.safetensors     | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors    | LIGHT - SD15 Only |
| ip-adapter_sd15_vit-G.safetensors     | CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors | VIT-G             |
| ip-adapter-plus_sd15.safetensors      | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors    | PLUS              |
| ip-adapter-plus-face_sd15.safetensors | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors    | PLUS FACE         |

### 5.2 IP Adapter with SDXL

| IP Adapter                                  | Matching Visual Transformer (ViT)              | Alias\*   |
| ------------------------------------------- | ---------------------------------------------- | --------- |
| ip-adapter_sdxl.safetensors                 | CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors | STANDARD  |
| ip-adapter_sdxl_vit-h.safetensors           | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors    | VIT-G     |
| ip-adapter-plus_sdxl_vit-h.safetensors      | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors    | PLUS      |
| ip-adapter-plus-face_sdxl_vit-h.safetensors | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors    | PLUS FACE |

\*When using Unified Loader

## 6 FaceID: Models and Transformers

### 6.1 FaceID with SD1.5

| IP Adapter                              | Matching Visual Transformer (ViT)           | LoRA                                           | Alias\*         |
| --------------------------------------- | ------------------------------------------- | ---------------------------------------------- | --------------- |
| ip-adapter-faceid_sd15.bin              | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors | ip-adapter-faceid_sd15_lora.safetensors        | FACEID          |
| ip-adapter-faceid-plusv2_sd15.bin       | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors | ip-adapter-faceid-plusv2_sd15_lora.safetensors | FACEID PLUS V2  |
| ip-adapter-faceid-portrait-v11_sd15.bin | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors | None                                           | FACEID PORTRAIT |

### 6.2 FaceID with SDXL

| IP Adapter                          | Matching Visual Transformer (ViT)           | LoRA                                           | Alias\*         |
| ----------------------------------- | ------------------------------------------- | ---------------------------------------------- | --------------- |
| ip-adapter-faceid_sdxl.bin          | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors | ip-adapter-faceid_sdxl_lora.safetensors        | FACEID          |
| ip-adapter-faceid-plusv2_sdxl.bin   | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors | ip-adapter-faceid-plusv2_sdxl_lora.safetensors | FACEID PLUS V2  |
| ip-adapter-faceid-portrait_sdxl.bin | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors | None                                           | FACEID PORTRAIT |

## 7 Community Models

| IP Adapter                                 | Matching Visual Transformer (ViT)           | LoRA | Alias\*                            |
| ------------------------------------------ | ------------------------------------------- | ---- | ---------------------------------- |
| ip-adapter-faceid-portrait_sdxl_unnorm.bin | CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors | None | FACEID PORTRAIT UNNORM - SDXL Only |

(to be continued)
