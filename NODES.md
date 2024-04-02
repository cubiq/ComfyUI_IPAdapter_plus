# Nodes reference

Below I'm trying to document all the nodes. It's still very incomplete, be sure to check back later.

## Loaders

### :knot: IPAdapter Unified Loader

Loads the full stack of models needed for IPAdapter to function. The returned object will contain information regarding the **ipadapter** and **clip vision models**.

Multiple unified loaders should always be daisy chained through the `ipadapter` in/out. **Failing to do so will cause all models to be loaded twice.** For **the first** unified loader the `ipadapter` input **should never be connected**.

#### Inputs
- **model**, main ComfyUI model pipeline

#### Optional Inputs
- **ipadapter**, it's important to note that this is optional and used exclusively to daisy chain unified loaders. **The `ipadapter` input is never connected in the first `IPAdapter Unified Loader` of the chain.**

#### Outputs
- **model**, the model pipeline is used exclusively for configuration, the model comes out of this node untouched and it can be considered a reroute. Note that this is different  from the Unified Loader FaceID that actually alters the model with a LoRA.
- **ipadapter**, connect this to any ipadater node. Each node will automatically detect if the `ipadapter` object contains the full stack of models or just one (like in the case [IPAdapter Model Loader](#ipadapter-model-loader)).

### :knot: IPAdapter Model Loader

Loads the IPAdapter model only. The returned object will be the IPAdapter model contrary to the [Unified loader](#ipadapter-unified-loader) that contains the full stack of models.

#### Configuration parameters
- **ipadapter_file**, the main IPAdapter model. It must be located into `ComfyUI/models/ipadapter` or in any path specified in the `extra_model_paths.yaml` configuration file.

#### Outputs
- **IPADAPTER**, contains the loaded model only. Note that `IPADAPTER` will have a different structure when loaded by the [Unified Loader](#ipadapter-unified-loader).

## Main IPAdapter Apply Nodes

### :knot: IPAdapter Advanced

This node contains all the options to fine tune the IPAdapter models. It is a drop in replacement for the old `IPAdapter Apply` that is no longer available. If you have an old workflow, delete the existing `IPadapter Apply` node, add `IPAdapter Advanced` and connect all the pipes as before.

#### Inputs
- **model**, main model pipeline.
- **ipadapter**, the IPAdapter model. It can be connected to the [IPAdapter Model Loader](#ipadapter-model-loader) or any of the Unified Loaders. If a Unified loader is used anywhere in the workflow and you don't need a different model, it's always adviced to reuse the previous `ipadapter` pipeline.
- **image**, the reference image used to generate the positive conditioning. It should be a square image, other aspect ratios are automatically cropped in the center.

#### Optional inputs
- **image_negative**, image used to generate the negative conditioning. This is optional and normally handled by the code. It is possible to send noise or actually any image to instruct the model about what we don't want to see in the composition.
- **attn_mask**, a mask that will be applied during the image generation. **The mask should have the same size or at least the same aspect ratio of the latent**. The mask will define the area of influence of the IPAdapter models on the final image. Black zones won't be affected, white zones will get maximum influence. It can be a grayscale mask.
- **clip_vision**, this is optional if using any of the Unified loaders. If using the [IPAdapter Model Loader](#knot-ipadapter-model-loader) you also have to provide the clip vision model with a `Load CLIP Vision` node.

#### Configuration parameters
- **weight**, weight of the IPAdapter model. For `linear` `weight_type` (the default), a good starting point is 0.8. If you use other weight types you can experiment with higher values.
- **weight_type**, this is how the IPAdapter is applied to the UNet block. For example `ease-in` means that the input blocks have higher weight than the output ones. `week input` means that the whole input block has lower weight. `style transfer (SDXL)` only works with SDXL and it's a very powerful tool to tranfer only the style of an image but not its content. This parameter hugely impacts how the composition reacts to the text prompting.
- **combine_embeds**, when sending more than one reference image the embeddings can be sent one after the other (`concat`) or combined in various ways. For low spec GPUs it is adviced to `average` the embeds if you send multiple images. `subtract` subtracts the embeddings of the second image to the first; in case of 3 or more images they are averaged and subtracted to the first.
- **start_at/end_at**, this is the timestepping. Defines at what percentage point of the generation to start applying the IPAdapter model. The initial steps are the most important so if you start later (eg: `start_at=0.3`) the generated image will have a very light conditioning.
- **embeds_scaling**, the way the IPAdapter models are applied to the K,V. This parameter has a small impact on how the model reacts to text prompting. `K+mean(V) w/ C penalty` grants good quality at high weights (>1.0) without burning the image.
