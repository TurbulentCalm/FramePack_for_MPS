import argparse

__version__ = "1.1.0"
print(f"\n===== FramePack for MPS: demo_gradio.py v{__version__} =====")

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true', help='Share the Gradio interface via a public URL (for remote access)')
parser.add_argument('--server', type=str, default='0.0.0.0', help='Server address to run the Gradio app (default: 0.0.0.0)')
parser.add_argument('--port', type=int, required=False, help='Port to run the Gradio app (default: random or 7860)')
parser.add_argument('--inbrowser', action='store_true', help='Open the Gradio app in your web browser after launch')
parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save generated videos and images (default: ./outputs)')
parser.add_argument('--fp32', action='store_true', default=False, help='Use float32 precision for models (recommended for some M1/M2 chips)')
parser.add_argument('--debug', action='store_true', help='Enable debug mode (allows CPU fallback if MPS unavailable, extra logging)')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging (more detailed logs in terminal)')

# New: Global quality preset
parser.add_argument('--quality', choices=['high', 'medium', 'low'], help='Set global quality/speed/memory preset')

# New: Per-component precision/device/unload flags
for comp in ['tokenizer', 'text_encoder', 'vae', 'transformer']:
    parser.add_argument(f'--{comp}-precision', choices=['fp16', 'fp8', 'fp32'], help=f'Precision for {comp}')
    parser.add_argument(f'--{comp}-device', choices=['mps', 'cpu'], help=f'Device for {comp}')
    parser.add_argument(f'--{comp}-unload', action='store_true', help=f'Unload {comp} after use to save memory')

args = parser.parse_args()

# Helper: resolve effective settings for each component
QUALITY_PRESETS = {
    'high': {
        'tokenizer':  {'precision': 'fp32', 'device': 'cpu', 'unload': False},
        'vae': {'precision': 'fp16', 'device': 'mps', 'unload': False},
        'text_encoder': {'precision': 'fp16', 'device': 'mps', 'unload': False},
        'transformer': {'precision': 'fp16', 'device': 'mps', 'unload': False},
    },
    'medium': {
        'tokenizer':  {'precision': 'fp32', 'device': 'cpu', 'unload': True},
        'vae': {'precision': 'fp16', 'device': 'mps', 'unload': True},
        'text_encoder': {'precision': 'fp16', 'device': 'mps', 'unload': True},
        'transformer': {'precision': 'fp16', 'device': 'mps', 'unload': True},
    },
    'low': {
        'tokenizer':  {'precision': 'fp32', 'device': 'cpu', 'unload': True},
        'vae': {'precision': 'fp8', 'device': 'cpu', 'unload': True},
        'text_encoder': {'precision': 'fp8', 'device': 'cpu', 'unload': True},
        'transformer': {'precision': 'fp16', 'device': 'mps', 'unload': True},  # FP8 not supported on MPS
    },
}

def get_component_setting(comp):
    # Start with quality preset if set
    preset = QUALITY_PRESETS.get(args.quality, {})
    # For text_encoder_2 and tokenizer_2, always use the same settings as text_encoder and tokenizer
    if comp == 'text_encoder_2':
        comp = 'text_encoder'
    if comp == 'tokenizer_2':
        comp = 'tokenizer'
    base = preset.get(comp, {})
    # Override with explicit CLI flags if provided
    precision = getattr(args, f'{comp}_precision', None) or base.get('precision')
    device = getattr(args, f'{comp}_device', None) or base.get('device')
    unload = getattr(args, f'{comp}_unload', None)
    if unload is None:
        unload = base.get('unload', False)
    return {'precision': precision, 'device': device, 'unload': unload}

def main():
    import os
    import psutil
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is None:
        print("[INFO] For best memory usage on Apple Silicon, set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 in your shell before running this app.")
        print("       Example: export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
        print("       See README for more details.")
    import gradio as gr
    import torch
    import traceback
    import einops
    import safetensors.torch as sf
    import numpy as np
    import math
    import logging
    from PIL import Image
    from diffusers import AutoencoderKLHunyuanVideo
    from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
    from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
    from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    from diffusers_helper.memory import (
        cpu, gpu, get_available_memory_gb, move_model_to_device,
        offload_model_to_cpu, unload_complete_models, load_model_as_complete,
        fake_diffusers_current_device
    )
    from diffusers_helper.thread_utils import AsyncStream, async_run
    from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
    from diffusers_helper.bucket_tools import find_nearest_bucket

    # Print active settings at startup
    print("Active settings:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("  (Device and memory info will be shown below...)\n")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        free_mem_gb = torch.mps.recommended_max_memory() / 1024 / 1024 / 1024
    except AttributeError:
        free_mem_gb = 8  # Fallback guess if not available

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    elif args.debug:
        device = torch.device("cpu")
        logger.warning("MPS not available. Falling back to CPU (debug mode)")
    else:
        logger.error("MPS device not found. This app requires Apple Silicon with MPS support")
        raise RuntimeError("MPS device not found")

    print(f"  device: {device}")
    print(f"  free_mem_gb: {free_mem_gb:.2f}")
    print()

    # Add a global variable to store debug log messages for the UI
    DEBUG_LOGS = []

    def append_debug_log(msg):
        DEBUG_LOGS.append(msg)
        if len(DEBUG_LOGS) > 20:
            DEBUG_LOGS.pop(0)

    def log_full_memory_status(logger=None):
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        try:
            mps_used = torch.mps.driver_allocated_memory() / (1024**3)
            mps_total = torch.mps.recommended_max_memory() / (1024**3)
            mps_str = f", MPS: {mps_used:.2f}GB used / {mps_total:.2f}GB total"
        except Exception:
            mps_str = ""
        msg = (
            f"RAM: {vm.used / (1024**3):.2f}GB used / {vm.total / (1024**3):.2f}GB total "
            f"({vm.percent}% used), "
            f"Swap: {swap.used / (1024**3):.2f}GB used / {swap.total / (1024**3):.2f}GB total "
            f"({swap.percent}% used)" + mps_str
        )
        append_debug_log(msg)
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # --- Model Loader Utility ---
    def resolve_torch_dtype(precision):
        if precision == 'fp16':
            return torch.float16
        elif precision == 'fp32':
            return torch.float32
        elif precision == 'fp8':
            # PyTorch FP8 support is experimental and not available on MPS; only use on CPU
            try:
                return torch.float8_e4m3fn
            except AttributeError:
                raise RuntimeError('FP8 not supported in this PyTorch build')
        else:
            return torch.float16  # fallback

    def load_model(component, model_id, subfolder=None):
        settings = get_component_setting(component)
        dtype = resolve_torch_dtype(settings['precision'])
        device_str = settings['device']
        # Only create device_obj if needed
        if component in ['text_encoder', 'text_encoder_2', 'vae', 'transformer']:
            device_obj = torch.device(device_str or 'cpu')
        print(f"[INFO] Loading {component} from {model_id} (subfolder: {subfolder}) as {settings['precision']} on {device_str}")
        if component == 'text_encoder':
            model = LlamaModel.from_pretrained(model_id, subfolder=subfolder, torch_dtype=dtype)
            model.to(device_obj)
        elif component == 'text_encoder_2':
            model = CLIPTextModel.from_pretrained(model_id, subfolder=subfolder, torch_dtype=dtype)
            model.to(device_obj)
        elif component == 'vae':
            model = AutoencoderKLHunyuanVideo.from_pretrained(model_id, subfolder=subfolder, torch_dtype=dtype)
            model.to(device_obj)
        elif component == 'transformer':
            model = HunyuanVideoTransformer3DModelPacked.from_pretrained(model_id, torch_dtype=dtype)
            model.to(device_obj)
        elif component == 'tokenizer':
            model = LlamaTokenizerFast.from_pretrained(model_id, subfolder=subfolder)
        elif component == 'tokenizer_2':
            model = CLIPTokenizer.from_pretrained(model_id, subfolder=subfolder)
        else:
            raise ValueError(f"Unknown component: {component}")
        log_full_memory_status(logger)
        return model

    def unload_model(model, component):
        print(f"[INFO] Unloading {component} from memory")
        del model
        torch.mps.empty_cache()
        log_full_memory_status(logger)

    # --- Pipeline Refactor: Lazy Load/Unload ---
    # Tokenizers (always on CPU, usually fp32)
    tokenizer = load_model('tokenizer', "hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = load_model('tokenizer_2', "hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')

    def get_text_encoders():
        text_encoder = load_model('text_encoder', "hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder')
        text_encoder.eval()
        text_encoder_2 = load_model('text_encoder_2', "hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2')
        text_encoder_2.eval()
        return text_encoder, text_encoder_2

    def get_vae():
        vae = load_model('vae', "hunyuanvideo-community/HunyuanVideo", subfolder='vae')
        vae.eval()
        # Enable VAE slicing/tiling if available
        if hasattr(vae, 'enable_slicing'):
            vae.enable_slicing()
        if hasattr(vae, 'enable_tiling'):
            vae.enable_tiling()
        return vae

    def get_transformer():
        transformer = load_model('transformer', 'lllyasviel/FramePackI2V_HY')
        transformer.eval()
        if hasattr(transformer, 'enable_attention_slicing'):
            transformer.enable_attention_slicing()
        return transformer

    stream = AsyncStream()

    outputs_folder = args.output_dir
    os.makedirs(outputs_folder, exist_ok=True)

    def get_debug_log():
        return '\n'.join(DEBUG_LOGS)

    @torch.no_grad()
    def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache, mp4_crf, resolution):
        log_full_memory_status(logger)
        total_latent_sections = (total_second_length * 24) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        job_id = generate_timestamp()

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

        try:
            # --- Text Encoding ---
            text_encoder, text_encoder_2 = get_text_encoders()
            fake_diffusers_current_device(text_encoder, torch.device(get_component_setting('text_encoder')['device']))
            fake_diffusers_current_device(text_encoder_2, torch.device(get_component_setting('text_encoder')['device']))
            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            if get_component_setting('text_encoder')['unload']:
                unload_model(text_encoder, 'text_encoder')
            if get_component_setting('text_encoder_2')['unload']:
                unload_model(text_encoder_2, 'text_encoder_2')
            log_full_memory_status(logger)

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                # Reload text encoders for negative prompt if needed
                text_encoder, text_encoder_2 = get_text_encoders()
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                if get_component_setting('text_encoder')['unload']:
                    unload_model(text_encoder, 'text_encoder')
                if get_component_setting('text_encoder_2')['unload']:
                    unload_model(text_encoder_2, 'text_encoder_2')

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=resolution)
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

            Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

            vae = get_vae()
            start_latent = vae_encode(input_image_pt, vae)
            if get_component_setting('vae')['unload']:
                unload_model(vae, 'vae')
            log_full_memory_status(logger)

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

            # Prepare for transformer
            transformer = get_transformer()
            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
            log_full_memory_status(logger)
            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3

            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            latent_paddings = reversed(range(total_latent_sections))

            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            for latent_padding in latent_paddings:
                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size
                log_full_memory_status(logger)

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    return

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                if use_teacache:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    transformer.initialize_teacache(enable_teacache=False)

                def callback(d):
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)

                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        raise KeyboardInterrupt('User ends the task.')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling {current_step}/{steps}'
                    desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 24) :.2f} seconds (FPS-24). The video is being extended now ...'
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                    return

                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=torch.device(get_component_setting('transformer')['device']),
                    dtype=transformer.dtype,
                    image_embeddings=None,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                if history_pixels is None:
                    vae = get_vae()
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                    if get_component_setting('vae')['unload']:
                        unload_model(vae, 'vae')
                    log_full_memory_status(logger)
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    vae = get_vae()
                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    if get_component_setting('vae')['unload']:
                        unload_model(vae, 'vae')
                    log_full_memory_status(logger)
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

                save_bcthw_as_mp4(history_pixels, output_filename, fps=24, crf=mp4_crf)

                print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

                stream.output_queue.push(('file', output_filename))

                if is_last_section:
                    break
            if get_component_setting('transformer')['unload']:
                unload_model(transformer, 'transformer')
            log_full_memory_status(logger)
        except:
            traceback.print_exc()

        stream.output_queue.push(('end', None))
        return

    def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache, mp4_crf, resolution):
        global stream
        assert input_image is not None, 'No input image!'

        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

        stream = AsyncStream()

        async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache, mp4_crf, resolution)

        output_filename = None

        while True:
            flag, data = stream.output_queue.next()

            if flag == 'file':
                output_filename = data
                yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

            if flag == 'progress':
                preview, desc, html = data
                yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

            if flag == 'end':
                yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
                break

    def end_process():
        stream.input_queue.push('end')

    quick_prompts = [
        'The girl dances gracefully, with clear movements, full of charm.',
        'A character doing some simple body movements.',
    ]
    quick_prompts = [[x] for x in quick_prompts]

    css = make_progress_bar_css()
    block = gr.Blocks(css=css).queue()
    with block:
        gr.Markdown('# FramePack')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
                resolution = gr.Slider(label="Resolution", minimum=240, maximum=720, value=416, step=16)
                prompt = gr.Textbox(label="Prompt", value='')
                example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

                with gr.Row():
                    start_button = gr.Button(value="Start Generation")
                    end_button = gr.Button(value="End Generation", interactive=False)
                    spinner = gr.HTML("<div id='spinner' style='display:none'><svg width='24' height='24' viewBox='0 0 24 24'><circle cx='12' cy='12' r='10' stroke='gray' stroke-width='4' fill='none' stroke-dasharray='60' stroke-dashoffset='0'><animateTransform attributeName='transform' type='rotate' from='0 12 12' to='360 12 12' dur='1s' repeatCount='indefinite'/></circle></svg></div>")

                with gr.Group():
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                    n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                    seed = gr.Number(label="Seed", value=31337, precision=0)

                    total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                    latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                    rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)

                    mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")

            with gr.Column():
                preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
                gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
                gr.Markdown('**Tip:** If you encounter memory errors or crashes, try lowering the Resolution, Total Video Length, or Latent Window Size sliders. This will reduce memory usage and help the app run on Macs with less RAM.')
                progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                progress_bar = gr.HTML('', elem_classes='no-generating-animation')
                debug_panel = gr.Textbox(label="Debug/Status Log", value="", lines=8, interactive=False)

        gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

        ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache, mp4_crf, resolution]
        outs = [result_video, preview_image, progress_desc, progress_bar, start_button, end_button, debug_panel]
        def process_with_debug(*args):
            import time
            spinner_html = "<div id='spinner' style='display:block'><svg width='24' height='24' viewBox='0 0 24 24'><circle cx='12' cy='12' r='10' stroke='gray' stroke-width='4' fill='none' stroke-dasharray='60' stroke-dashoffset='0'><animateTransform attributeName='transform' type='rotate' from='0 12 12' to='360 12 12' dur='1s' repeatCount='indefinite'/></circle></svg></div>"
            yield [None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), get_debug_log()]
            for result in process(*args):
                debug_log = get_debug_log()
                if result[0] is not None:
                    spinner_html = "<div id='spinner' style='display:none'></div>"
                yield list(result) + [debug_log]

        start_button.click(fn=process_with_debug, inputs=ips, outputs=outs)
        end_button.click(fn=end_process)

    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        allowed_paths=[outputs_folder],
    )

if __name__ == "__main__":
    import sys
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        main()
