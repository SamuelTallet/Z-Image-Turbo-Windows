"""ZPix Gradio app."""
# Based on https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo

import logging
from argparse import ArgumentParser
from json import load as load_json
from os import environ
from pathlib import Path
from random import randint
from re import search
from shutil import rmtree

import gradio as gr
from diffusers import ZImagePipeline
from platformdirs import user_pictures_path
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from torch import bfloat16, cuda, manual_seed, xpu

from source.py.disclaimer import TERMS_OF_USE, TermsOfUse
from source.py.gen_history import (
    add_prompt_to_history,
    get_prompts_history,
    on_prompts_history_row_select,
)
from source.py.generation import Generation
from source.py.lora_model import LoraModel
from source.py.os_abstract import open_with_default_app

logging.basicConfig(format="%(levelname)s: %(message)s")

# Path to Triton cache directory
# shortened by good measure to avoid too long path errors on Windows
# even if this has been fixed recently.
environ["TRITON_CACHE_DIR"] = str(Path.home() / ".triton")

app_dir = Path(__file__).parent
"""App directory."""

# We store temp files created by this app in a local directory
# so we can remove them later without impacting other Gradio apps.
environ["GRADIO_TEMP_DIR"] = str(app_dir / "temp" / "GradioApp")

assets_dir = app_dir / "assets"
"""Assets directory."""

# Let's serve assets directly.
gr.set_static_paths(paths=[assets_dir])

output_dir: Path
"""The folder where ZPix saves generated images, prompts, etc."""

try:
    output_dir = user_pictures_path() / "ZPix"
except Exception:
    logging.warning("Can't get user pictures path, using default.")
    output_dir = Path.home() / "Pictures" / "ZPix"

translation: dict[str, str] = {}
"""Translation."""

metadata: dict[str, str] = {}
"""Metadata."""

pipe: ZImagePipeline | None = None
"""Pipeline."""

optimized: bool = False
"""Pipeline is optimized?"""

pipe_is_busy: bool = False
"""Pipeline is busy? e.g. loading a LoRA."""


def load_translation(locale: str) -> None:
    """Load translation for a given locale, if available."""
    global translation

    translation_file = app_dir / "translations" / f"{locale}.json"
    if not translation_file.exists():
        logging.warning(f"Translation for {locale} not found.")
        return

    with open(translation_file, "r", encoding="utf-8") as file:
        translation = load_json(file)


def t(string: str) -> str:
    """Translate a string."""
    return translation.get(string, string)


def get_metadata(filename: str) -> str:
    """Get metadata."""
    if filename not in metadata:
        file = app_dir / "metadata" / filename
        metadata[filename] = file.read_text()

    return metadata[filename]


def get_example_prompts() -> list[str]:
    """Get example prompts."""
    prompts_file = app_dir / "examples" / "prompts.json"

    with open(prompts_file, "r", encoding="utf-8") as file:
        prompts = load_json(file)

    return [prompt["text"] for prompt in prompts]


def get_theme():
    """Get customized theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f7f6ff",
            c100="#efedff",
            c200="#d8d2ff",
            c300="#c0b7ff",
            c400="#a192ff",
            c500="#624aff",
            c600="#5843e6",
            c700="#4534b3",
            c800="#312580",
            c900="#1d164d",
            c950="#0a071a",
        )
    )


def on_app_load():
    """On app load."""
    if not optimized:
        gr.Warning(
            t(
                "Image generation may be slow because diffusion pipeline is not optimized."
            )
            + "<br>"
            + t(
                "Try upgrading your graphics card drivers, then reboot your PC and restart"
            )
            + f" {get_metadata('NAME')}.",
            duration=None,  # Until user closes it.
        )


def get_aspects_and_resolutions() -> tuple:
    """Get aspect ratios and resolutions,
    possibly translated.

    Returns:
        Tuple of (
            resolutions by aspect,
            default resolution choices,
            aspect ratio choices,
            default aspect ratio
        )
    """
    default_aspect_ratio = "16:9"

    resolutions_by_aspect = {
        "1:1": [
            "1024x1024",
            "1280x1280",
            "1440x1440",
        ],
        "16:9": [
            "1280x720",
            "1920x1088",
        ],
        "9:16": [
            "720x1280",
            "1088x1920",
        ],
        "4:3": [
            "1152x864",
            "1440x1088",
            "1920x1440",
        ],
        "3:4": [
            "864x1152",
            "1088x1440",
            "1440x1920",
        ],
        "16:10": [
            "1280x800",
            "1440x912",
            "1920x1200",
        ],
        "10:16": [
            "800x1280",
            "912x1440",
            "1200x1920",
        ],
        "21:9": [
            "1344x576",
        ],
    }

    default_resolution_choices = resolutions_by_aspect[default_aspect_ratio]
    aspect_ratio_choices = list(resolutions_by_aspect.keys())

    return (
        resolutions_by_aspect,
        default_resolution_choices,
        aspect_ratio_choices,
        default_aspect_ratio,
    )


def parse_resolution(resolution):
    """Parse resolution string into width and height.

    Args:
        resolution: Resolution string in format "WIDTHxHEIGHT" or "WIDTH×HEIGHT".

    Returns:
        Tuple of (width, height) as integers. Defaults to (1024, 1024) if parsing fails.
    """
    match = search(r"(\d+)\s*[×x]\s*(\d+)", resolution)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1024, 1024


def update_trigger_word(trigger_words: list, prompt: str) -> str:
    """Update the trigger word in the prompt.

    Args:
        trigger_words: List of [previous, current] trigger words.
        prompt: The current prompt as a string.

    Returns:
        Updated prompt.
    """
    previous_tw, current_tw = trigger_words

    # Removes the previous trigger word from start of the prompt.
    if previous_tw and prompt.startswith(previous_tw):
        prompt = prompt[len(previous_tw) :].lstrip()

    # Adds the current trigger word to start of the prompt.
    if current_tw:
        prompt = f"{current_tw} {prompt}"

    return prompt


def remove_trigger_word(trigger_words: list, prompt: str) -> tuple:
    """Remove the current trigger word from the prompt.

    Args:
        trigger_words: List of [previous, current] trigger words.
        prompt: The current prompt as a string.

    Returns:
        Tuple of (empty trigger words list, updated prompt).
    """
    _, current_tw = trigger_words

    # Removes the current trigger word from start of the prompt.
    if current_tw and prompt.startswith(current_tw):
        prompt = prompt[len(current_tw) :].lstrip()

    return [None, None], prompt


def load_model(model: str, backup_model: str):
    """Load and configure the Z-Image pipeline.

    Args:
        model: Hugging Face (HF) model name.
        backup_model: HF backup model name.
    """
    global pipe
    global optimized

    try:
        pipe = ZImagePipeline.from_pretrained(
            model,
            torch_dtype=bfloat16,
        )
    except Exception:
        logging.warning(f"Can't load {model}, falling back to {backup_model}.")
        pipe = ZImagePipeline.from_pretrained(
            backup_model,
            torch_dtype=bfloat16,
        )

    # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
    if triton_is_available and (cuda.is_available() or xpu.is_available()):
        pipe.transformer = apply_sdnq_options_to_model(
            pipe.transformer, use_quantized_matmul=True
        )
        pipe.text_encoder = apply_sdnq_options_to_model(
            pipe.text_encoder, use_quantized_matmul=True
        )
        try:
            pipe.transformer.set_attention_backend("_sage_qk_int8_pv_fp16_triton")
            optimized = True
        except Exception as e:
            logging.warning(f"SageAttention is not available: {e}")

    pipe.enable_model_cpu_offload()


def swap_lora(path: str) -> str | None:
    """Swap or load a new LoRA model.

    Args:
        path: Path to a LoRA file.

    Returns:
        Trigger word of LoRA model.
    """
    global pipe_is_busy

    if pipe_is_busy:
        raise gr.Error(
            t("Pipeline is busy. Please try again shortly."),
            duration=4,
        )

    if not path.endswith(".safetensors"):
        raise gr.Error(
            t("LoRA file extension must be .safetensors"),
            duration=20,
        )

    lora = LoraModel(path)

    try:
        if lora.base_model() != "zimage":
            gr.Warning(
                f"{t('This LoRA seems incompatible with')} Z-Image.<br>"
                f"{t('It might not work.')}",
                duration=5,
            )
    except Exception as e:
        logging.warning(f"Can't check LoRA compatibility: {e}")

    bfloat16_lora = lora.to_bf16()
    gr.Info(t("Loading LoRA..."), duration=2)

    try:
        pipe_is_busy = True
        pipe.unload_lora_weights()
        pipe.load_lora_weights(bfloat16_lora, adapter_name="lora_1")
    finally:
        pipe_is_busy = False

    trigger_word = lora.trigger_word()

    return trigger_word


def set_lora_strength(strength: float):
    """Set LoRA strength."""
    adapters = pipe.get_list_adapters()

    if "transformer" not in adapters or "lora_1" not in adapters["transformer"]:
        raise gr.Error("No LoRA loaded.")

    pipe.set_adapters("lora_1", strength)


def unload_lora():
    """Unload LoRA model."""
    global pipe_is_busy

    if pipe_is_busy:
        raise gr.Error(
            t("Pipeline is busy. Please try again shortly."),
            duration=4,
        )

    try:
        pipe_is_busy = True
        pipe.unload_lora_weights()
    finally:
        pipe_is_busy = False


def generate_image(
    pipe,
    prompt,
    resolution="1024x1024",
    seed=42,
    num_inference_steps=8,
):
    """Generate an image using the Z-Image pipeline.

    Args:
        pipe: The loaded ZImagePipeline instance.
        prompt: Text prompt describing the desired image.
        resolution: Output resolution as "WIDTHxHEIGHT" string.
        seed: Random seed for reproducible generation.
        num_inference_steps: Number of denoising steps.

    Returns:
        Generated PIL Image.
    """
    global pipe_is_busy
    width, height = parse_resolution(resolution)

    try:
        pipe_is_busy = True
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=manual_seed(seed),
        ).images[0]
    finally:
        pipe_is_busy = False

    return image


def generate(
    prompt,
    resolution="1024x1024",
    seed=42,
    random_seed=True,
    steps=8,
    gallery_images=None,
):
    """Generate an image and possibly a seed, and update gallery.

    Args:
        prompt: Text prompt for image generation.
        resolution: Resolution string (e.g. "1024x1024").
        seed: Seed value for reproducibility.
        random_seed: Ignore seed argument and generate a seed?
        steps: Number of inference (denoising) steps.
        gallery_images: Existing gallery images to append to.

    Returns:
        Tuple of (updated gallery, last image index, used seed).

    Raises:
        gr.Error: If the pipeline is not loaded or busy.
    """
    if pipe is None:
        raise gr.Error("Pipeline not loaded.")

    if pipe_is_busy:
        raise gr.Error(
            t("Pipeline is busy. Please try again shortly."),
            duration=4,
        )

    if random_seed:
        used_seed = randint(1, 1000000)
    else:
        used_seed = int(seed) if seed != -1 else randint(1, 1000000)

    generation_args = {
        "pipe": pipe,
        "prompt": prompt,
        "resolution": resolution,
        "seed": used_seed,
        "num_inference_steps": int(steps + 1),
    }
    try:
        image = generate_image(**generation_args)
    except UnicodeDecodeError:
        # A corrupted Triton cache can cause an UnicodeDecodeError.
        rmtree(Path.home() / ".triton", ignore_errors=True)
        gr.Warning(t("Cleared Triton cache as it may be corrupted."), duration=6)

        gr.Info(t("Regenerating same image..."), duration=8)
        image = generate_image(**generation_args)

    generation = Generation(
        model="Z-Image Turbo",  # TODO: Make this dynamic.
        image=image,
        prompt=prompt,
        resolution=resolution,
        seed=used_seed,
        steps=int(steps),
    )
    image_file, _prompt_file, _settings_file = generation.save(output_dir)

    if gallery_images is None:
        gallery_images = []

    # Prompt is added as image caption.
    gallery_images.append((image_file, prompt))

    return gallery_images, len(gallery_images) - 1, used_seed


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backup-model", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--locale", type=str, required=False, default="en-US")
    args, _ = parser.parse_known_args()

    load_model(args.model, args.backup_model)

    if args.locale != "en-US":
        load_translation(args.locale)

    tou = TermsOfUse(app_dir / ".tou_accepted")

    (
        resolutions_by_aspect,
        default_resolution_choices,
        aspect_ratio_choices,
        default_aspect_ratio,
    ) = get_aspects_and_resolutions()

    with gr.Blocks(
        fill_width=True,
        analytics_enabled=False,
    ) as app:
        with gr.Row(elem_classes=[] if tou.accepted() else ["blurred"]) as ui_row:
            with gr.Column(min_width=48, elem_classes=["sidebar"]):
                gr.Button(
                    "",
                    icon=assets_dir / "noto-emoji" / "emoji_u26a1.svg",
                    link=get_metadata("HOME_URL"),
                    link_target="_blank",  # Opens default browser. See app.js
                    elem_id="home-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("home-btn")
                        btn.title = "{t("Visit project homepage to check updates")}"
                    """
                )
                gr.Button(
                    "",
                    icon=assets_dir / "lora_grad.svg",
                    elem_id="swap-lora-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("swap-lora-btn")
                        btn.title = "{t("Load a LoRA file to apply a new style")}"
                    """
                )
                lora_path = gr.Textbox(
                    visible="hidden",  # See "portal" in app.js
                    elem_id="lora-path",
                )
                show_seed_btn = gr.Button(
                    "",
                    icon=assets_dir / "juicy-fish" / "dice.png",
                    elem_id="show-seed-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("show-seed-btn")
                        btn.title = "{t("Use a specific or random seed")}"
                    """
                )
                gr.Button(
                    "",
                    icon=assets_dir / "kerismaker" / "tech_13631866.png",
                    link=f"{get_metadata('HOME_URL')}/blob/main/docs/FAQ.md",
                    link_target="_blank",
                    elem_id="faq-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("faq-btn")
                        btn.title = "{t("Access the FAQ of this application")}"
                    """
                )
                gr.Button(
                    "",
                    icon=assets_dir / "kofi_symbol.svg",
                    link=get_metadata("DONATE_URL"),
                    link_target="_blank",
                    elem_id="donate-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("donate-btn")
                        btn.title = "{t("Keep project developer awake with a coffee")} 😄"
                    """
                )

            with gr.Column():
                trigger_words = gr.State(value=[None, None])
                """Trigger words (previous, current)."""

                with gr.Row():
                    prompt = gr.Textbox(
                        label=t("Prompt"),
                        lines=3,
                        placeholder=t("Enter your prompt here..."),
                        html_attributes=gr.InputHTMLAttributes(spellcheck=False),
                    )

                with gr.Row():
                    aspect_ratio = gr.Dropdown(
                        value=default_aspect_ratio,
                        choices=aspect_ratio_choices,
                        container=False,
                        elem_id="aspect-ratio",
                    )
                    gr.HTML(
                        visible="hidden",
                        js_on_load=f"""
                            let select = document.getElementById("aspect-ratio")
                            select.title = "{t("Aspect Ratio")}"
                        """,
                    )
                    resolution = gr.Dropdown(
                        value=default_resolution_choices[0],
                        choices=default_resolution_choices,
                        container=False,
                        elem_id="resolution",
                    )
                    gr.HTML(
                        visible="hidden",
                        js_on_load=f"""
                            let select = document.getElementById("resolution")
                            select.title = "{t("Resolution")}"
                        """,
                    )
                    generate_btn = gr.Button(
                        t("Generate Image"),
                        variant="primary",
                    )

                with gr.Row(visible=False) as lora_row:
                    lora_strength = gr.Slider(
                        scale=2,
                        label=t("LoRA Strength"),
                        minimum=-2.5,
                        maximum=2.5,
                        step=0.1,
                        value=1.0,
                    )
                    lora_strength.change(
                        set_lora_strength,
                        inputs=lora_strength,
                    )
                    unload_lora_btn = gr.Button(t("Unload LoRA"))

                    # On "Unload LoRA" button click:
                    # - unload LoRA model,
                    # - remove trigger word from prompt,
                    # - empty trigger words history,
                    # - make LoRA row invisible.
                    unload_lora_btn.click(
                        unload_lora,
                    ).then(
                        remove_trigger_word,
                        inputs=[trigger_words, prompt],
                        outputs=[trigger_words, prompt],
                    ).then(
                        lambda: gr.update(visible=False),
                        outputs=lora_row,
                    )

                # When a LoRA path is selected:
                # - discard appended timestamp,
                # - unload any LoRA model,
                # - load selected LoRA model,
                # - shift trigger words history,
                # - update trigger word in prompt,
                # - make LoRA row visible.
                lora_path.change(
                    lambda p, tw: [tw[1], swap_lora(p)],
                    inputs=[lora_path, trigger_words],
                    outputs=trigger_words,
                    js="(p, tw) => [p.split('|')[0], tw]",
                ).then(
                    update_trigger_word,
                    inputs=[trigger_words, prompt],
                    outputs=prompt,
                ).then(
                    lambda: gr.update(visible=True),
                    outputs=lora_row,
                )

                with gr.Row(visible=False) as seed_row:
                    seed = gr.Number(label=t("Seed"), value=42, precision=0)
                    random_seed = gr.Checkbox(label=t("Random"), value=True)

                show_seed_state = gr.State(value=False)

                def toggle_seed_row(visibility):
                    visibility = not visibility
                    return visibility, gr.update(visible=visibility)

                show_seed_btn.click(
                    toggle_seed_row,
                    inputs=show_seed_state,
                    outputs=[show_seed_state, seed_row],
                )

                with gr.Row(visible=False) as steps_row:
                    steps = gr.Slider(
                        label=t("Denoising Steps"),
                        minimum=4,
                        maximum=9,
                        value=8,
                        step=1,
                    )

                prompts_history = get_prompts_history(
                    output_dir,
                    max_prompts=100,
                )
                prompts_history_frame = gr.DataFrame(
                    visible=bool(prompts_history),
                    label=t("Previous Prompts"),
                    value=prompts_history,
                    type="array",
                    interactive=False,
                    max_height=200,
                    show_search="search",
                    buttons=None,
                    elem_id="prompts-history",
                )
                prompts_history_frame.select(
                    on_prompts_history_row_select,
                    outputs=prompt,
                    show_progress="hidden",
                )
                gr.HTML(
                    js_on_load=f"""
                        let input = document.querySelector("#prompts-history .search-input")
                        input.placeholder = "{t("Search...")}"
                        input.title = "{t("among last 100 prompts")}"
                        input.spellcheck = false
                    """
                )

                gr.Examples(
                    visible=not prompts_history,  # Onboarding-like.
                    examples=get_example_prompts(),
                    inputs=prompt,
                    label=t("Example Prompts"),
                )

            with gr.Column(scale=2):
                gallery_images = gr.Gallery(
                    label=t("Generated Images"),
                    columns=3,
                    rows=2,
                    height=600,
                    object_fit="contain",
                    format="png",
                    type="filepath",
                    buttons=["fullscreen"],
                    interactive=False,
                )
                last_image_index = gr.State(value=None)

                open_output_folder_btn = gr.Button(
                    t("Open Output Folder"),
                    variant="primary",
                    elem_classes=["align-right"],
                )

                def create_open_output_dir():
                    output_dir.mkdir(parents=True, exist_ok=True)
                    open_with_default_app(output_dir)

                open_output_folder_btn.click(create_open_output_dir)

        with gr.Row():
            # Add source model link to footer, after Gradio credit.
            gr.HTML(
                js_on_load=f"""
                    document.querySelector("footer").insertAdjacentHTML(
                        "beforeend",
                        `<a
                            href="https://huggingface.co/{args.model}"
                            target="_blank"
                        >
                            {t("Source model")} 🤗
                        </a>`
                    )
                """
            )

        with gr.Row(
            visible=not tou.accepted(),
            elem_id="tou-row",
        ) as tou_row:
            with gr.Column(elem_id="tou-card"):
                gr.Markdown(f"### {t('Terms of Use')}")
                gr.Markdown(t(TERMS_OF_USE))
                agree_tou_btn = gr.Button(t("I agree"), variant="primary")

                agree_tou_btn.click(tou.accept).then(
                    lambda: (gr.update(visible=False), gr.update(elem_classes=[])),
                    outputs=[tou_row, ui_row],
                )

        def update_resolution_choices(_aspect_ratio):
            resolution_choices = resolutions_by_aspect.get(
                _aspect_ratio, default_resolution_choices
            )
            return gr.update(value=resolution_choices[0], choices=resolution_choices)

        aspect_ratio.change(
            update_resolution_choices,
            inputs=aspect_ratio,
            outputs=resolution,
            show_progress="hidden",
        )

        generate_btn.click(
            generate,
            inputs=[prompt, resolution, seed, random_seed, steps, gallery_images],
            outputs=[gallery_images, last_image_index, seed],
            show_progress_on=gallery_images,
        ).then(
            # Select generated image in gallery:
            lambda idx: gr.update(selected_index=idx),
            inputs=last_image_index,
            outputs=gallery_images,
        ).then(
            add_prompt_to_history,
            inputs=[prompt, prompts_history_frame],
            outputs=[prompts_history_frame],
        )

        app.load(on_app_load)

    app.launch(
        server_port=args.port,
        footer_links=["gradio"],  # Credit
        theme=get_theme(),
        css_paths=[app_dir / "source" / "app.css"],
        js=(app_dir / "source" / "app.js").read_text(),
        allowed_paths=[output_dir],
    )
