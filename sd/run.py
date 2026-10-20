import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer

DEVICE = "cpu"

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
models = model_loader.preload_models_from_standard_weights(
    "../data/v1-5-pruned-emaonly.ckpt", DEVICE
)

output = pipeline.generate(
    prompt="A serene misty pine forest where the trees have glowing fiber-optic branches, magical fireflies, soft morning sunbeams piercing through the fog, surrealism, ethereal beauty, highly detailed",
    uncond_prompt="",
    input_image=None,          # or Image.open("../images/dog.jpg") for img2img
    strength=0.9,
    do_cfg=True,
    cfg_scale=8,
    sampler_name="ddpm",
    n_inference_steps=20,      # 50 is the default but very slow on CPU
    seed=42,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

Image.fromarray(output).save("output.png")
print("Saved output.png")