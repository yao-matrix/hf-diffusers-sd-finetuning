from diffusers import StableDiffusionPipeline
import torch

model_id = "~/textual_inversion_cat"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cpu")

prompt = [
         "a lovely <dicoo> in red dress and hat, in the snowly and brightly night, with many brighly buildings."
        ]

seed = 333
num_images = 5

names = [
        "dicoo"
        ]

flag = "hf_4node_16ddp_1batch_1accum_ccl"
prefix = "inversion_"

for p in prompt:
    generator = torch.Generator("cpu").manual_seed(seed)
    idx = 0
    for i in range(num_images):
        image = pipe(prompt, generator=generator, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(prefix+names[idx]+flag+str(i)+".png")
    idx += 1

