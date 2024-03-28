import torch
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm
import os
import sys

if not os.path.exists('./sample-9/'):
    os.mkdir('./sample-9/')

scheduler = DDPMScheduler.from_pretrained("ddpm-eyediffusion-9/scheduler")
model = UNet2DModel.from_pretrained("ddpm-eyediffusion-9/unet")
scheduler.set_timesteps(1000)

n_classes = 9

sample_size = model.config.sample_size
noise = torch.randn((1, 1, sample_size, sample_size)).repeat(n_classes, 1, 1, 1)
classes = torch.arange(0, n_classes).reshape(n_classes,).int()

if sys.argv[1] == 'cuda':
    model = model.to("cuda")
    noise = noise.cuda()
    classes = classes.cuda()
    
input = noise

for t in tqdm(scheduler.timesteps):
    with torch.no_grad():
        timesteps = t.repeat(n_classes,)
        if sys.argv[1] == 'cuda':
            timesteps = timesteps.cuda()
        noisy_residual = model(input, timesteps, classes).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

images = (input / 2.0 + 0.5).clamp(0, 1)
for i in range(images.shape[0]):
    image = images[i][0].cpu().numpy()
    image = Image.fromarray((image * 255).round().astype("uint8"))
    image.save('./sample/' + str(i) + '.png')



