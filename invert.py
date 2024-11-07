from inversion.ddim_inversion import DDIMInversionModel
from inversion.ddpm_inversion import DDPMInversionModel
from PIL import Image
import numpy as np
import argparse

def _main(args):
    num_steps = args.num_steps
    inversion_steps = args.inversion_steps
    assert inversion_steps <= num_steps, "inversion_steps should be less than or equal to num_steps"

    if args.model == 'ddpm':
        inversion_model = DDPMInversionModel(num_steps=num_steps, inversion_steps=inversion_steps)
    else:
        inversion_model = DDIMInversionModel(num_steps=num_steps, inversion_steps=inversion_steps)

    with open(args.img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")        
        img = np.array(img)
        inverted_img = inversion_model.invert(img, inversion_steps)
        inverted_img = Image.fromarray(inverted_img)
        inverted_img.save(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='figures/test.png')
    parser.add_argument('--output_path', type=str, default='figures/output.png')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--inversion_steps', type=int, default=15)
    parser.add_argument('--model', type=str, default='ddpm', choices=['ddpm', 'ddim'])
    args = parser.parse_args()
    _main(args)
    




