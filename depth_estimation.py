import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils/DPT'))

import argparse
import torch
#from utils.DPT.run_monodepth import run as dpt_depth
from utils.ZoeDepth.zoedepth.utils.misc import get_image_from_url, colorize
from utils.DPT.run_monodepth import depth_estimation as dpt_depth_estimation
from PIL import Image
import matplotlib.pyplot as plt




def depth_estimation(input_path, output_path, model):

    if model == "zoe":
        zoe = torch.hub.load("./utils/ZoeDepth/", "ZoeD_N", source="local", pretrained=True)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        zoe = zoe.to(DEVICE)

        img = Image.open(input_path).convert("RGB")
        depth = zoe.infer_pil(img)

        colored_depth = colorize(depth)
        output = Image.fromarray(colored_depth)
        output.save(output_path)

    elif model=="DPT":
        dpt_depth_estimation(input_path, output_path, model_path="utils/DPT/weights/dpt_large-midas-2f21e586.pt", model_type="dpt_large")

    elif model=="DeepBump":
        print(input_path)
        os.system(f"python3 utils/DeepBump/cli.py {input_path} {output_path}/normals.png color_to_normals")
        os.system(f"python3 utils/DeepBump/cli.py {output_path}/normals.png {output_path}/depth.png normals_to_height")

    else:
        raise Exception("Depth estimation model not specified")

    return output_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Depth estimation script")

    parser.add_argument('--input_path', type=str, default="input/vase.png", help='Path to the masks')
    parser.add_argument('--output_path', type=str, default="output/", help='Path to output the results')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # Run depth-estimation
    depth_estimation(input_path, output_path, model="DPT")
    print("Depth estimation done")
