import os
import argparse

from inflation import inflate_mesh

def remove_background(img_path, output_path):
	from rembg import remove
	from PIL import Image

	image_raw = Image.open(img_path).convert('RGB')
	output = remove(image_raw, only_mask=True)
	output.save(output_path)
	return output_path


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Main script")


	parser.add_argument('--img_path', type=str, default="data/pillow.png", help='Path to the masks')
	parser.add_argument('--output_path', type=str, default="output/", help='Path to output the results')

	args = parser.parse_args()

	img_path = args.img_path
	output_path = args.output_path

	# Greyscale with Background removed
	print("Removing background...")
	mask = remove_background(img_path, output_path+'mask.png')

	# Generate depth map
	# depthmap

	# Inflate mesh based on Baran Method
	print("Inflating mesh...")
	inflate_mesh(mask, output_path)

	# Mirror inflated mesh
	print("Mirroring mesh...")
	os.system("/Applications/Blender.app/Contents/MacOS/blender --background --python mirror.py -noaudio")

	# Add depth map to mesh

	# * subdivision surface
	# * displacement modifier
	# * optional: Remesh

	# Export mesh
	# Export blender file

