import os
import argparse

from inflation import inflate_mesh

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Main script")


	parser.add_argument('--img_path', type=str, default="data/character.png", help='Path to the masks')
	parser.add_argument('--output_path', type=str, default="output/", help='Path to output the results')

	args = parser.parse_args()

	img_path = args.img_path
	output_path = args.output_path

	# Greyscale with Background removed
	mask = img_path

	# Generate depth map
	# depthmap

	# Inflate mesh based on Baran Method
	inflate_mesh(mask, output_path)

	# Mirror inflated mesh
	#os.system("!blender --background --python bridge_mesh.py -noaudio")


	# Add depth map to mesh

	# * subdivision surface
	# * displacement modifier
	# * optional: Remesh

	# Export mesh
	# Export blender file

