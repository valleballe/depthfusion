import os
import argparse
import openai

# remove background
import cv2
import numpy as np
from rembg import remove
from PIL import Image	

# own packages
from inflation import inflate_mesh
from depth_estimation import depth_estimation
from ImageGenerator import DallE

def remove_background(img_path, output_path):

	image_raw = Image.open(img_path).convert('RGB')
	output = remove(image_raw, only_mask=True)
	output.save(output_path)

	# threshold images
	mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
	mask[mask > 100] = 255
	mask[mask <= 100] = 0

	# make black boundary around image
	mask[0,:] = 0
	mask[-1,:] = 0
	mask[:,0] = 0
	mask[:,-1] = 0

	# eliminate floating points
	new_mask = eliminate_floating_regions(mask)

	# contract mask slighly from edge
	new_mask = contract_mask(mask, contraction_size=5)

	# save result
	cv2.imwrite(output_path, new_mask)

	return output_path

def eliminate_floating_regions(mask):
    # close everything inside
	contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE);

	# extract alpha channel
	alpha = mask[:,:]

	# get the biggest contour # returns _, contours, _ if using OpenCV 3
	biggest_area = -1;
	biggest = None;
	for con in contour:
		area = cv2.contourArea(con);
		if biggest_area < area:
			biggest_area = area;
			biggest = con;

	# smooth contour
	peri = cv2.arcLength(biggest, True)
	biggest = cv2.approxPolyDP(biggest, 0.0001 * peri, True)

	# draw white filled contour on black background
	mask = np.zeros_like(alpha)
	#cv2.drawContours(contour_img, [big_contour], 0, 255, -1)


	# fill in the contour
	cv2.drawContours(mask, [biggest], -1, 255, -1);
	return mask

def contract_mask(mask, contraction_size):
    # Create a kernel of ones with the same shape as the contraction size
    kernel = np.ones((contraction_size, contraction_size), np.uint8)

    # Use OpenCV's erosion function to "contract" the mask
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Main script")


	parser.add_argument('--prompt', type=str, default="", help='Prompt to generate image')
	parser.add_argument('--input_path', type=str, default="data/vase.png", help='Path to the masks')
	parser.add_argument('--output_path', type=str, default="output/", help='Folder to output the results')
	parser.add_argument('--apply_mirror', type=bool, default=True, help='Boolean for whether to mirror mesh')
	parser.add_argument('--decimate', type=float, default=0.001, help='Value for decimating mesh. If \'0\' then no decimation')
	parser.add_argument('--subdivide', type=int, default=1, help='Value for subdividing mesh. If \'0\' then no subdivision')
	parser.add_argument('--displace', type=float, default=1.0, help='Value for displacing mesh. If \'0.0\' then no subdivision')

	args = parser.parse_args()
	
	"""
	# Setup Streamlit APP
	st.title('DepthFusion')
	file = st.file_uploader("Please choose a file")

	if file != None:
		st.image(Image.open(file), caption='Uploaded Images')
	"""

	prompt = "sideview of monkey vase"#args.prompt
	img_path = args.input_path
	output_path = args.output_path
	apply_mirror = args.apply_mirror
	decimate = args.decimate
	subdivide = args.subdivide
	displace = args.displace


	# API key
	openai.api_key = "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"


	# Load image
	if prompt != "":

		# Create folders
		project_path = os.path.join(output_path, prompt.replace(" ","_"))
		os.makedirs(project_path, exist_ok = True)

		# Generate Image
		#img_generator = DallE(project_path=project_path)
		#filename = img_generator.generate_image(prompt)
		img_path = os.path.join(project_path, "colormap.png")



	elif input_path!="":

		# Create folders
		project_path = os.path.join(output_path, input_path)
		os.makedirs(project_path, exist_ok = True)
			
		# Copy color map to project dir
		img = cv2.imread(img_path)
		img_path = project_path+"colormap.png"
		cv2.imwrite(img_path, img)

	else:
		print("No input image or prompt to generate image specified")


	# Remove background
	print("Removing background...")
	mask = remove_background(img_path, os.path.join(project_path,'mask.png'))
	#mask = output_path+'mask.png'
	#st.image(Image.open(mask), caption='Mask Image')

	# Generate depth map
	print("Generating depthmap...")
	#depth_map_path = zoe_depth_estimation(img_path, project_path+"depth.png")
	depth_map_path = depth_estimation(img_path, project_path, model="DeepBump")
	
	# Inflate mesh based on Baran Method
	print("Inflating mesh...")
	inflate_mesh(mask, img_path, project_path, depth_map_path, apply_depth_map=False, max_depth=1, depth_map_weight=1)

	# Mirror inflated mesh
	if apply_mirror:
		print("Mirroring mesh...")
		os.system(f"/Applications/Blender.app/Contents/MacOS/blender --background -noaudio --python mirror.py -- \
			{project_path} \
			{'inflated_mesh.obj'} \
			{'final_mesh.obj'} \
			")

	# Decimate mesh
	#if decimate > 0:
		#print("Decimating mesh...")
		#os.system(f"/Applications/Blender.app/Contents/MacOS/blender --background -noaudio --python decimate_mesh.py -- {project_path} {'mirrored_mesh.obj'} {'decimated_mesh.obj'}")

	# Subdivision
	#if subdivide > 0:
		#print("Decimating mesh...")
		#os.system(f"/Applications/Blender.app/Contents/MacOS/blender --background -noaudio --python subdivide_mesh.py -- {project_path} {'decimated_mesh.obj'} {'subdivided_mesh.obj'}")

	# Displace mesh with depthmap
	#if displace > 0:
		#print("Displacing mesh...")
		#os.system("/Applications/Blender.app/Contents/MacOS/blender --background -noaudio --python displacement.py -- {project_path} {'subdivided_mesh.obj'} {'final_mesh.obj'}")

	# * optional: Remesh

	# Export blender file
	# os.system("/Applications/Blender.app/Contents/MacOS/blender --background --python generate_blender_file.py -noaudio")


	# Finish
	print("Finished")

