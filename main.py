import os
import argparse

# UI
#import streamlit as st
#import pyvista as pv
#from stpyvista import stpyvista

# remove background
import cv2
import numpy as np
from rembg import remove
from PIL import Image	

# own packages
from inflation import inflate_mesh
from depth_estimation import depth_estimation
from run_monodepth import depth_estimation as dpt_depth_estimation

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



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Main script")


	parser.add_argument('--img_path', type=str, default="data/cenk/5.png", help='Path to the masks')
	parser.add_argument('--output_path', type=str, default="output/", help='Path to output the results')

	args = parser.parse_args()
	
	"""
	# Setup Streamlit APP
	st.title('DepthFusion')
	file = st.file_uploader("Please choose a file")

	if file != None:
		st.image(Image.open(file), caption='Uploaded Images')
	"""

	img_path = args.img_path
	output_path = args.output_path

	project_path = output_path+img_path.split("/")[-1].split(".")[-2]+"/"
	if not os.path.isdir(project_path):
		os.makedirs(project_path)

	# Greyscale with Background removed
	print("Removing background...")
	mask = remove_background(img_path, project_path+'mask.png') ## < --- needs to add black region all across object
	#mask = output_path+'mask.png'
	#st.image(Image.open(mask), caption='Mask Image')

	# Generate depth map
	print("Generating depthmap...")
	#depth_map_path = zoe_depth_estimation(img_path, project_path+"depth.png")
	depth_map_path = dpt_depth_estimation(img_path, project_path, model_path="utils/DPT/weights/dpt_large-midas-2f21e586.pt", model_type="dpt_large")
	
	# Inflate mesh based on Baran Method
	print("Inflating mesh...")
	inflate_mesh(mask, img_path, project_path, depth_map_path, apply_depth_map=True, max_depth=1, depth_map_weight=1)
	#viz3D(output_path + 'inflated_mesh.ply')

	# Mirror inflated mesh
	print("Mirroring mesh...")
	apply_subdivision = False if Image.open(img_path).size[0] > 512 else True
	os.system(f"/Applications/Blender.app/Contents/MacOS/blender --background -noaudio --python mirror.py -- {project_path+'inflated_mesh.ply'} {project_path+'final_mesh.ply'} {apply_subdivision}")

	# Finish
	print("Finished")

	# Convert Vertex color to UV
	#print("Generate UV...")
	#os.system("/Applications/Blender.app/Contents/MacOS/blender --background --python vertex2uv.py -noaudio")	


	# Add depth map to mesh
	#print("Adding subdivider, displacement, remeshing...")
	#os.system("/Applications/Blender.app/Contents/MacOS/blender --background --python displacement.py -noaudio")
	#print("Finished")

	# * subdivision surface
	# * displacement modifier
	# * optional: Remesh

	# Export mesh
	# Export blender file

