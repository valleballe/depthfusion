import os
import json
import openai	
import urllib.request
from PIL import Image  

class DallE:

	def __init__(self, project_path, n_images=1, size="1024x1024"):

		self.prompt = "A smiling AI system"
		self.n_images = n_images
		self.size = size
		self.project_path = project_path

		# Make output dir
		os.makedirs(self.project_path, exist_ok = True)
		
		
	def generate_image(self, prompt=""):

		print("Generating image...")

		if prompt == "":
			prompt = self.prompt

		# Generate Image
		response = openai.Image.create(
		  prompt=prompt,
		  n=self.n_images,
		  size=self.size,
		)

		# Set filename
		filename = os.path.join(self.project_path, "colormap.png")

		# Download file
		self.download_file(
			download_url=response["data"][0]["url"], 
			filename=filename)

		return filename

	def set_API_key(self):
		openai.api_key = OPENAI_API_KEY


	def download_file(self, download_url, filename):
	    response = urllib.request.urlopen(download_url)    
	    file = open(filename, 'wb')
	    file.write(response.read())
	    file.close()

	def show_image(self, filename):
		im = Image.open(filename) 
		im.show()



if __name__ == "__main__":
	img_generator = DallE(project_path="output/vase/")
	img_generator.generate_image("Image of an otter swimming in a lake")
