# Depthfusion
For a manual version with more finegrained control [click here](https://github.com/valleballe/ai-generated-artifacts).


## Requirements
Tested on macOS

```
# Install packages
pip install rembg timm==0.6.7 openai numpy onnxruntime imageio

# install blender
apt install blender
apt install libboost-all-dev
apt install libgl1-mesa-dev

# download depth inference models
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt -P utils/DPT/weights
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-ade20k-53898607.pt -P utils/DPT/weights

# install zoedepth
cd zoedepth
python sanity.py # downloading models
```

If you are on mac, you can use homebrew to install blender
```
brew install blender
echo "alias blender=/Applications/Blender.app/Contents/MacOS/blender" >> ~/.bash_profile
alias blender="open /Applications/Blender.app --args" 
```

## Limitations
* Can only generate mirrorable objects. Complex objects like chairs are not recommended.

##  Contribute

Depthfusion is under active development and contributors are welcome. If you have any suggestions, feature requests, or bug reports, please open new [issues](https://github.com/valleballe/depthfusion/issues) on GitHub. 


## BibTeX Citation

If you use AI Generated Artifacts in a scientific publication or installation, we would appreciate using the following citations:

```
@software{danry2023,
  author = {Danry, Valdemar},
  month = {4},
  title = {{Depthfusion}},
  url = {https://github.com/valleballe/depthfusion},
  year = {2023}
}

