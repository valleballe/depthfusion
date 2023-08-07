# depthfusion
 depthfusion


## Requirements
Tested on macOS


### Blender for... well.. blender
!apt install blender
!apt install libboost-all-dev
!apt install libgl1-mesa-dev

### Blender MAC
install in homebrew
echo "alias blender=/Applications/Blender.app/Contents/MacOS/blender" >> ~/.bash_profile
alias blender="open /Applications/Blender.app --args" 


### rembg for masking
!pip install rembg

### zoedepth for depth estimation
pip install timm==0.6.7
cd zoedepth
´´´python sanity.py´´´ # downloading models


### UI
!pip install streamlit
!pip install pyvista 
!pip install stpyvista 

