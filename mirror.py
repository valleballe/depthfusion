import bpy
import sys
import os

argv = sys.argv
argv = argv[argv.index("--") + 1:] 

project_path = argv[0]
input_name = argv[1]
output_name = argv[2]


# Get the current script directory
current_script_directory = os.path.dirname(os.path.realpath(__file__))

# Create absolute directory path
input_file_path = os.path.join(current_script_directory, project_path, input_name)
texture_file_path = os.path.join(current_script_directory, project_path, "depth.png")
output_file_path = os.path.join(current_script_directory, project_path, output_name)
output_blender_file_path = os.path.join(current_script_directory, project_path, "scene.blend")

# Clear all mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Import the .ply file
#bpy.ops.import_mesh.ply(filepath=input_file_path)
bpy.ops.import_scene.obj(filepath=input_file_path)

# Ensure there's at least one object imported
if len(bpy.context.selected_objects) == 0:
    raise Exception("No objects found in the .obj file.")

# Get the imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Add a Mirror modifier
#bpy.ops.object.modifier_add(type='MIRROR')
#print("Applying mirror...")

# Set use_axis for the Mirror modifier, False for X and Y, True for Z  
#bpy.context.object.modifiers["Mirror"].use_axis = (False, False, True)

# Enable merging and specify the merge limit
#bpy.context.object.modifiers["Mirror"].use_mirror_merge = True
#bpy.context.object.modifiers["Mirror"].merge_threshold = 0.01

# Apply the modifier
#bpy.ops.object.modifier_apply(modifier="Mirror")

# Scale down
#bpy.context.scene.objects.active.scale = (0.01, 0.01, 0.01)
bpy.context.view_layer.objects.active.scale = (0.01, 0.01, 0.01)

# Apply scale
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Decimate
#print("Applying decimation...")
#bpy.ops.object.modifier_add(type='DECIMATE')
#bpy.context.object.modifiers["Decimate"].ratio = 0.01 # put desirable ratio


# Create a subdivision surface modifier
#print("Applying subdivision...")
#bpy.ops.object.modifier_add(type='SUBSURF')
#bpy.context.object.modifiers["Subdivision"].levels = 2
#bpy.context.object.modifiers["Subdivision"].render_levels = 2

# Create a displacement modifier
print("Applying displacement...")
bpy.ops.object.modifier_add(type='DISPLACE')
tex = bpy.data.textures.new('DepthMapTexture', type='IMAGE')
tex.image = bpy.data.images.load(texture_file_path)
bpy.context.object.modifiers["Displace"].texture = tex
bpy.context.object.modifiers["Displace"].texture_coords = 'UV'
bpy.context.object.modifiers["Displace"].strength = 2.0
bpy.context.object.modifiers["Displace"].mid_level = 0.5

# Apply Smooth Shading
bpy.ops.object.shade_smooth()

# Select only mesh
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)


# Save the scene as a blender file
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_mainfile(filepath=output_blender_file_path)

# Apply modifiers
#bpy.ops.object.modifier_apply(modifier="Displace")
#bpy.ops.object.modifier_apply(modifier="Subdivision")


# Export selected object
bpy.ops.export_scene.obj(filepath=output_file_path, use_selection=True)


# Export the result as .OBJ along with .MTL
#bpy.ops.export_scene.obj(filepath=output_file_path)
#bpy.ops.export_mesh.ply(filepath=output_file_path)
