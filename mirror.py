import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] 

input_file_path = argv[0]
output_file_path = argv[1]
apply_subdivision = argv[2]

# Clear all mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Import the .ply file
bpy.ops.import_mesh.ply(filepath=input_file_path)

# Get the imported object
obj = bpy.context.selected_objects[0]

# Add a Mirror modifier
bpy.ops.object.modifier_add(type='MIRROR')
print("Applying mirror...")

# Set use_axis for the Mirror modifier, False for X and Y, True for Z  
bpy.context.object.modifiers["Mirror"].use_axis = (False, False, True)

# Enable merging and specify the merge limit
bpy.context.object.modifiers["Mirror"].use_mirror_merge = True
bpy.context.object.modifiers["Mirror"].merge_threshold = 5

# Apply the modifier
bpy.ops.object.modifier_apply(modifier="Mirror")

# Create a subdivision surface modifier
if apply_subdivision:
	print("Applying subdivision...")
	bpy.ops.object.modifier_add(type='SUBSURF')
	bpy.context.object.modifiers["Subdivision"].levels = 1
	bpy.context.object.modifiers["Subdivision"].render_levels = 2
	bpy.ops.object.modifier_apply(modifier="Subdivision")

# Apply Smooth Shading
bpy.ops.object.shade_smooth()

#Finally, we can then export the result
bpy.ops.export_mesh.ply(filepath=output_file_path) 
