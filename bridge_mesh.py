import bpy

# Clear all mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Import the .ply file
bpy.ops.import_mesh.ply(filepath="output/output_baran.ply")

# Get the imported object
obj = bpy.context.selected_objects[0]

# Add a Mirror modifier
bpy.ops.object.modifier_add(type='MIRROR')

# Set use_axis for the Mirror modifier, False for X and Y, True for Z  
bpy.context.object.modifiers["Mirror"].use_axis = (False, False, True)

# Enable merging and specify the merge limit
bpy.context.object.modifiers["Mirror"].use_mirror_merge = True
bpy.context.object.modifiers["Mirror"].merge_threshold = 5

# Apply the modifier
bpy.ops.object.modifier_apply(modifier="Mirror")

#Finally, we can then export the result
bpy.ops.export_mesh.ply(filepath="output/blender_output.ply") 