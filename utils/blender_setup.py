import bpy

scene = bpy.context.scene
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()

rlayers = tree.nodes.new(type='CompositorNodeRLayers')
composite = tree.nodes.new(type='CompositorNodeComposite')
file_output = tree.nodes.new(type='CompositorNodeOutputFile')

file_output.format.file_format = 'PNG'
file_output.file_slots.clear()

slots = {
    "Render": "Image",
    "Normal": "Normal",
    "Albedo": "DiffCol"
}

for slot_name in slots:
    file_output.file_slots.new(slot_name)

view_layer = bpy.context.view_layer
view_layer.use_pass_normal = True
view_layer.use_pass_diffuse_color = True
view_layer.use_pass_z = True

for slot_name, output_name in slots.items():
    tree.links.new(rlayers.outputs[output_name], file_output.inputs[slot_name])

bpy.context.scene.render.engine = 'CYCLES'
tree.links.new(rlayers.outputs["Image"], composite.inputs["Image"])