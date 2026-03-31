bl_info = {
    "name": "CudaSVGF",
    "blender": (3, 0, 0),
    "category": "Compositing",
}

import bpy
import subprocess
import os


# ----------------------------
# Compositor Setup
# ----------------------------

def setup_compositor(output_path):
    scene = bpy.context.scene
    scene.use_nodes = True

    tree = scene.node_tree
    tree.nodes.clear()

    rlayers = tree.nodes.new(type='CompositorNodeRLayers')
    composite = tree.nodes.new(type='CompositorNodeComposite')
    file_output = tree.nodes.new(type='CompositorNodeOutputFile')

    file_output.base_path = output_path
    file_output.format.file_format = 'PNG'
    file_output.file_slots.clear()

    slots = {
        "Render": "Image",
        "Normal": "Normal",
        "Albedo": "DiffCol",
        "Depth": "Depth"
    }

    for slot_name in slots:
        file_output.file_slots.new(slot_name)

    view_layer = bpy.context.view_layer
    view_layer.use_pass_normal = True
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_z = True

    for slot_name, output_name in slots.items():
        tree.links.new(rlayers.outputs[output_name], file_output.inputs[slot_name])

    tree.links.new(rlayers.outputs["Image"], composite.inputs["Image"])


# ----------------------------
# External execution
# ----------------------------

def run_cuda_svgf(props):
    exe_path = props.executable_path

    if not os.path.isfile(exe_path):
        print("CudaSVGF executable not found:", exe_path)
        return 1

    args = [
        exe_path,
        str(props.sigma_spatial),
        str(props.sigma_render),
        str(props.sigma_albedo),
        str(props.sigma_normal),
        props.output_path
    ]

    result = subprocess.run(args, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    return result.returncode


# ----------------------------
# Properties
# ----------------------------

class CudaSVGFProps(bpy.types.PropertyGroup):
    sigma_spatial: bpy.props.FloatProperty(name="Sigma Spatial", min=1, max=10.0, default=1)
    sigma_render: bpy.props.FloatProperty(name="Sigma Render", min=5, max=30.0, default=5)
    sigma_albedo: bpy.props.FloatProperty(name="Sigma Albedo", min=0.05, max=.25, default=0.05)
    sigma_normal: bpy.props.FloatProperty(name="Sigma Normal", min=0.01, max=0.1, default=0.01)

    output_path: bpy.props.StringProperty(name="Output Path", subtype='FILE_PATH', default="")
    executable_path: bpy.props.StringProperty(name="CUDA Executable", subtype='FILE_PATH', default="")


# ----------------------------
# Operators
# ----------------------------

class CUDASVGF_OT_setup(bpy.types.Operator):
    bl_idname = "cudasvgf.setup"
    bl_label = "Setup"

    def execute(self, context):
        props = context.scene.cudasvgf_props
        setup_compositor(props.output_path)
        self.report({'INFO'}, "Compositor setup complete")
        return {'FINISHED'}


class CUDASVGF_OT_denoise(bpy.types.Operator):
    bl_idname = "cudasvgf.denoise"
    bl_label = "Run"

    def execute(self, context):
        props = context.scene.cudasvgf_props
        code = run_cuda_svgf(props)

        if code == 0:
            self.report({'INFO'}, "CudaSVGF finished successfully")
        else:
            self.report({'ERROR'}, "CudaSVGF failed")

        return {'FINISHED'}


# ----------------------------
# UI Panel
# ----------------------------

class CUDASVGF_PT_panel(bpy.types.Panel):
    bl_label = "CudaSVGF"
    bl_idname = "CUDASVGF_PT_panel"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "CudaSVGF"

    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'CompositorNodeTree'

    def draw(self, context):
        layout = self.layout
        props = context.scene.cudasvgf_props

        layout.prop(props, "sigma_spatial")
        layout.prop(props, "sigma_render")
        layout.prop(props, "sigma_albedo")
        layout.prop(props, "sigma_normal")

        layout.separator()

        layout.prop(props, "output_path")
        layout.prop(props, "executable_path")

        layout.separator()

        layout.operator("cudasvgf.setup")
        layout.operator("cudasvgf.denoise")


# ----------------------------
# Register
# ----------------------------

classes = (
    CudaSVGFProps,
    CUDASVGF_OT_setup,
    CUDASVGF_OT_denoise,
    CUDASVGF_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.cudasvgf_props = bpy.props.PointerProperty(type=CudaSVGFProps)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    if hasattr(bpy.types.Scene, "cudasvgf_props"):
        del bpy.types.Scene.cudasvgf_props


if __name__ == "__main__":
    register()