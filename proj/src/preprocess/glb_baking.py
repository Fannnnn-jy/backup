import argparse
import sys
from pathlib import Path

import bpy


def parse_args():
    import argparse
    import sys

    argv = sys.argv[1:]

    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--margin", type=int, default=16)
    return parser.parse_args(argv)

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.textures,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def import_glb(path: str):
    bpy.ops.import_scene.gltf(filepath=path)
    objs = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objs:
        raise RuntimeError("No mesh objects imported from GLB")
    return objs


def join_meshes(objs):
    if len(objs) == 1:
        obj = objs[0]
        bpy.context.view_layer.objects.active = obj
        return obj

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def triangulate_mesh(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")


def ensure_uv(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=66.0,
        island_margin=0.02,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def detect_color_attribute(mesh):
    names = []

    if hasattr(mesh, "color_attributes"):
        for attr in mesh.color_attributes:
            names.append(attr.name)

    if hasattr(mesh, "vertex_colors"):
        for attr in mesh.vertex_colors:
            if attr.name not in names:
                names.append(attr.name)

    if not names:
        raise RuntimeError("No vertex color / color attribute found on imported mesh")

    for preferred in ["COLOR_0", "Col", "color", "Color"]:
        if preferred in names:
            return preferred

    return names[0]


def build_bake_material(obj, color_attr_name: str, texture_size: int):
    mat = bpy.data.materials.new(name="BakeVertexColorMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    for node in list(nodes):
        nodes.remove(node)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (500, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (220, 0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0

    attr = nodes.new(type="ShaderNodeAttribute")
    attr.location = (-420, 0)
    attr.attribute_name = color_attr_name

    bake_tex = nodes.new(type="ShaderNodeTexImage")
    bake_tex.location = (-120, -220)

    image = bpy.data.images.new(
        name="BakedBaseColor",
        width=texture_size,
        height=texture_size,
        alpha=False,
        float_buffer=False,
    )
    bake_tex.image = image

    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    nt.nodes.active = bake_tex
    return mat, image, bake_tex


def bake_diffuse_color(obj, margin: int):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.render.bake.margin = margin

    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})


def build_export_material(obj, image):
    mat = bpy.data.materials.new(name="ExportPBRMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    for node in list(nodes):
        nodes.remove(node)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (450, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (150, 0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0

    tex = nodes.new(type="ShaderNodeTexImage")
    tex.location = (-180, 0)
    tex.image = image

    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def export_glb(path: str, obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_vertex_color="NONE",
        export_image_format="AUTO",
        export_yup=True,
    )


def main():
    args = parse_args()

    input_path = str(Path(args.input).expanduser().resolve())
    output_path = str(Path(args.output).expanduser().resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    objs = import_glb(input_path)
    obj = join_meshes(objs)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    triangulate_mesh(obj)
    ensure_uv(obj)

    color_attr_name = detect_color_attribute(obj.data)
    print(f"[blender] Using color attribute: {color_attr_name}")

    _, image, _ = build_bake_material(obj, color_attr_name, args.texture_size)
    bake_diffuse_color(obj, args.margin)

    image.pack()

    build_export_material(obj, image)
    export_glb(output_path, obj)

    print(f"[blender] Wrote textured GLB to: {output_path}")


if __name__ == "__main__":
    main()