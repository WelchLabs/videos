"""Image and SVG layout for the DeepSeek attention diagrams."""
from welch_paths import welch_path

from pathlib import Path

import manimgx as gx
from welch_scene import image_at_height


def get_attention_head(svg_path=welch_path('deepseek/graphics/to_manim'),
                       svg_file='mha_2d_segments-',
                       img_path=welch_path('deepseek/hackin/linux_workdir/deepseek/flowchart_graphics')):

    q1=image_at_height(str(img_path/'q_1.png'))
    q1.scale([0.0415, 0.08, 1])
    q1.move_to([-0.2,0.38,0])
    # self.add(q1)
    # self.remove(q1)

    k1=image_at_height(str(img_path/'k_1.png'))
    k1.scale([0.0415, 0.08, 1])
    k1.move_to([-0.2,-0.06,0])
    # self.add(k1)
    # self.remove(k1)

    v1=image_at_height(str(img_path/'v_1.png'))
    v1.scale([0.0415, 0.08, 1])
    v1.move_to([-0.2,-0.48,0])
    # self.add(v1)
    # self.remove(v1)

    kt=image_at_height(str(img_path/'k_1.png'))
    kt.scale([0.0215,0.035, 1])
    kt.rotate(-gx.PI/2)
    kt.move_to([0.405,0.305,0])
    # self.add(kt)
    # self.remove(kt)

    a1=image_at_height(str(img_path/'attention_scores.png'))
    a1.scale([0.055,0.055, 1])
    a1.move_to([0.66,0.37,0])
    # self.add(a1)
    # self.remove(a1)

    a2=image_at_height(str(img_path/'attention_pattern.png'))
    a2.scale([0.13,0.13, 1])
    a2.move_to([1.27,0.25,0])
    # self.add(a2)
    # self.remove(a2)

    z1=image_at_height(str(img_path/'z_1.png'))
    z1.scale([0.0425, 0.08, 1])
    z1.move_to([1.035,-0.48,0])
    # self.add(z1)
    # self.remove(z1)

    all_images=gx.Group(q1, k1, v1, kt, a1, a2, z1)

    svg_files=list(sorted((svg_path).glob('*'+svg_file+'*')))
    # print(svg_files)

    all_labels=gx.Group()
    for svg_file in svg_files:
        svg_image=gx.SVGMobject(str(svg_file))
        all_labels.add(svg_image[1:]) #Thowout background

    large_white_connectors=gx.SVGMobject(str(svg_path/'mha_2d_large_white_connectors_2.svg'))

    return gx.Group(all_images, all_labels, large_white_connectors[1:])


def get_input_x(svg_path=welch_path('deepseek/graphics/to_manim'),
           svg_file='mha_2d_grouping_test.svg',
           img_path=welch_path('deepseek/hackin/linux_workdir/deepseek/flowchart_graphics')):


    x1=image_at_height(str(Path(img_path)/'input_1_1.png'))
    x1.scale([0.075,0.125, 1])
    x1.move_to([-1.45,-0.03,0])

    x2=image_at_height(str(Path(img_path)/'input_2_1.png'))
    x2.scale([0.075,0.125, 1])
    x2.move_to([-1.225,-0.03,0])

    all_images=gx.Group(x1, x2)
    svg_image=gx.SVGMobject(str(Path(svg_path)/svg_file))
    x_labels_1=svg_image[1:45]

    return gx.Group(all_images,x_labels_1)


def get_niave_mla_head(layer_id=0):


        input_image_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek/flowchart_graphics'))
        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek/deepseek_viz_2'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))
        svg_file='niave_mla_panels-'

        # self.set_view(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)
        # self.set_view(0, 0, 0, (0.58, 0.21, 0.0), 1.86)

        svg_files=list(sorted((svg_path).glob('*'+svg_file+'*')))
        # print(svg_files)

        all_labels=gx.Group()
        for svg_file in svg_files[:-1]: #Dont bring in last panel
            svg_image=gx.SVGMobject(str(svg_file))
            all_labels.add(svg_image[1:]) #Thowout background

        # for l in all_labels:
        #     self.add(l)

        x1=image_at_height(str(Path(input_image_path)/'input_1_1.png'))
        x1.scale([0.075,0.125, 1])
        x1.move_to([-1.47,0.155,0])
        # self.add(x1)
        # self.remove(x1)

        x2=image_at_height(str(Path(input_image_path)/'input_2_1.png'))
        x2.scale([0.075,0.125, 1])
        x2.move_to([-1.24,0.155,0])
        # self.add(x2)
        # self.remove(x2)

        q1=image_at_height(str(img_path/str(layer_id)/'q_nope.png'))
        q1.scale([0.0215, 0.078, 1])
        q1.move_to([-0.185,0.59,0])
        # self.add(q1)
        # self.remove(q1)

        k1=image_at_height(str(img_path/str(layer_id)/'k_1.png'))
        k1.scale([0.0218, 0.078, 1])
        k1.move_to([-0.18,0.155,0])
        # self.add(k1)
        # self.remove(k1)

        kt=image_at_height(str(img_path/str(layer_id)/'k_1.png'))
        kt.scale([0.0112, 0.03, 1])
        kt.rotate(-gx.PI/2)
        kt.move_to([0.425,0.5,0])
        # self.add(kt)
        # self.remove(kt)

        v1=image_at_height(str(img_path/str(layer_id)/'v_1.png'))
        v1.scale([0.0218, 0.078, 1])
        v1.move_to([-0.18,-0.27,0])
        # self.add(v1)
        # self.remove(v1)

        kv=image_at_height(str(img_path/'kv2.png'))
        kv.scale([0.0111, 0.115, 1])
        kv.move_to([-1.095,-0.735,0])
        # self.add(kv)
        # self.remove(kv)

        a1=image_at_height(str(img_path/str(layer_id)/'scores_single_1.png'))
        a1.scale([0.055,0.055, 1])
        a1.move_to([0.69,0.57,0])
        # self.add(a1)
        # self.remove(a1)

        a2=image_at_height(str(img_path/str(layer_id)/'scores_single_4.png'))
        a2.scale([0.13,0.13, 1])
        a2.move_to([1.28,0.445,0])
        # self.add(a2)
        # self.remove(a2)

        z1=image_at_height(str(img_path/str(layer_id)/'x_1.png'))
        z1.scale([0.022, 0.07, 1])
        z1.move_to([1.055,-0.28,0])
        # self.add(z1)
        # self.remove(z1)

        all_images=gx.Group(x1, x2, kv, q1, k1, kt, v1, a1, a2, z1)

        return gx.Group(all_images, all_labels)


def get_mla_head(layer_id=0):
    img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek/deepseek_viz_absorbed_1'))
    svg_path=Path(welch_path('deepseek/graphics/to_manim'))
    svg_file='mla_panels-'

    # self.set_view(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)
    # self.set_view(0, 0, 0, (0.58, 0.21, 0.0), 1.86)

    svg_files=list(sorted((svg_path).glob('*'+svg_file+'*')))
    # print(svg_files)

    all_labels=gx.Group()
    for svg_file in svg_files[1:]: #Skip first artboard
        svg_image=gx.SVGMobject(str(svg_file))
        all_labels.add(svg_image[1:]) #Thowout background

    # self.add(all_labels[4:7])
    # self.add(all_labels[8:])

    # layer_id=0

    q1=image_at_height(str(img_path/str(layer_id)/'q_latent.png'))
    q1.scale([0.0075, 0.14, 1])
    q1.move_to([-0.285,0.6,0])
    # self.add(q1)

    # self.remove(q1)

    k1=image_at_height(str(img_path/'kv2.png'))
    k1.scale([0.0075, 0.14, 1])
    k1.move_to([-0.285,-0.01,0])
    # self.add(k1)

    # self.remove(k1)

    kt=image_at_height(str(img_path/'kv2.png'))
    kt.scale([0.004, 0.045, 1])
    kt.rotate(-gx.PI/2)
    kt.move_to([0.293,0.485,0])
    # self.add(kt)
#
    # self.remove(kt)

    a1=image_at_height(str(img_path/str(layer_id)/'scores_single_1.png'))
    a1.scale([0.055,0.055, 1])
    a1.move_to([0.56,0.56,0])
    # self.add(a1)

    # self.remove(a1)

    a2=image_at_height(str(img_path/str(layer_id)/'scores_single_4.png'))
    a2.scale([0.13,0.13, 1])
    a2.move_to([1.155,0.445,0])
    # self.add(a2)

    # self.remove(a2)

    z1=image_at_height(str(img_path/str(layer_id)/'alkv.png'))
    z1.scale([0.0079, 0.10, 1])
    z1.move_to([0.91,-0.2,0])
    # self.add(z1)

    # self.remove(z1)

    all_images=gx.Group(q1, k1, kt, a1, a2, z1)
    # self.embed()

    return gx.Group(all_images, all_labels)
