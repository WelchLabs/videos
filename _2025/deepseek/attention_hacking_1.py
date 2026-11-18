from welch_paths import welch_path
import manimgx as gx
from deepseek_diagrams import get_attention_head
from welch_scene import WelchScene, image_at_height

from tqdm import tqdm
from pathlib import Path
# import glob

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


def get_attention_head_course(svg_path=welch_path('deepseek/graphics/to_manim'),
                       svg_file='mha_2d_grouped_separate_lines.svg',
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

    svg_image=gx.SVGMobject(str(svg_path/svg_file))
    # x_labels_1=svg_image[1:45] 
    attention_border=svg_image[45:233]
    text_labels=svg_image[233:449]
    lines=svg_image[449:]
    all_labels=gx.Group(attention_border, text_labels, lines)

    large_white_connectors=gx.SVGMobject(str(svg_path/'mha_2d_large_white_connectors_2.svg'))

    # all_head_labels=svg_image[188:-44]
    # attention_border=svg_image[:188]
    # all_labels=Group(all_head_labels, attention_border)

    return gx.Group(all_images, all_labels, large_white_connectors[1:])


def get_input_x(svg_path=welch_path('deepseek/graphics/to_manim'),
           svg_file='mha_2d_grouping_test.svg',
           img_path=welch_path('deepseek/hackin/linux_workdir/deepseek/flowchart_graphics')):


    x1=image_at_height(str(img_path/'input_1_1.png'))
    x1.scale([0.075,0.125, 1]) 
    x1.move_to([-1.45,-0.03,0])

    x2=image_at_height(str(img_path/'input_2_1.png'))
    x2.scale([0.075,0.125, 1]) 
    x2.move_to([-1.225,-0.03,0])
    
    all_images=gx.Group(x1, x2)
    svg_image=gx.SVGMobject(str(svg_path/svg_file))
    x_labels_1=svg_image[1:45] 

    return gx.Group(all_images,x_labels_1)


def get_mla_head(layer_id=0):
    img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek/deepseek_viz_absorbed_1'))
    svg_path=Path(welch_path('deepseek/graphics/to_manim'))
    svg_file='mla_panels-'

    # self.set_view(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)
    # self.set_view(0, 0, 0, (0.58, 0.21, 0.0), 1.86)

    svg_files=list(sorted((svg_path).glob('*'+svg_file+'*')))
    # print(svg_files)

    all_labels=gx.Group()
    for svg_file in svg_files:
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

class mla_draft(WelchScene):
    def construct(self):

        self.set_view(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)

        attention_heads=gx.Group()
        spacing=0.25
        for i in range(12): 
            a=get_mla_head(layer_id=6*i)
            a.rotate(gx.PI/2, axis=gx.RIGHT)
            a.move_to([0, spacing*i,0])
            attention_heads.add(a)

        for i in range(11, 0, -1): #Render in reverse order for occlusions

            self.add(attention_heads[i][1][7].set_opacity(0.9)) #Brown arrows on right side
            self.add(attention_heads[i][1][8]) #Thick white arrows
            self.add(attention_heads[i][0][2:].set_opacity(0.8)) #Images on right side
            # self.add(attention_heads[i][0][0].set_opacity(0.8)) #Proably show queries? We'll see

        ## -- Now do first layer a little differently
        self.add(attention_heads[0][0][2:].set_opacity(0.9)) #Images on right side
        # self.add(attention_heads[0][1][5].set_opacity(0.8)) #query labels
        self.add(attention_heads[0][1][7].set_opacity(0.9)) #Brown arrows on right side
        self.add(attention_heads[0][1][8].set_opacity(0.9)) #Thick white arrows
        self.add(attention_heads[0][1][9].set_opacity(0.9)) #Rigth side text
        self.add(attention_heads[0][1][11].set_opacity(0.9)) #Weighted latents labels
        

        connector_1=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_white_connector_1.svg'))
        connector_1.scale([0.75, 1.39, 1])
        connector_1.move_to([0.03, 1.375, -0.165]) #[-0.12, 1.375, -0.08]
        self.add(connector_1)

        connector_2=connector_1.copy()
        connector_2.move_to([0.03, 1.375, -0.335])
        self.add(connector_2)

        attention_heads[0][0][1].move_to([-0.63,  1.35,  -0.25 ]) #KV Cache
        self.add(attention_heads[0][0][1]) 
        self.add(attention_heads[0][1][4].move_to([-0.63 ,  1.35,  -0.46  ]))# Key labels


        self.wait()
        # Scene authoring stops here in the interactive editor.

class mla_hacking(WelchScene):
    def construct(self):

        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek/deepseek_viz_absorbed_1'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))
        svg_file='mla_panels-'

        # self.set_view(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)
        self.set_view(0, 0, 0, (0.58, 0.21, 0.0), 1.86)

        svg_files=list(sorted((svg_path).glob('*'+svg_file+'*')))
        # print(svg_files)

        all_labels=gx.Group()
        for svg_file in svg_files:
            svg_image=gx.SVGMobject(str(svg_file))
            all_labels.add(svg_image[1:]) #Thowout background

        self.add(all_labels[4:7])
        self.add(all_labels[8:])

        layer_id=0

        q1=image_at_height(str(img_path/str(layer_id)/'q_latent.png'))
        q1.scale([0.0075, 0.14, 1]) 
        q1.move_to([-0.285,0.6,0]) 
        self.add(q1)

        # self.remove(q1)      

        k1=image_at_height(str(img_path/'kv2.png'))
        k1.scale([0.0075, 0.14, 1]) 
        k1.move_to([-0.285,-0.01,0]) 
        self.add(k1)

        # self.remove(k1)

        kt=image_at_height(str(img_path/'kv2.png'))
        kt.scale([0.004, 0.045, 1])
        kt.rotate(-gx.PI/2)
        kt.move_to([0.293,0.485,0])     
        self.add(kt)
# 
        # self.remove(kt)

        a1=image_at_height(str(img_path/str(layer_id)/'scores_single_1.png'))
        a1.scale([0.055,0.055, 1])
        a1.move_to([0.56,0.56,0]) 
        self.add(a1)

        # self.remove(a1)

        a2=image_at_height(str(img_path/str(layer_id)/'scores_single_4.png'))
        a2.scale([0.13,0.13, 1])
        a2.move_to([1.155,0.445,0]) 
        self.add(a2)

        # self.remove(a2)

        z1=image_at_height(str(img_path/str(layer_id)/'alkv.png'))
        z1.scale([0.0079, 0.10, 1]) 
        z1.move_to([0.91,-0.2,0]) 
        self.add(z1)

        # self.remove(z1)

        # all_images=Group(q1, k1, v1, kt, a1, a2, z1)
        # Scene authoring stops here in the interactive editor.

        # large_white_connectors=SVGMobject(str(svg_path/'mha_2d_large_white_connectors_2.svg'))



class gqa_draft(WelchScene):
    def construct(self):

        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))

        self.set_view(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)

        attention_heads=gx.Group()
        spacing=0.25
        for i in range(12): 
            a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                        img_path=img_path/'gpt_2_attention_viz_1'/str(i))
            a.rotate(gx.PI/2, axis=gx.RIGHT)
            a.move_to([0, spacing*i,0])
            attention_heads.add(a)
        

        for i in range(11, 0, -1): #Render in reverse order for occlusions
            self.add(attention_heads[i][1][13].set_opacity(0.9)) #Arrows on right side
            # self.add(attention_heads[i][1][0].set_opacity(0.9)) #Outlines

            # self.add(attention_heads[i][0][0].set_opacity(0.8)) #queries
            ##should replace kt with same matrix in each head ->
            self.add(attention_heads[i][0][3:].set_opacity(0.8)) #attention patterns and output
            self.add(attention_heads[i][2]) #Thick white lines

        # -- Now do first layer a little differently
        # self.add(attention_heads[0][0][0].set_opacity(0.4)) #Queries
        self.add(attention_heads[0][0][3:].set_opacity(0.8)) #attention patterns and output.
        self.add(attention_heads[0][1][13].set_opacity(0.9)) #Arrows on right side
        # self.add(attention_heads[0][1][3].set_opacity(0.9)) #Query Label
        #Opationally add labels/variable names on front head
        self.add(attention_heads[0][1][14].set_opacity(0.5)) #Yea I'm 50/50 on this - nice to have the option i think
        self.add(attention_heads[0][2]) #Thick white lines


        #Keys
        connector_1=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_small_white_connector.svg'))
        connector_1.scale([0.3, 0.39, 1])
        connector_1.move_to([0.18, 1.38, 0.05]) #[-0.12, 1.375, -0.08]
        
        keys_a=attention_heads[0][0][1].move_to([-0.53 ,  1.391,  0.06 ])
        keys_a_labels=attention_heads[0][1][4].move_to([-0.53 ,  1.391,  -0.10 ])# Key labels

        connector_1b=connector_1.copy()
        keys_b=keys_a.copy()
        keys_b_labels=keys_a_labels.copy()
        connector_1b.move_to([0.18, 2.38, 0.05])
        keys_b.move_to([-0.53,  2.38,  0.06 ])
        keys_b_labels.move_to([-0.53 ,  2.38,  -0.10 ])


        connector_1c=connector_1.copy()
        keys_c=attention_heads[0][0][1].copy()
        keys_c_labels=attention_heads[0][1][4].copy()
        connector_1c.move_to([0.18, 0.38, 0.05])
        keys_c.move_to([-0.53,  0.38,  0.06 ])
        keys_c_labels.move_to([-0.53 ,  0.38,  -0.10 ])
        

        #Values
        connector_2=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_small_white_connector.svg'))
        connector_2.scale([0.3, 0.39, 1])
        connector_2.move_to([ 0.18 ,  1.38, -0.34 ])

        values_a=attention_heads[0][0][2].move_to([-0.53 ,  1.38, -0.34])
        values_a_labels=attention_heads[0][1][5].move_to([-0.53 ,  1.38,  -0.51 ])

        connector_2b=connector_2.copy()
        values_b=values_a.copy()
        values_b_labels=values_a_labels.copy()
        connector_2b.move_to([0.18, 2.38, -0.34])
        values_b.move_to([-0.53 ,  2.38, -0.34])
        values_b_labels.move_to([-0.53 ,  2.38,  -0.51 ])

        connector_2c=connector_2.copy()
        values_c=values_a.copy()
        values_c_labels=values_a_labels.copy()
        connector_2c.move_to([ 0.18 ,  0.38, -0.34 ])
        values_c.move_to([-0.53 ,  0.38, -0.34])
        values_c_labels.move_to([-0.53 ,  0.38,  -0.51 ])        


        self.add(connector_1b, keys_b, keys_b_labels)
        self.add(connector_2b, values_b, values_b_labels)
        self.add(connector_1, keys_a, keys_a_labels)
        self.add(connector_2, values_a, values_a_labels)
        self.add(connector_1c, keys_c, keys_c_labels)
        self.add(connector_2c, values_c, values_c_labels)
    
        
        self.wait()
        # Scene authoring stops here in the interactive editor.

        self.wait()


class mqa_draft_2(WelchScene):
    def construct(self):

        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))


        # a=get_attention_head(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1/0', 
        #                             svg_file='mha_2d_segments-')
        # self.add(a)
        # self.wait()

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(x)

        # self.set_view(0, 0, 0, (0.00, 0.00, 0.00), 2.80)
        # self.wait()

        self.set_view(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # x.move_to([-1.85,0,0]) #Shift input data to the left
        # x.rotate([PI/2,0,0], axis=RIGHT)
        # self.add(x)        

        # self.remove(x)

        attention_heads=gx.Group()
        spacing=0.25
        for i in range(12): #Render in reverse order for occlusions
            a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                        img_path=img_path/'gpt_2_attention_viz_1'/str(i))
            a.rotate(gx.PI/2, axis=gx.RIGHT)
            a.move_to([0, spacing*i,0])
            # a.set_opacity(0.5)
            attention_heads.add(a)
        
        # attention_heads.set_opacity(0.9)

        for i in range(11, 0, -1):
            self.add(attention_heads[i][1][13].set_opacity(0.9)) #Arrows on right side
            # self.add(attention_heads[i][1][0].set_opacity(0.9)) #Outlines

            self.add(attention_heads[i][0][0].set_opacity(0.8)) #queries
            ##should replace kt with same matrix in each head ->
            self.add(attention_heads[i][0][3:].set_opacity(0.8)) #attention patterns and output
            self.add(attention_heads[i][2]) #Thick white lines

        # -- Now do first layer a little differently
        # self.add(attention_heads[0][0][0].set_opacity(0.4)) #Queries
        self.add(attention_heads[0][0][3:].set_opacity(0.8)) #attention patterns and output.
        self.add(attention_heads[0][1][13].set_opacity(0.9)) #Arrows on right side
        # self.add(attention_heads[0][1][3].set_opacity(0.9)) #Query Label
        #Opationally add labels/variable names on front head
        self.add(attention_heads[0][1][14].set_opacity(0.5)) #Yea I'm 50/50 on this - nice to have the option i think
        self.add(attention_heads[0][2]) #Thick white lines

        connector_1=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_white_connector_1.svg'))
        connector_1.scale([1.0, 1.39, 1])
        connector_1.move_to([0.2, 1.386, 0.05]) #[-0.12, 1.375, -0.08]
        # (np.array([0.2, 1.386, 0.05])-np.array([-0.12, 1.375, -0.08]))+np.array([-0.85, 1.38, -0.07])
        self.add(connector_1) 

        #Keys
        attention_heads[0][0][1].move_to([-0.53 ,  1.391,  0.06 ])
        self.add(attention_heads[0][0][1]) 
        self.add(attention_heads[0][1][4].move_to([-0.53 ,  1.391,  -0.10 ]))# Key labels

        connector_1b=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_white_connector_1.svg'))
        connector_1b.scale([1.0, 1.39, 1])
        connector_1b.move_to([ 0.14 ,  1.386, -0.34 ])
        # (np.array([0.2, 1.386, 0.05])-np.array([-0.12, 1.375, -0.08]))+np.array([-0.18, 1.375, -0.47])
        self.add(connector_1b)

        #Values
        attention_heads[0][0][2].move_to([-0.53 ,  1.391, -0.34])
        self.add(attention_heads[0][0][2]) 
        self.add(attention_heads[0][1][5].move_to([-0.53 ,  1.391,  -0.51 ]))# Values labels
        # (np.array([0.2, 1.386, 0.05])-np.array([-0.12, 1.375, -0.08]))+np.array([-0.85, 1.38, -0.47])



        # head_connector=SVGMobject('/Users/stephen/welch_labs/deepseek/graphics/to_manim/head_connector_1.svg')
        # head_connector.scale([0.35, 1.385, 1])
        # head_connector.move_to([-1.45, 1.375, -0.047])
        # self.add(head_connector)
        self.wait()
        # Scene authoring stops here in the interactive editor.

        # attention_heads[:-1].set_opacity(0.5)
        # self.add(attention_heads)

        


        self.wait()


class mqa_draft(WelchScene):
    def construct(self):

        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))


        # a=get_attention_head_course(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1/0', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(a)

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(x)

        # self.set_view(0, 0, 0, (0.00, 0.00, 0.00), 2.80)
        # self.wait()

        self.set_view(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # x.move_to([-1.85,0,0]) #Shift input data to the left
        # x.rotate([PI/2,0,0], axis=RIGHT)
        # self.add(x)        

        # self.remove(x)

        attention_heads=gx.Group()
        spacing=0.25
        for i in range(12): #Render in reverse order for occlusions
            a=get_attention_head_course(svg_path=svg_path,svg_file='mha_2d_grouped_separate_lines.svg',
                                        img_path=img_path/'gpt_2_attention_viz_1'/str(i))
            a.rotate(gx.PI/2, axis=gx.RIGHT)
            a.move_to([0, spacing*i,0])
            # a.set_opacity(0.5)
            attention_heads.add(a)
        
        # attention_heads.set_opacity(0.9)

        for i in range(11, 0, -1):
            # attention_heads[i].set_opacity(0.6)
            # self.add(attention_heads[i][1][0]) #border
            # self.add(attention_heads[i][1][2]) #Lines
            # attention_heads[i][0].set_opacity(0.4)
            self.add(attention_heads[i][2]) #Thick white lines
            self.add(attention_heads[i][0][0].set_opacity(0.25)) #queries
            self.add(attention_heads[i][0][4:].set_opacity(0.25)) #attention patterns and output

        # self.add(attention_heads[0][0])
        # self.add(attention_heads[0][1][0]) #Border
        # self.add(attention_heads[0][1][1]) #Text 
        # self.add(attention_heads[0][1][2]) #Line

        connector_1=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_white_connector_1.svg'))
        connector_1.scale([1.0, 1.39, 1])
        connector_1.move_to([-0.12, 1.375, -0.08])
        self.add(connector_1)

        #Keys
        attention_heads[0][0][1].move_to([-0.85, 1.38, -0.07])
        self.add(attention_heads[0][0][1]) 

        connector_1b=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/thick_white_connector_1.svg'))
        connector_1b.scale([1.0, 1.39, 1])
        connector_1b.move_to([-0.18, 1.375, -0.47])
        self.add(connector_1b)

        #Values
        attention_heads[0][0][2].move_to([-0.85, 1.38, -0.47])
        self.add(attention_heads[0][0][2]) 



        # head_connector=SVGMobject('/Users/stephen/welch_labs/deepseek/graphics/to_manim/head_connector_1.svg')
        # head_connector.scale([0.35, 1.385, 1])
        # head_connector.move_to([-1.45, 1.375, -0.047])
        # self.add(head_connector)
        self.wait()
        # Scene authoring stops here in the interactive editor.

        # attention_heads[:-1].set_opacity(0.5)
        # self.add(attention_heads)

        


        self.wait()
        



class attention_hacking_4(WelchScene):
    def construct(self):

        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))


        # a=get_attention_head_course(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1/0', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(a)

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(x)

        # self.set_view(0, 0, 0, (0.00, 0.00, 0.00), 2.80)
        # self.wait()

        self.set_view(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)

        x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
                                    svg_file='mha_2d_grouped_separate_lines.svg')
        x.move_to([-1.85,0,0]) #Shift input data to the left
        x.rotate(gx.PI/2, axis=gx.RIGHT)
        self.add(x)        

        # self.remove(x)

        attention_heads=gx.Group()
        spacing=0.25
        for i in range(12): #Render in reverse order for occlusions
            a=get_attention_head_course(svg_path=svg_path,svg_file='mha_2d_grouped_separate_lines.svg',
                                        img_path=img_path/'gpt_2_attention_viz_1'/str(i))
            a.rotate(gx.PI/2, axis=gx.RIGHT)
            a.move_to([0, spacing*i,0])
            # a.set_opacity(0.5)
            attention_heads.add(a)
        
        # attention_heads.set_opacity(0.9)

        for i in range(11, 0, -1):
            attention_heads[i].set_opacity(0.6)
            self.add(attention_heads[i][1][0]) #border
            self.add(attention_heads[i][1][2]) #Lines
            # attention_heads[i][0].set_opacity(0.4)
            self.add(attention_heads[i][0]) #images

        self.add(attention_heads[0])

        head_connector=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/head_connector_1.svg'))
        head_connector.scale([0.35, 1.385, 1])
        head_connector.move_to([-1.45, 1.375, -0.047])
        self.add(head_connector)

        self.remove(head_connector)

        self.wait()
        # Scene authoring stops here in the interactive editor.

        # attention_heads[:-1].set_opacity(0.5)
        # self.add(attention_heads)

        


        self.wait()
        



class attention_hacking_3(WelchScene):
    def construct(self):

        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek/gpt_2_attention_viz_1/0'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))

        # self.set_view(0, 0, 0, (0.00, 0.00, 0.00), 2.80)
        self.set_view(0, 0, 0, (0.17, -0.07, 0.0), 2.23)

        svg_image=gx.SVGMobject(str(svg_path/'mha_2d_grouped_separate_lines.svg'))

        # Not sure yet if i want to break apart yet here on in illustrator
        # Sooo..what would really suck here is i make a change in illustrator, and have 
        # to figure out all this ineding again
        # I kinda suspect that ordering here has to do with depth/forward/backward values in illustrator
        # Which could certinaly change willy-nilly
        # So, it's more annoying to break it up in illustrator now, but I think that's a more sustainable
        # Way to solve the problem. 

        #Ok finding group breakpoints is not he worst workflow in the world:
        x_labels_1=svg_image[1:45] 
        self.add(x_labels_1)

        # self.remove(x_labels_1)

        attention_border=svg_image[45:233]
        self.add(attention_border)

        # self.remove(attention_border)

        text_labels=svg_image[233:449]
        self.add(text_labels)

        # self.remove(text_labels)

        lines=svg_image[449:]
        self.add(lines)

        # self.remove(wk)
        # self.add(svg_image)

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
        self.add(all_images)

        self.wait()
        # Scene authoring stops here in the interactive editor.

class attention_hacking_2(WelchScene):
    def construct(self):

        img_path=Path(welch_path('deepseek/hackin/linux_workdir/deepseek/flowchart_graphics'))
        svg_path=Path(welch_path('deepseek/graphics/to_manim'))

        self.set_view(0, 0, 0, (0.00, 0.00, 0.00), 2.80)

        x1=image_at_height(str(img_path/'input_1_1.png'))
        x1.scale([0.11,0.17, 1]) 
        x1.move_to([-1.96,-0.03,0])
        self.add(x1)

        x2=image_at_height(str(img_path/'input_2_1.png'))
        x2.scale([0.11,0.17, 1]) 
        x2.move_to([-1.65,-0.03,0])
        self.add(x2)       

        q1=image_at_height(str(img_path/'q_1.png'))
        q1.scale([0.062,0.10, 1]) 
        q1.move_to([-0.25,0.53,0])
        self.add(q1)       

        # self.remove(q1)
        k1=image_at_height(str(img_path/'k_1.png'))
        k1.scale([0.062,0.10, 1]) 
        k1.move_to([-0.25,-0.07,0])
        self.add(k1) 

        v1=image_at_height(str(img_path/'v_1.png'))
        v1.scale([0.062,0.10, 1]) 
        v1.move_to([-0.25,-0.66,0])
        self.add(v1) 

        kt=image_at_height(str(img_path/'k_1.png'))
        kt.scale([0.033,0.048, 1])
        kt.rotate(-gx.PI/2)
        kt.move_to([0.58,0.425,0])
        self.add(kt)     

        a1=image_at_height(str(img_path/'attention_scores.png'))
        a1.scale([0.083,0.083, 1])
        a1.move_to([0.93,0.525,0])
        self.add(a1) 

        a2=image_at_height(str(img_path/'attention_pattern.png'))
        a2.scale([0.18,0.18, 1])
        a2.move_to([1.76,0.35,0])
        self.add(a2) 

        z1=image_at_height(str(img_path/'z_1.png'))
        z1.scale([0.0625,0.10, 1]) 
        z1.move_to([1.445,-0.66,0])
        self.add(z1) 

        # self.remove(z1)


        # self.remove(a2)
        # self.remove(kt)


        svg_image=gx.SVGMobject(str(svg_path/'mha_2d_1.svg'))
        self.add(svg_image)

        self.wait()
        # Scene authoring stops here in the interactive editor.

class attention_hacking_1(WelchScene):
    def construct(self):

        svg_image=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/mha_2d_1.svg'))
        # self.add(svg_image)
        self.wait()



        self.play(gx.Create(svg_image)) #Not terrible.

        # svg_image_2=SVGMobject('/Users/stephen/welch_labs/deepseek/graphics/to_manim/attention_flow_10.svg')
        # svg_image.shift([0,0,2])
        # self.add(svg_image_2)

        img=image_at_height(welch_path('deepseek/hackin/linux_workdir/deepseek/flowchart_graphics/q_1.png'))

        # img.scale(0.07)
        img.scale([0.07,0.2, 1]) #Dope! We can do non-uniform scaling, nice. 
        img.move_to([-.5,.5,0])
        self.add(img)


        svg_image.set_opacity(0.5)
        img.set_opacity(0.5)

        # self.remove(img)

        self.wait()


        # mob = Line(
        #     start=_convert_point_to_3d(513.64, 540.0),
        #     end=_convert_point_to_3d(1451.75, 540.0)
        # )
        # mob.set_stroke_width(5)
        # mob = Line(
        #     start=_convert_point_to_3d(0,0),
        #     end=_convert_point_to_3d(4,4)
        # )

        # self.add(mob)

        self.wait()

        # Scene authoring stops here in the interactive editor.

# class EnhancedSVGMobject(SVGMobject):
#     def line_to_mobject(self, line: se.SimpleLine) -> Line:
#         # Create the line mobject
#         mob = Line(
#             start=_convert_point_to_3d(line.x1, line.y1),
#             end=_convert_point_to_3d(line.x2, line.y2)
#         )
        
#         # Set stroke properties
#         stroke_color = "#948979"  # Default from your SVG
#         stroke_width = 4  # Default from your SVG
        
#         if hasattr(line, 'stroke') and line.stroke:
#             stroke_color = line.stroke.hexrgb
#         if hasattr(line, 'stroke_width') and line.stroke_width:
#             stroke_width = line.stroke_width
            
#         # Explicitly set both stroke color and width
#         mob.set_stroke(color=stroke_color, width=stroke_width)
#         mob.set_fill(opacity=0)  # Lines should have no fill

#         print('here', line.x1, line.y1, line.x2, line.y2)
        
#         return mob

#     def apply_style_to_mobject(self, mob: VMobject, shape: se.GraphicObject) -> VMobject:
#         # Ensure stroke properties are explicitly set
#         if hasattr(shape, 'stroke'):
#             mob.set_stroke(
#                 color=shape.stroke.hexrgb if shape.stroke else None,
#                 width=shape.stroke_width if hasattr(shape, 'stroke_width') else None,
#                 opacity=shape.stroke.opacity if shape.stroke else None
#             )
        
#         if hasattr(shape, 'fill'):
#             mob.set_fill(
#                 color=shape.fill.hexrgb if shape.fill else None,
#                 opacity=shape.fill.opacity if shape.fill else None
#             )
        
#         return mob