from welch_paths import welch_path
import manimgx as gx
from deepseek_diagrams import get_mla_head, get_niave_mla_head
from welch_scene import WelchScene, image_at_height

from tqdm import tqdm
from pathlib import Path


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


class thumbnail_1(WelchScene):
    def construct(self):
        '''This sequence and the next will be a bit tricky, but I'll get some good mileage out of them 
           and then be home free. With this first one, I do like the idea of startiwth full 3d Niave MLA,
           and landing on 2d view to really explain it. Will require "bringing forward" the KV cache to 
           the 3d flat view and I think adding back in X (and probably queries) when I land on 2d space.'''

        # self.set_view(0, 0, 0, (-0.12, 0.02, 0.0), 2.36)
        
        # a=get_niave_mla_head(0)
        # self.add(a)

        #Ok great, now what elements do we want in the 3d view?
        # elements_3d=Group(a[0][3:], a[1][4:13])
        # self.add(elements_3d)

        attention_heads=gx.Group()
        spacing=0.25
        for i in range(12): 
            a=get_niave_mla_head(layer_id=6*i) #Step by 6 to get more variety
            a.rotate(gx.PI/2, axis=gx.RIGHT)
            a.move_to([0, spacing*i,0])
            attention_heads.add(a) #Group((a[0][3:], a[1][4:13])))





        ## Now need single latents and connector - need latents to come forward aw se collapse down to 2D
        # Let's sstart with trying latents fully "beneath", move to to the left if that doesn't work. 
        #I may want to make my white connectors shorter?
        og_kv_cache_center=attention_heads[0][0][2].get_center().copy()
        self.add(attention_heads[0][0][2].move_to([-1.0780741 ,  1.38,  -0.714-0.15])) #KV Cache

        connector_1=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/medium_white_connector.svg'))
        connector_1.rotate(gx.PI/2, gx.DOWN)
        connector_1.scale([0.4, 1.39, 1])
        connector_1.move_to([ -0.99 ,  1.38, -0.56 ])
        self.add(connector_1)

        # self.remove(connector_1)

        self.set_view(-41, 72, 0, (-0.28, 0.77, -0.13), 2.98)
        # self.set_view(-37, 67, 0, (-0.84, 1.04, -0.56), 2.47) #Option to start more zoomed in on kv cache
        self.add(attention_heads[0][1][4:6]) #These aren't sticking on top for some reaon, add again. 
        self.wait()

        for i in range(11, 0, -1): #Render in reverse order for occlusions
            self.add(attention_heads[i][0][3:].set_opacity(0.8)) #Images
            self.add(attention_heads[i][1][2:4].set_opacity(0.8)) #Flow chart
            self.add(attention_heads[i][1][11:13].set_opacity(0.8))
            self.add(attention_heads[i][1][14].set_opacity(0.8))
            self.add(attention_heads[i][1][4:6].set_opacity(0.8)) #Flow chart - thick white lines
            self.add(attention_heads[i][1][-1].set_opacity(0.8))


            # --- Add some then all of first head info -> unclear when I fade out other heads yet
        self.add(attention_heads[0][0][3:].set_opacity(0.9)) #Images on right side
        self.add(attention_heads[0][1][2:4].set_opacity(0.9))
        self.add(attention_heads[0][1][6:17].set_opacity(0.9)) #Flow chart
        self.add(attention_heads[0][1][4:6].set_opacity(0.9))
        self.add(attention_heads[0][1][-1].set_opacity(0.9))

        ##
        self.wait()
        # self.set_view(0, 65, 0, (0.25, 0.86, 0.07), 3.29)
        self.set_view(0, 66, 0, (0.22, 0.86, 0.07), 3.24)









        # Now pan camera to front while bringing KV cache forward and fading out think white connector - maybe leaving behind 
        # more chill brown ones?

        self.play(*[gx.FadeOut(attention_heads[i][1][4:6]) for i in range(12)]+
                   [gx.FadeOut(connector_1)]+
                   [attention_heads[0][0][2].animate.move_to(og_kv_cache_center)]+
                   [self.animate_view(0, 90, 0, (-0.05, 0.79, -0.04), 2.70)],
                   run_time=20) #Do an option with cranked up runtime - this covers a couple paragraphs
  
        #Tried a nice fade out here, couldn't quite get it - sratchign this plan      
        # to_fade_out_a=Group(*[attention_heads[i][0][3:] for i in range(1,12)]+
        #                      [attention_heads[i][1][2:4] for i in range(1,12)]+
        #                      [attention_heads[i][1][11:13] for i in range(1,12)]+
        #                      [attention_heads[i][1][14] for i in range(1,12)])

        # self.wait()
        # self.play(to_fade_out_a.animate.set_opacity(0)) #, FadeIn(attention_heads[0][0][:2]), FadeIn(attention_heads[0][1][:2]))

        for i in range(1, 12): 
            self.remove(attention_heads[i])
        self.add(attention_heads[0][0][:2])
        self.add(attention_heads[0][1][:2])

        self.wait()

        # Considering zooming in here - but there's a lot going on visually and I think everything is visible - I think 
        # we pretty much hold this view and highlight stuff in editing for p37 and p38. 
        # Eh stuff is kinda small, let me add a little zoom in/zoom out - I can edit this out if I don't like it. 

        self.play(attention_heads[0][0][3].animate.set_opacity(0.1), attention_heads[0][0][5].animate.set_opacity(0.1), 
                  attention_heads[0][0][7:].animate.set_opacity(0.1), 
                  attention_heads[0][1][8].animate.set_opacity(0.1),
                  attention_heads[0][1][11:17].animate.set_opacity(0.1),
                  self.animate_view(0, 89, 0, (-0.69, 0.79, -0.26), 2.18), run_time=2) #Zoom in to new steps

        self.wait()


        self.play(attention_heads[0][0][3].animate.set_opacity(1.0), attention_heads[0][0][5].animate.set_opacity(1.0), 
                  attention_heads[0][0][7:].animate.set_opacity(1.0), 
                  attention_heads[0][1][8].animate.set_opacity(1.0),
                  attention_heads[0][1][11:17].animate.set_opacity(1.0),
                  self.animate_view(0, 90, 0, (-0.05, 0.79, -0.04), 2.70), run_time=2) #Zoom back out
        self.wait()


        # self.set_view(0, 89, 0, (-0.05, 0.79, -0.04), 2.67)

        #Ok so I was going to do a break here to a new class and go back to 2d -> but since I want to go back to 3d at 
        # the end, perhaps i do the "absorbption" in 3d space - this is a little scary but I think i can figure it out
        # and it will pay off in being able to pop right back to 3d. 
        # Ok, now how do I do the matrix absorption?
        # Maybe I get the end state together here and then figure out how to animate to it?
        # Man kinda a lot changes! I don't want it to necessarily feel like that though - ya know?
        mla_heads=gx.Group()
        spacing=0.25
        for i in range(12): 
            a=get_mla_head(layer_id=6*i)
            a.rotate(gx.PI/2, axis=gx.RIGHT)
            a.move_to([0, spacing*i,0]) #Probably some shifting and scaling I need to come back and apply here
            mla_heads.add(a)


        self.wait()
        # Hmm do i actually just want the names of the matrices to move? And the arrows and boxes just fade in/out?
        # That might be cleaner

        old_query_arrow_and_box = gx.Group(attention_heads[0][1][0][54:62], #Dimensions of WQ
                                        attention_heads[0][1][0][43:46], #Bounding Box and one arrow
                                        attention_heads[0][1][0][37:39], #Part of arrow
                                        attention_heads[0][1][0][76])

        self.wait()
        self.play(gx.FadeOut(old_query_arrow_and_box), gx.FadeOut(attention_heads[0][1][3]),
                  gx.FadeOut(attention_heads[0][1][7][3:]),
                  self.animate_view(0, 89, 0, (-0.55, 0.79, 0.21), 2.03))
        self.play(attention_heads[0][1][0][41:43].animate.move_to([-0.96,  0., 0.6]),
                  attention_heads[0][1][7][:3].animate.move_to([-0.855,0,0.604]))
        self.play(gx.FadeOut(attention_heads[0][1][0][41:43]), gx.FadeOut(attention_heads[0][1][7][:3]),
                 gx.FadeIn(mla_heads[0][1][1]))

        self.wait()

        #man so many little piece to align. 
        mla_heads[0][1][11].move_to([1.75, 0, -0.26]) #Output matrix multiply, add when off sscreen
        self.add(mla_heads[0][1][11])
        self.remove(mla_heads[0][1][11][1:4]) #Wuv final


        self.play(self.animate_view(0, 89, 0, (0.51, 0.78, -0.15), 2.43))
        self.play(attention_heads[0][1][6][:3].animate.move_to([1.78, 0, -0.24]), run_time=2)
        # self.remove(attention_heads[0][1][6][:3])
        # self.add(mla_heads[0][1][11][1:4]) #Swapping out these is kinda jarring - just keep old and group if I need to move stuff?
        self.wait()


        # Ok so I think this will fall on the last sentence of 39, we need to make all the other freaking changes
        # that fall out of weight absoprtion I'm really hoping that I can get away with a simple fade in/fade out
        # we'll see! I want the weight matrices I just put together to stick around - but if I need to shift them a little
        # I think that's fine? Hmm lookin at how stuff falls, I do kinda want to at least animate my latent matrix moving over
        # This should help with continuity and shouldn't be too bad. 
        self.play(self.animate_view(0, 89, 0, (0.13, 0.78, -0.05), 2.86))
        self.wait()


        # self.add(mla_heads[0][1][4])
        # self.add(mla_heads[0][1][6])
        # self.add(mla_heads[0][1][8:11])
        # self.add(mla_heads[0][1])

        labels_to_remove=gx.Group(attention_heads[0][1][1:3], attention_heads[0][1][6][3:],
                                 attention_heads[0][1][1:3], attention_heads[0][1][0][69:],
                                 attention_heads[0][1][0][39:41], attention_heads[0][1][8:17]) 
        labels_to_add=gx.Group(mla_heads[0][1][2:5], mla_heads[0][1][6], mla_heads[0][1][8:12])

        # self.add(labels_to_remove)
        # self.remove(labels_to_remove)
        # self.remove(labels_to_add)

        combined_out_matrix_multiply=gx.Group(mla_heads[0][1][11][0], mla_heads[0][1][11][4:], attention_heads[0][1][6][:3])
        # combined_out_matrix_multiply.move_to([1.57, 0, -0.21])

        # self.remove(labels_to_add)

        # self.remove(combined_out_matrix_multiply)

        # self.play(attention_heads[0][0][2].animate.move_to())
        self.remove(labels_to_remove)
        self.play(gx.Transform(attention_heads[0][0][2], mla_heads[0][0][1]),
                  gx.Transform(attention_heads[0][0][7], mla_heads[0][0][3]),
                  gx.Transform(attention_heads[0][0][8], mla_heads[0][0][4]),
                  gx.FadeOut(attention_heads[0][0][3:7]), gx.FadeOut(attention_heads[0][0][9]),
                  gx.FadeIn(mla_heads[0][0][0]), gx.FadeIn(mla_heads[0][0][2]), gx.FadeIn(mla_heads[0][0][5]),
                  combined_out_matrix_multiply.animate.move_to([1.57, 0, -0.21]), run_time=1.5)
        self.add(labels_to_add) #Fading these in makes for weird arrow overlaps - ah actualy I'm getting them both ways
        self.remove(mla_heads[0][1][11][1:4])
        self.play(self.animate_view(0, 89, 0, (0.03, 0.78, 0.03), 2.72))
        self.wait()


        self.play(self.animate_view(0, 89, 0, (-0.12, 0.78, 0.1), 2.48)) #This is really just for an easier cut later
        self.wait()
        self.play(self.animate_view(0, 89, 0, (0.03, 0.78, 0.03), 2.72))
        self.wait()


        #Additional zoom in/zoom outs
        self.play(self.animate_view(0, 89, 0, (-0.65, 0.78, 0.24), 1.90))
        self.wait()
        self.play(self.animate_view(0, 89, 0, (0.16, 0.78, 0.28), 2.12))
        self.wait()
        self.play(self.animate_view(0, 89, 0, (0.03, 0.78, 0.03), 2.72))
        self.wait()




        ## -- Probably want to do some kinda fade in and pan - let me get everything added first. 
        connector_1=gx.SVGMobject(welch_path('deepseek/graphics/to_manim/medium_white_connector.svg'))
        connector_1.scale([0.75, 1.39, 1])
        connector_1.move_to([0.08, 1.38, 0.070]) #[-0.12, 1.375, -0.08]

        connector_2=connector_1.copy()
        connector_2.move_to([0.08, 1.38, -0.095])

        full_mla_3d_layers=gx.Group(*[mla_heads[i][0][2:] for i in range(11, 0, -1)]+
                                  [mla_heads[i][1][6] for i in range(11, 0, -1)]+
                                  [mla_heads[i][1][7] for i in range(11, 0, -1)]) #+#thick white arrows
                                  # [mla_heads[i][0][0] for i in range(11, 0, -1)]) #Queries

        to_fade_out_2d=gx.Group(attention_heads[0][0][0], attention_heads[0][0][1], mla_heads[0][1][1:3],
                             mla_heads[0][1][4], #mla_heads[0][1][11], #attention_heads[0][1][6][:3], 
                             attention_heads[0][1][0][:37], attention_heads[0][1][0][41:43], attention_heads[0][1][0][46:54],
                             attention_heads[0][1][0][62:69]) #, attention_heads[0][1][0][77:])


        #Ok this kinda sucks but I think i need to remove these first to avoid some weird jump behavior. 
        self.remove(attention_heads[0][1][6][:3])
        self.remove(mla_heads[0][1][11])
        # # self.remove(to_fade_out_2d)
        # # self.add(to_fade_out_2d)
        # self.play(FadeOut(attention_heads[0][1][6][:3]))
        # self.remove(mla_heads[0][1][1:3])
        # self.add(full_mla_3d_layers)


        full_mla_3d_layers.set_opacity(0)
        self.remove(attention_heads[0][0][2])
        self.add(mla_heads[0][0][1])

        self.wait()
        self.play(full_mla_3d_layers.animate.set_opacity(1.0), 
                 gx.FadeOut(to_fade_out_2d),
                 mla_heads[0][0][1].animate.move_to([-0.5,  1.35,  0 ]), #KV Cache
                 mla_heads[0][1][3].animate.move_to([-0.5 ,  1.35,  -0.2 ]), #Key labels
                 self.animate_view(-38, 72, 0, (0.08, 0.71, 0.08), 2.62),
                  run_time=3)

        #Re add top layer items to avoid occlusions
        self.add(mla_heads[0][0][2:6])
        self.add(mla_heads[0][1][3:5])
        self.add(mla_heads[0][1][6])
        self.add(mla_heads[0][1][8:11])
        self.remove(mla_heads[0][1][11][1:4])
        # self.add(mla_heads[0][1][3]) 
        # self.add(mla_heads[0][0][0]) #Quries
        self.remove(mla_heads[0][0][0])
        self.remove(mla_heads[0][1][4])
        self.add(mla_heads[0][1][7]) #add bold white arrows on top layer
        self.add(connector_1)
        self.add(connector_2)
        self.add(mla_heads[0][0][1]) #Move in front of occlusions array

        self.wait()

        #Optional zoom in to the kv cache matrix
        # self.play(self.animate_view(-30, 69, 0, (-0.47, 1.09, -0.0), 1.01), run_time=10)
        # self.wait()

        # self.play(self.animate_view(-38, 72, 0, (0.08, 0.71, 0.08), 2.62), run_time=3)
        # self.wait() 

        # self.play(full_mla_3d_layers.animate.set_opacity(0.0), 
        #          FadeIn(to_fade_out_2d),
        #          mla_heads[0][0][1].animate.move_to([-0.5,  1.35,  0 ]), #KV Cache
        #          mla_heads[0][1][3].animate.move_to([-0.5 ,  1.35,  -0.2 ]), #Key labels
        #          self.animate_view(0, 89, 0, (0.03, 0.78, 0.03), 2.72),
        #           run_time=3)       


        # self.play(self.animate_view(4, 81, 0, (0.08, 0.71, 0.08), 2.62), run_time=10, rate_func=linear) #Slow pan.
        # self.play(self.animate_view(0, 89, 0, (-0.12, 0.78, 0.1), 2.48), run_time=10, rate_func=linear) #Slow pan.
        # self.wait()

        # # self.set_view(-38, 59, 0, (0.12, 0.93, -0.18), 2.93)
        # self.set_view(-38, 59, 0, (0.36, 0.99, 0.14), 2.48)



        # self.set_view(-38, 70, 0, (0.33, 0.76, 0.15), 1.93)
        # self.set_view(-50, 70, 0, (0.39, 1.0, 0.07), 2.18)


        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class niave_mla_hacking(WelchScene):
    def construct(self):
        '''This sequence and the next will be a bit tricky, but I'll get some good mileage out of them 
           and then be home free. With this first one, I do like the idea of startiwth full 3d Niave MLA,
           and landing on 2d view to really explain it. Will require "bringing forward" the KV cache to 
           the 3d flat view and I think adding back in X (and probably queries) when I land on 2d space.'''

        # niave_mla_panels-01.svg
        # Ok I don't have 3d naive MLA preped, so first this is to probably build the 2d version live here
        # and then roll up into a support function

        self.set_view(0, 0, 0, (-0.12, 0.02, 0.0), 2.36)

        layer_id=0

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

        for l in all_labels:
            self.add(l)

            

        x1=image_at_height(str(Path(input_image_path)/'input_1_1.png'))
        x1.scale([0.075,0.125, 1]) 
        x1.move_to([-1.47,0.155,0])
        self.add(x1)
        # self.remove(x1)    

        x2=image_at_height(str(Path(input_image_path)/'input_2_1.png'))
        x2.scale([0.075,0.125, 1]) 
        x2.move_to([-1.24,0.155,0])
        self.add(x2)
        # self.remove(x2)        

        q1=image_at_height(str(img_path/str(layer_id)/'q_nope.png'))
        q1.scale([0.0215, 0.078, 1]) 
        q1.move_to([-0.185,0.59,0]) 
        self.add(q1)
        # self.remove(q1)      

        k1=image_at_height(str(img_path/str(layer_id)/'k_1.png'))
        k1.scale([0.0218, 0.078, 1]) 
        k1.move_to([-0.18,0.155,0]) 
        self.add(k1)
        # self.remove(k1) 

        kt=image_at_height(str(img_path/str(layer_id)/'k_1.png'))
        kt.scale([0.0112, 0.03, 1])
        kt.rotate(-gx.PI/2)
        kt.move_to([0.425,0.5,0])     
        self.add(kt)
        self.remove(kt)

        v1=image_at_height(str(img_path/str(layer_id)/'v_1.png'))
        v1.scale([0.0218, 0.078, 1]) 
        v1.move_to([-0.18,-0.27,0]) 
        self.add(v1)
        # self.remove(v1)      

        kv=image_at_height(str(img_path/'kv2.png'))
        kv.scale([0.0111, 0.115, 1]) 
        kv.move_to([-1.095,-0.735,0]) 
        self.add(kv)
        # self.remove(kv)

        a1=image_at_height(str(img_path/str(layer_id)/'scores_single_1.png'))
        a1.scale([0.055,0.055, 1])
        a1.move_to([0.69,0.57,0]) 
        self.add(a1)
        # self.remove(a1)

        a2=image_at_height(str(img_path/str(layer_id)/'scores_single_4.png'))
        a2.scale([0.13,0.13, 1])
        a2.move_to([1.28,0.445,0]) 
        self.add(a2)
        # self.remove(a2)

        z1=image_at_height(str(img_path/str(layer_id)/'x_1.png'))
        z1.scale([0.022, 0.07, 1]) 
        z1.move_to([1.055,-0.28,0]) 
        self.add(z1)
        # self.remove(z1)

        all_images=gx.Group(x1, x2, q1, k1, kt, v1, kv, a1, a2, z1)







        self.wait(20)
        # Scene authoring stops here in the interactive editor.
