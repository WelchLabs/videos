from manimlib import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/vla/graphics/to_manim/')
hacking_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/vla/hackin')

class P31_61_1(InteractiveScene):
    def construct(self): 
        '''
        Ok not sure how I want to break stuff up just yet - lets start hacking
        and see where we end up. 
        '''

        svgs_to_skip=[0, 2, 3]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(3.9)
            all_svgs.add(svg_image[1:]) #Thowout background

        
        # P31
        # I'll have to some hacking in premiere to make the
        # video actually play
        # Although it wouldn'b be that bad to make this play in manim
        # Now that I have things setup - probably do this in a separte
        # scene to avoid slowing down the big scene - could be a quick 
        # pranav hack later. 
        final_time_series=ImageMobject(str(hacking_dir/'p31/000/pred_tall/299.png'))
        final_time_series.scale(0.7)
        final_time_series.move_to([2.3, -2.1, 0])

        legend=ImageMobject(str(hacking_dir/'p31/legend_2.png'))
        legend.scale(0.6)
        legend.move_to([6.1, -2.1, 0])
        legend.set_opacity(0.8)

        final_image_overhead=ImageMobject(str(hacking_dir/'p31/000/high/299.jpg'))
        final_image_overhead.scale(0.78)
        final_image_overhead.move_to([-4.55, 1.83, 0])

        final_image_left=ImageMobject(str(hacking_dir/'p31/000/left_wrist/299.jpg'))
        final_image_left.scale(0.78)
        final_image_left.move_to([-0.15, 1.83, 0])

        final_image_right=ImageMobject(str(hacking_dir/'p31/000/right_wrist/299.jpg'))
        final_image_right.scale(0.78)
        final_image_right.move_to([4.21, 1.83, 0])

        prompt=Text('"Uncap the pen"', font="Myriad Pro", weight='bold', font_size=28)
        prompt.move_to([-5.6, -2.08, 0])

        pi0_box = RoundedRectangle(
            width=1.85,
            height=1.55,
            corner_radius=0.1,
            stroke_color=FRESH_TAN,
            stroke_width=2,
            fill_opacity=0,
        )
        pi0_box.move_to([-2.76, -2.1, 0])       

        pi0_logo=Tex(r'\pi_0', font_size=60)
        pi0_logo.set_color(FRESH_TAN)
        pi0_logo.move_to(pi0_box)
        pi0_logo.shift([0, -0.05, 0])


        self.add(final_time_series, legend, final_image_overhead, final_image_left, final_image_right)
        self.add(prompt, pi0_box, pi0_logo)
        self.add(all_svgs[0])
        self.wait()

        # P31b 
        # Move things around to start drilling into Pi0!
        pi0_box_2 = RoundedRectangle(
            width=5.1,
            height=4.0,
            corner_radius=0.1,
            stroke_color=CHILL_BROWN,
            stroke_width=1,
            fill_opacity=0,
        )
        pi0_box_2.move_to([-0.25, 0.7, 0])   
        self.wait()

        self.remove(all_svgs[0], legend)
        self.play(ReplacementTransform(pi0_box, pi0_box_2), 
                  prompt.animate.scale(0.9).move_to([-5.5, -3.34, 0]),
                  final_image_overhead.animate.scale(0.66).move_to([-5.23, 2.58, 0]),
                  final_image_left.animate.scale(0.66).move_to([-5.23, 0.35, 0]),
                  final_image_right.animate.scale(0.66).move_to([-5.23, -1.82, 0]),
                  final_time_series.animate.scale(0.6).move_to([4.7, 0.78, 0]),
                  pi0_logo.animate.scale(0.70).move_to([2, -1.1, 0]),
                  run_time=6)


        # self.add(all_svgs[1])
        # prompt.scale(0.9).move_to([-5.5, -3.34, 0])
        # final_image_overhead.scale(0.66)
        # final_image_overhead.move_to([-5.23, 2.58, 0])
        # final_image_left.scale(0.66)
        # final_image_left.move_to([-5.23, 0.35, 0])
        # final_image_right.scale(0.66)
        # final_image_right.move_to([-5.23, -1.82, 0])
        # final_time_series.scale(0.6)
        # final_time_series.move_to([4.7, 0.78, 0])

        #Pi0 is built on top of PaliGemma
        self.wait()
        self.play(Write((all_svgs[3])), 
                  self.frame.animate.reorient(0, 0, 0, (-0.24, 0.62, 0.0), 5.09), 
                  run_time=4)
        self.wait()

        self.play(Write((all_svgs[2])), run_time=4)
        self.play(Write((all_svgs[4])), run_time=4)
        self.wait()

        self.play(self.frame.animate.reorient(0, 0, 0, (0, 0, 0), 8), 
                  Write(all_svgs[1]), 
                  run_time=5)
        


        #P32 

        action_expert_box = RoundedRectangle(
            width=3.2,
            height=1.5,
            corner_radius=0.1,
            stroke_color=YELLOW,
            stroke_width=2,
            fill_opacity=0,
        )
        action_expert_box.move_to([0.33, -1.8, 0])
        pi0_box_3 = RoundedRectangle(
            width=5.1,
            height=5.7,
            corner_radius=0.1,
            stroke_color=CHILL_BROWN,
            stroke_width=1,
            fill_opacity=0,
        )
        pi0_box_3.move_to([-0.25, -0.15, 0]) 

        action_expert_label=Text('ACTION EXPERT', font="Myriad Pro", weight='bold', font_size=24)
        action_expert_label.set_color(YELLOW)
        action_expert_label.move_to(action_expert_box)

        action_expert_sublabel=Text('gemma_expert = GemmaForCausalLM()', font="consolas", font_size=16)
        action_expert_sublabel.next_to(action_expert_label, DOWN, buff=0.13)


        self.wait()
        self.play(ReplacementTransform(pi0_box_2, pi0_box_3), 
                  pi0_logo.animate.move_to([2, -2.8, 0]), 
                  final_time_series.animate.move_to([4.7, -1.7, 0]),
                  all_svgs[1][-2:].animate.move_to([2.5, -1.8, 0]), #lil Arrow
                  run_time=4)
        self.play(ShowCreation(action_expert_box), 
                  Write(action_expert_label), 
                  run_time=3)

        self.wait()
        self.play(Write(action_expert_sublabel), 
                 self.frame.animate.reorient(0, 0, 0, (0.45, -1.81, 0.0), 4.13), 
                 run_time=3)
        self.wait()

        # Hmm this might actually be a good time to go ahead and do the image patch switcheroo
        # before zooming back out!

        # P35
        # Ok, a few kinda messy/complicated things going on here as 
        # We get into P35. 
        # Ok so I do want to move smoothly from current scene into new one 
        # here. Let me just start by hacking on moving pieces over that I 
        # know need to move!

        # Ok ok ok ok how are we going to break apart these images
        # Ok Pranav's approach looks pretty solid -
        # Image patches to disk - I like that
        # Kinda weird to do it in manim I feel like
        # Let me export patches in jupyter - 
        # I think we go ahead and do it for all timesteps
        # Then I can replace the image with the patchified version


        #The old switcheroo with image patches?
        height, width = 224, 244
        grid_n = 16
        patch_h = height // grid_n
        patch_w = width // grid_n
        total_height = 2.72
        patch_size = total_height / grid_n

        FRAME_IDX=150
        pixel_squares = Group()

        for image_name in ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']:
            pixel_squares.add(Group())
            patch_dir = hacking_dir/('p35/'+str(FRAME_IDX)+'/'+image_name)
            for i in range(2, 14): #Skip top 2 and botton 2 rows
                for j in range(grid_n):
                    patch_path = os.path.join(patch_dir, f'patch_{i}_{j}.png')
                    patch_mob = ImageMobject(patch_path)
                    patch_mob.set_height(patch_size)
                    patch_mob.set_width(patch_size, stretch=True)
                    x_pos = (j - grid_n/2 + 0.5) * patch_size
                    y_pos = -(i - grid_n/2 + 0.5) * patch_size
                    patch_mob.move_to([x_pos, y_pos, 0])
                    pixel_squares[-1].add(patch_mob)

        pixel_squares[0].move_to([-5.23,  2.58,  0.])
        pixel_squares[1].move_to([-5.23,  0.375,  0.])
        pixel_squares[2].move_to([-5.23,  -1.82,  0.])

        self.add(pixel_squares)
        self.remove(final_image_overhead, final_image_left, final_image_right)
        self.remove(all_svgs[1]); self.add(all_svgs[1]) #Occlusions bra

        self.wait()



        #Zoom back out to setup p35, image patches already in place!
        self.play(self.frame.animate.reorient(0, 0, 0, (0, 0, 0), 8), 
                  run_time=3)
        self.wait()
        self.remove(action_expert_sublabel)
        self.wait()

        # Ok I'm a little fuzzy on order here, definitely want to show images
        # breaking apart very soon, probably with some zoom in action. 

        siglip_1=all_svgs[2][:13]
        siglip_2=all_svgs[2][13:26]
        siglip_3=all_svgs[2][26:39]
        image_encoders_label=all_svgs[2][39:]

        self.wait()
        self.play(pi0_logo.animate.scale(0.85).set_color(CHILL_BROWN).to_corner(DOWN + RIGHT, buff=0.25),
                  FadeOut(pi0_box_3), 
                  FadeOut(all_svgs[3]),
                  FadeOut(all_svgs[1]),
                  FadeOut(final_time_series), 
                  FadeOut(action_expert_label), 
                  FadeOut(action_expert_box), 
                  all_svgs[4].animate.set_color(CHILL_BROWN).move_to([2, 0.4, 0]),
                  pixel_squares[1].animate.shift([0, -0.2, 0.0]),
                  pixel_squares[2].animate.shift([0, -0.4, 0.0]),
                  prompt.animate.shift([0.2, -0.3, 0.0]),
                  siglip_1.animate.scale(1.1).move_to([-3.0, 2.6, 0]),
                  siglip_2.animate.scale(1.1).move_to([-3.0, 0.2, 0]),
                  siglip_3.animate.scale(1.1).move_to([-3.0, -2.15, 0]),
                  image_encoders_label.animate.scale(1.1).move_to([-3.05, 3.5, 0]),
                  run_time=4)

        # siglip_1.scale(1.1)
        # siglip_1.move_to([-3.1, 2.6, 0])

        # siglip_2.scale(1.1)
        # siglip_2.move_to([-3.1, 0.2, 0])

        # siglip_3.scale(1.1)
        # siglip_3.move_to([-3.1, -2.15, 0])

        # image_encoders_label.scale(1.1)
        # image_encoders_label.move_to([-3.15, 3.5, 0])

        

        # Hmm still fuzzy on order and zooming in vs not -
        # Let me try to build the "end product" a little bit, and the work backwards
        # Image expansion is especially a little tricky

        # self.play(*animations, 
        #             pixel_squares[1].animate.shift([0, -0.2, 0.0]),
        #             pixel_squares[2].animate.shift([0, -0.4, 0.0]),
        #             prompt.animate.shift([0.2, -0.3, 0.0]),

        
        animations = []
        gap_factor = 0.12

        for i in range(len(pixel_squares)):
            center = pixel_squares[i].get_center()
            
            for pixel in pixel_squares[i]:
                pixel_pos = pixel.get_center()
                direction_vector = pixel_pos - center
                distance = np.linalg.norm(direction_vector)

                if distance > 0:
                    unit_vector = direction_vector / distance
                    displacement = unit_vector * distance * gap_factor
                    new_position = pixel_pos + displacement
                    animations.append(ApplyMethod(pixel.move_to, new_position))

        self.wait()
        self.play(*animations, run_time=3.0)
        self.wait()


        # pixel_squares[1].shift([0, -0.2, 0.0])
        # pixel_squares[2].shift([0, -0.4, 0.0])
        # prompt.shift([0.2, -0.3, 0.0])


        # pi0_logo.set_color(CHILL_BROWN)
        # pi0_logo.scale(0.85)
        # pi0_logo.to_corner(DOWN + RIGHT, buff=0.25)

        # self.remove(pi0_box_3) #Do a fade out 
        # self.remove(all_svgs[3])
        # self.remove(all_svgs[1])
        # self.remove(final_time_series)
        # self.remove(action_expert_label)
        # self.remove(action_expert_box)
        # all_svgs[4].set_color(CHILL_BROWN) #LLM baux
        # all_svgs[4].move_to([2, 0.4, 0])










        self.wait()





        self.wait(20)
        self.embed()














