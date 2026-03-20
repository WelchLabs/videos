from manimlib import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import colorsys


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


SATURATION_BOOST=1.5 #1.3
MIN_SATURATION=0.2 #0.1
MIN_VALUE=0.5 #0.3

def patch_bright_average(img, exponent=2.0):
    patches = img.reshape(16, 14, 16, 14, 3)
    chroma = patches.max(axis=-1, keepdims=True) - patches.min(axis=-1, keepdims=True)
    weights = chroma**exponent / (chroma**exponent).sum(axis=(1,3), keepdims=True)
    return (patches * weights).sum(axis=(1, 3))  # (16, 16, 3)

def boost_colors_hsv(colors, saturation_boost=1.0, min_saturation=0.0, min_value=0.0):
    """
    Adjust colors in HSV space for better visibility in the barcode viz.
 
    Parameters
    ----------
    colors : ndarray, shape (N, 3)
        RGB colors in [0, 1].
    saturation_boost : float
        Multiplier on S channel. 1.0 = no change, 1.5 = 50% more saturated.
    min_saturation : float in [0, 1]
        Floor for S channel. Prevents fully gray rows.
    min_value : float in [0, 1]
        Floor for V channel. Lifts dark/muddy colors so the hue reads clearly.
 
    Returns
    -------
    boosted : ndarray, shape (N, 3), dtype float32
    """
    colors = np.asarray(colors, dtype=np.float32)
    out = np.empty_like(colors)
    for i in range(len(colors)):
        h, s, v = colorsys.rgb_to_hsv(*colors[i])
        s = min(1.0, s * saturation_boost)
        s = max(s, min_saturation)
        v = max(v, min_value)
        out[i] = colorsys.hsv_to_rgb(h, s, v)
    return out

class P31_61_1(InteractiveScene):
    def construct(self): 
        '''
        Ok not sure how I want to break stuff up just yet - lets start hacking
        and see where we end up. 
        '''

        svgs_to_skip=[0, 2, 3, 8, 12, 13, 20, 21]
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

        for k, image_name in enumerate(['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']):
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
                  siglip_3.animate.scale(1.1).move_to([-3.0, -2.1, 0]),
                  image_encoders_label.animate.scale(1.1).move_to([-3.0, 3.5, 0]),
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


        #Ok, mid p35 here -> now I think it's a zoom in and move patches over kinda deal

        lil_arrows_pair_1=all_svgs[5]
        lil_arrows_pair_2=lil_arrows_pair_1.copy()
        lil_arrows_pair_3=lil_arrows_pair_1.copy()
        lil_arrows_pair_1.move_to([-3.0, 2.57, 0])
        lil_arrows_pair_2.move_to([-3.0, 0.18, 0])
        lil_arrows_pair_3.move_to([-3.0, -2.18, 0])

        embedding_brackets_1=all_svgs[6]
        embedding_brackets_1.shift([0.08, 0.00,0 ])

        # Ok I think a zoom in here, then pan down as we move patches over?

        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (-3.4, 1.91, 0.0), 4.03),
                  Write(embedding_brackets_1),
                  Write(lil_arrows_pair_1),
                  Write(lil_arrows_pair_2),
                  Write(lil_arrows_pair_3),
                  run_time=5
                  )

        # Ok, making progress here, now I want to bring over embedding vectors 
        # as I pan down - I think that will work
        # Maybe want to do a little script tweaking - we'll see!


        # self.add(embedding_brackets_1)
        # self.add(lil_arrows_pair_1, lil_arrows_pair_2, lil_arrows_pair_3)

        # Alright let me figure out how to draw these embedding vectors
        # Then how to animate between patches and vectors
        # So I think what will make sense is to compute the colors first
        # That was kinda hacky -> i'll compute the mod'd average colors 
        # for each patch and then export this to disk bruh. 
        # Hmm actually main method is quite simple, so unless 
        # I end up needing to do crazy mod stuff, then let's try computing 
        # patch colors in manim. 


        # Hmm I don't actually have the image numpy arrays loaded up
        # Let's do that next. 
        # patch_bright_average()


        #Ok now we just make some colored lines and put em in the right spots?
        # l=Line([-2.05, 3.1, 0], [-0.95, 3.1, 0])
        # l.set_stroke(color=boosted_color, width=4)
        # self.add(l)

        # overhead_im_full=Image.open(hacking_dir/('p35/full_size_base_0_rgb/'+str(FRAME_IDX).zfill(3)+'.jpg'))
        # left_im_full=Image.open(hacking_dir/('p35/full_size_base_0_rgb/'+str(FRAME_IDX).zfill(3)+'.jpg'))
        # right_im_full=Image.open(hacking_dir/('p35/full_size_base_0_rgb/'+str(FRAME_IDX).zfill(3)+'.jpg'))

        overhead_im_full=np.load(hacking_dir/'p35/150_overhead.npy')
        left_im_full=np.load(hacking_dir/'p35/150_left.npy')
        right_im_full=np.load(hacking_dir/'p35/150_right.npy')

        overhead_colors=patch_bright_average(np.array(overhead_im_full), exponent=2.0).reshape(-1, 3) #(16, 16, 3)  
        left_colors=patch_bright_average(np.array(left_im_full), exponent=2.0).reshape(-1, 3) #(16, 16, 3)  
        right_colors=patch_bright_average(np.array(right_im_full), exponent=2.0).reshape(-1, 3) #(16, 16, 3)  

        patches_indices_to_move_1=[0, 1, 2, 3, 4, 5, 6, 7, 8]
        embedding_rows_1=VGroup()
        starting_squares_1=VGroup()
        vertical_spacing=0.2
        for i, patch_index in enumerate(patches_indices_to_move_1):

            boosted_color=rgb_to_color(boost_colors_hsv(overhead_colors[patch_index+32].reshape(1,3)/255., 
                                          saturation_boost=SATURATION_BOOST, min_saturation=MIN_SATURATION, min_value=MIN_VALUE).ravel())
            flat_rect = Rectangle(width=1.1, height=0.03)  # tweak height for your "line" thickness
            flat_rect.set_fill(boosted_color, opacity=1)
            flat_rect.set_stroke(width=0)
            flat_rect.move_to([-1.5, 3.15-i*vertical_spacing, 0])  

            color_square = Square(side_length=patch_size)
            color_square.set_fill(rgb_to_color(overhead_colors[patch_index+32]/255.), opacity=1)
            color_square.set_stroke(width=0)
            color_square.move_to(pixel_squares[0][patch_index])

            embedding_rows_1.add(flat_rect)
            starting_squares_1.add(color_square)


        self.wait()
        self.play(
            LaggedStart(
                *[Succession(
                    FadeIn(starting_squares_1[i], run_time=0.1),
                    ReplacementTransform(starting_squares_1[i], embedding_rows_1[i]),
                ) for i in range(len(embedding_rows_1))],
                lag_ratio=0.2,
            ),
            run_time=12
        )

        ##Ok now a little ...
        ellipsis_dots = VGroup(*[
            Dot(radius=0.025).set_color(CHILL_BROWN)
            for _ in range(3)
        ])
        ellipsis_dots.arrange(DOWN, buff=0.035)
        ellipsis_dots.next_to(embedding_rows_1[-1], DOWN, buff=0.15)

        self.play(Write(ellipsis_dots), run_time=2)
        self.wait()

        #Pan down, or maybe just out and do it again as VO talks about colors. 
        #Hmm seems lke we're grabbing the wrong colors...

        patches_indices_to_move_2=[82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
        embedding_rows_2=VGroup()
        starting_squares_2=VGroup()
        vertical_spacing=0.2
        for i, patch_index in enumerate(patches_indices_to_move_2):

            boosted_color=rgb_to_color(boost_colors_hsv(left_colors[patch_index+32].reshape(1,3)/255., 
                                          saturation_boost=SATURATION_BOOST, min_saturation=MIN_SATURATION, min_value=MIN_VALUE).ravel())
            
            # boosted_color=rgb_to_color(left_colors[patch_index+32]/255.)
            flat_rect = Rectangle(width=1.1, height=0.03)  # tweak height for your "line" thickness
            flat_rect.set_fill(boosted_color, opacity=1)
            flat_rect.set_stroke(width=0)
            flat_rect.move_to([-1.5, 1.0-i*vertical_spacing, 0])  

            color_square = Square(side_length=patch_size)
            color_square.set_fill(rgb_to_color(left_colors[patch_index+32]/255.), opacity=1)
            color_square.set_stroke(width=0)
            color_square.move_to(pixel_squares[1][patch_index])

            embedding_rows_2.add(flat_rect)
            starting_squares_2.add(color_square)

        self.wait()
        self.play(
            self.frame.animate.reorient(0, 0, 0, (-2.39, 1.34, 0.0), 5.19),
            LaggedStart(
                *[Succession(
                    FadeIn(starting_squares_2[i], run_time=0.1),
                    ReplacementTransform(starting_squares_2[i], embedding_rows_2[i]),
                ) for i in range(len(embedding_rows_2))],
                lag_ratio=0.3,
            ),
            run_time=12
        )

        ##Ok now a little ...
        ellipsis_dots_2 = VGroup(*[
            Dot(radius=0.025).set_color(CHILL_BROWN)
            for _ in range(3)
        ])
        ellipsis_dots_2.arrange(DOWN, buff=0.035)
        ellipsis_dots_2.next_to(embedding_rows_2[-1], DOWN, buff=0.1)

        self.play(Write(ellipsis_dots_2), run_time=2)
        # self.wait()

        # This will probably liine up with p36. 
        # Ok now the final batch!
        # Could zoom then move, but I think doing them both at 
        # once will be a little better?
        patches_indices_to_move_3=[186, 186, 187, 188, 189, 190, 191]
        embedding_rows_3=VGroup()
        starting_squares_3=VGroup()
        vertical_spacing=0.2
        for i, patch_index in enumerate(patches_indices_to_move_3):

            boosted_color=rgb_to_color(boost_colors_hsv(right_colors[patch_index+32].reshape(1,3)/255., 
                                          saturation_boost=SATURATION_BOOST, min_saturation=MIN_SATURATION, min_value=MIN_VALUE).ravel())
            
            # boosted_color=rgb_to_color(right_colors[patch_index+32]/255.)
            flat_rect = Rectangle(width=1.1, height=0.03)  # tweak height for your "line" thickness
            flat_rect.set_fill(boosted_color, opacity=1)
            flat_rect.set_stroke(width=0)
            flat_rect.move_to([-1.5, -1.3-i*vertical_spacing, 0])  

            color_square = Square(side_length=patch_size)
            color_square.set_fill(rgb_to_color(right_colors[patch_index+32]/255.), opacity=1)
            color_square.set_stroke(width=0)
            color_square.move_to(pixel_squares[2][patch_index])

            embedding_rows_3.add(flat_rect)
            starting_squares_3.add(color_square)

        self.wait()
        self.play(
            self.frame.animate.reorient(0, 0, 0, (0, 0, 0.0), 8.0),
            LaggedStart(
                *[Succession(
                    FadeIn(starting_squares_3[i], run_time=0.1),
                    ReplacementTransform(starting_squares_3[i], embedding_rows_3[i]),
                ) for i in range(len(embedding_rows_3))],
                lag_ratio=0.3,
            ),
            run_time=10
        )        

        # Ok now P37 is on to the text prompt!
        # So I think we lost the quotes, turn it blue, break into 
        # 4 tokens. Maybe make it a little smaller
        # Expand the big matrix
        # Add the blue line, and have each token turn into each vector!


        embedding_brackets_2=all_svgs[7][2:8].shift([0.08, 0.00,0 ])
        blue_text_embedding_arrow=all_svgs[7][:2].shift([0.08, 0.00,0 ])
        embedding_exit_arrow=all_svgs[7][8:].shift([0.08, 0.00,0 ])

        # Some noddling to get stuff to match bro
        # self.remove(embedding_brackets_2)

        # len(embedding_brackets_2)
        # self.remove(embedding_brackets_2[0]) #Right
        # self.remove(embedding_brackets_2[1]) #Upper Left
        # self.remove(embedding_brackets_2[2]) #bottom left

        # self.remove(embedding_brackets_1[0]) #Right
        # self.remove(embedding_brackets_1[7]) #Bottom left
        # self.remove(embedding_brackets_1[8]) #Upper left
        embedding_brackets_1_only=VGroup(*[embedding_brackets_1[i] for i in [0, 8, 7, 9, 10, 11]])

        # self.remove(embedding_brackets_1_only)
        # self.remove(embedding_brackets_1[])

        # Let me go ahead and get these 4 lines in place, then I'll 
        # figure out how to animate to dems

        embedding_rows_4=VGroup() #Text ones
        for i in range(4):
            # flat_rect = Rectangle(width=1.1, height=0.03)  # tweak height for your "line" thickness
            # flat_rect.set_fill(BLUE, opacity=1)
            # flat_rect.set_stroke(width=0)
            # flat_rect.move_to([-1.5, -2.75-i*vertical_spacing, 0])  

            flat_line = Line(LEFT * 0.55, RIGHT * 0.55)
            flat_line.set_stroke(BLUE, width=4)
            flat_line.move_to([-1.5, -2.75 - i * vertical_spacing, 0])

            embedding_rows_4.add(flat_line)

        tokenized_prompt=Text('Un  cap  the  pen', font="Myriad Pro", weight='bold', font_size=25)
        tokenized_prompt.set_color(BLUE)
        tokenized_prompt.set_stroke(BLUE, width=0.1)
        tokenized_prompt.move_to(prompt)


        self.wait()

        self.play(self.frame.animate.reorient(0, 0, 0, (-2.87, -1.71, 0.0), 4.66), run_time=3)
        
        #Break apart
        self.wait()
        self.remove(prompt[0], prompt[-1])
        self.play(ReplacementTransform(prompt[1:3], tokenized_prompt[:2]),
                    ReplacementTransform(prompt[3:6], tokenized_prompt[2:5]),
                    ReplacementTransform(prompt[6:9], tokenized_prompt[5:8]),
                    ReplacementTransform(prompt[9:12], tokenized_prompt[8:11]), 
                  run_time=2.5)
        self.wait()


        self.wait()
        # self.play(ReplacementTransform(embedding_brackets_1_only, embedding_brackets_2), 
        #          ReplacementTransform(tokenized_prompt[:2].copy(), embedding_rows_4[0]),
        #          ReplacementTransform(tokenized_prompt[2:4].copy(), embedding_rows_4[1]),
        #          ReplacementTransform(tokenized_prompt[5:8].copy(), embedding_rows_4[2]),
        #          ReplacementTransform(tokenized_prompt[8:11].copy(), embedding_rows_4[3]),
        #           run_time=4)

        self.play(
            ReplacementTransform(embedding_brackets_1_only, embedding_brackets_2),
            LaggedStart(
                ReplacementTransform(tokenized_prompt[:2].copy(), embedding_rows_4[0]),
                ReplacementTransform(tokenized_prompt[2:4].copy(), embedding_rows_4[1]),
                ReplacementTransform(tokenized_prompt[5:8].copy(), embedding_rows_4[2]),
                ReplacementTransform(tokenized_prompt[8:11].copy(), embedding_rows_4[3]),
                lag_ratio=0.5,
            ),
            run_time=5
        )

        blue_text_embedding_arrow.set_color(BLUE)
        blue_text_embedding_arrow.shift([-0.1, 0.05, 0])

        self.wait()
        self.play(Write(blue_text_embedding_arrow))

        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (0, 0, 0), 8),
                  run_time=4)

        embedding_out_arrow=all_svgs[7][-2:]
        simple_llm_box=all_svgs[4]
        embedding_out_arrow.shift([-0.13, 0.18, 0])
   

        full_gemma=Group(all_svgs[8], all_svgs[9], all_svgs[10], all_svgs[11], all_svgs[12], all_svgs[13])
        full_gemma.shift([0.2, 0, 0])


        # Ok, so I think the vibe is zoom and transform the outer box?
        # Then we can fill stuff in!

        # Zoom in, expand LLM box
        self.wait()
        self.play(ReplacementTransform(simple_llm_box[-1], all_svgs[8][0]),
                  ReplacementTransform(simple_llm_box[0:3], all_svgs[8][1:4]),
                  ReplacementTransform(simple_llm_box[3:-1], all_svgs[8][4:]),
                  FadeIn(embedding_out_arrow),
                  self.frame.animate.reorient(0, 0, 0, (3.08, 0.07, 0.0), 4.73),
                 run_time=5)

        self.wait()
        self.play(Write(all_svgs[9]), Write(all_svgs[13]), run_time=3)
        # self,add(all_svgs[12])
        self.wait()

        self.play(Write(all_svgs[10]), Write(all_svgs[11]), Write(all_svgs[12]), run_time=7)
        

        # Hmm do we "zoom out" before "zooming in" on the attention head?
        # Hmm ok this might be annoying lol, but what about fading out 
        # everything except the attention heads, then fade out all but one
        # Then I can do a nice zoom out and expand the attention head at the 
        # same time deal?

        self.wait()
        self.remove(pi0_logo)
        self.play(FadeOut(all_svgs[8]), 
                  FadeOut(all_svgs[9]), 
                  FadeOut(all_svgs[10]), 
                  FadeOut(all_svgs[13]), 
                  run_time=3)
        
 
        # Now blow up a single attention head box as we zoom back out!
        # P40 Shorty
        h6_label=all_svgs[12][1:]
        self.wait()
        self.play(FadeOut(all_svgs[11]), run_time=1.5)
        # self.remove(all_svgs[11]) #Probably remove here instead of fade?
        self.play(ReplacementTransform(all_svgs[12][0], all_svgs[14][-1]),
                  h6_label.animate.scale(1.5).move_to([6.4, -3.4, 0]),
                  self.frame.animate.reorient(0, 0, 0, (0, 0, 0), 8),
                  run_time=5)


        queries=Group(); keys=Group(); values=Group(); 
        attn_dots=VGroup()
        q_spacing=0.15
        for i in range(11):
            q=ImageMobject(str(hacking_dir/('p40_1/queries_'+ str(i).zfill(2) +'.png')))
            q.scale(0.022)
            q.move_to([3.5, 3.2-i*q_spacing, 0])
            queries.add(q)
            if i==1 or i==6:
                e=VGroup(*[Dot(radius=0.01).set_color(CHILL_BROWN) for _ in range(3)])
                e.arrange(DOWN, buff=0.012)
                e.move_to(q)
                attn_dots.add(e)

        for i in range(11):
            k=ImageMobject(str(hacking_dir/('p40_1/keys_'+ str(i).zfill(2) +'.png')))
            k.scale(0.022)
            k.move_to([3.5, 0.9-i*q_spacing, 0])
            keys.add(k)
            if i==1 or i==6:
                e=VGroup(*[Dot(radius=0.01).set_color(CHILL_BROWN) for _ in range(3)])
                e.arrange(DOWN, buff=0.012)
                e.move_to(k)
                attn_dots.add(e)

        for i in range(11):
            v=ImageMobject(str(hacking_dir/('p40_1/values_'+ str(i).zfill(2) +'.png')))
            v.scale(0.022)
            v.move_to([3.5, -1.37-i*q_spacing, 0])
            values.add(v)
            if i==1 or i==6:
                e=VGroup(*[Dot(radius=0.01).set_color(CHILL_BROWN) for _ in range(3)])
                e.arrange(DOWN, buff=0.012)
                e.move_to(v)
                attn_dots.add(e)

        all_svgs[17].shift([0.04, 0.03, 0])
        all_svgs[18].shift([0.04, 0.00, 0])
        all_svgs[19].shift([0.04, 0.00, 0])

        self.wait()
        self.play(Write(all_svgs[15]), run_time=5)

        self.wait()

        self.play(FadeIn(queries), FadeIn(all_svgs[17]), FadeIn(attn_dots[:2]), run_time=2)
        self.play(FadeIn(keys), FadeIn(all_svgs[18]), FadeIn(attn_dots[:2]), run_time=2)
        self.play(FadeIn(values), FadeIn(all_svgs[19]), FadeIn(attn_dots[:2]), run_time=2)


        all_svgs[20].scale(1.015)
        all_svgs[20].shift([0.2, 0.03, 0])

        self.wait()
        self.play(Write(all_svgs[20]))

        #Ok for the middle of this paragraph we’ll rely on some illustrator overlays to call out connections etc → I think that will work better than animation? 
        #And we’ll do one zoom in/out when we talk about the light/dark regions of the vectors.

        self.remove(all_svgs[20])
        self.play(self.frame.animate.reorient(0, 0, 0, (3.47, 2.5, 0.0), 2.97), run_time=5)

        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (0, 0, 0), 8), run_time=4)

    
        self.wait()

        self.remove(attn_dots); self.add(attn_dots) #Occluserns


        #P41 
        #Slow zoom on Queries and keys
        self.play(self.frame.animate.reorient(0, 0, 0, (3.63, 1.2, 0.0), 4.37), 
                 FadeOut(all_svgs[14][-1]),#Attention head boarder
                 run_time=10) 


        self.wait()

        #Reduce opacity on all but final query row
        self.play(Group(*[queries[i] for i in [0, 2, 3, 4, 5, 7, 8, 9]]).animate.set_opacity(0.35), run_time=3)


        #Move copies on top of every key, adding numerical results -> i thnk one at a time. 

        # Add a column of Dot product results to the right of the keys matrix:
        # -27.49

        # -48.69
        #  36.30
        #  41.04
        # -42.65

        # -19.15
        # -19.71
        # -29.76
        # -43.39

        # Title on top of this column "DOT PRODUCT" with a line under it. Dot product is myriad pro, numbers are standard font

        # Dot product column
        dot_product_values = {
            0: "-27.49",
            2: "-48.69",
            3: " 36.30",
            4: " 41.04",
            5: "-42.65",
            7: "-19.15",
            8: "-19.71",
            9: "-29.76",
            10: "-43.39",
        }

        dp_x = 6.3  # tweak to sit right of keys

        dp_title = Text("DOT PRODUCT", font="Myriad Pro", weight='bold', font_size=14)
        dp_title.set_color(FRESH_TAN)
        dp_title.move_to([dp_x, 0.9 + 0.22, 0])  # just above first key row

        dp_underline = Line(
            dp_title.get_left() + DOWN * 0.08,
            dp_title.get_right() + DOWN * 0.08,
        )
        dp_underline.set_stroke(FRESH_TAN, width=2)

        dp_numbers = VGroup()
        for i in range(11):
            y = 0.9 - i * q_spacing
            if i in dot_product_values:
                num = Text(dot_product_values[i], font_size=14)
                num.set_color(MAGENTA)
                num.move_to([dp_x, y, 0])
                dp_numbers.add(num)

        # self.add(dp_title, dp_underline, dp_numbers)

        self.play(Write(dp_title), ShowCreation(dp_underline), run_time=1.5)

        self.wait()

        query_copy=queries[-1].copy()
        query_copy.set_opacity(0.8)
        self.add(query_copy)

        self.play(query_copy.animate.move_to(keys[0]), run_time=3)
        self.play(Write(dp_numbers[0]))
        self.play(FadeOut(query_copy))

        animations=[]
        query_copies=Group()
        for count, i in enumerate([2, 3, 4, 5, 7, 8, 9, 10]):
            query_copy=queries[-1].copy()
            query_copy.set_opacity(0.5)
            query_copies.add(query_copy)
            animations.append(query_copy.animate.move_to(keys[i]))
            animations.append(Write(dp_numbers[count+1]))

        self.add(query_copies)
        #ok this is pretty good now gotta add lag ratio and drawing in results. 
        self.play(LaggedStart(*animations, lag_ratio=0.4), run_time=10)

        self.wait()
        self.play(FadeOut(query_copies), run_time=3)

        #P42 - Will need a little Illustrator overlay here to point to those two large results - no problem

        attn_values = {
            0: "0.000",
            2: "0.000",
            3: "0.030",
            4: "0.041",
            5: "0.001",
            7: "0.001",
            8: "0.001",
            9: "0.000",
            10: "0.000",
        }

        attn_x = 7.5  #

        attn_title = Text("ATTENTION VALUE", font="Myriad Pro", weight='bold', font_size=14)
        attn_title.set_color(FRESH_TAN)
        attn_title.move_to([attn_x, 0.9 + 0.22, 0])  # just above first key row

        attn_underline = Line(
            attn_title.get_left() + DOWN * 0.08,
            attn_title.get_right() + DOWN * 0.08,
        )
        attn_underline.set_stroke(FRESH_TAN, width=2)

        attn_numbers = VGroup()
        for i in range(11):
            y = 0.9 - i * q_spacing
            if i in attn_values:
                num = Text(attn_values[i], font_size=14)
                num.set_color(MAGENTA)
                num.move_to([attn_x, y, 0])
                attn_numbers.add(num)

        all_svgs[21].scale(0.8) #Softmax arrow
        all_svgs[21].move_to([6.93, 0.1, 0])

        self.wait()
        self.add(all_svgs[21])
        self.play(self.frame.animate.reorient(0, 0, 0, (4.56, 1.1, 0.0), 4.22), 
                  Write(attn_title), 
                  ShowCreation(attn_underline), 
                  Write(attn_numbers),
            )
        self.remove(h6_label) #Probably add back later when we do the big zoom out. 

        self.wait()


        # Ok mid P42 now, this on is going to be a little tricky
        # So we're building up to bringing the three tiled images
        # front and center, for that animation, we're probably not rendering
        # in manim right? It will be maptotlib, probably like very close
        # to the demo I already rendered?
        # Ok ok ok ok ok ok ok ok ok ok ok ok ok ok 
        # So I think the move is to pretend like we're adding a magenta layer
        # on top of the patches in manim, but like actually just do a switcheroo 
        # kidna thing? baking in the magenta values from matplotlib into a 
        # Different set of image patches? It also might not be terrible, form that 
        # perspective to actually play the animation in matplotlib, 
        # just would be resampling the cached colored patches in a loop
        # let's do that as plan A, and fall back if that's problematic. 
        # Ok lol scratch that, let's try this shit in manim bro. 

        all_attn_values=np.load(hacking_dir/'p42_1/p42_1.npy')

        attn_row=all_attn_values[FRAME_IDX] #816
        cam_attn_1=attn_row[:256].reshape(16,16) #Overhead
        cam_attn_2=attn_row[256:512].reshape(16,16) #Left
        cam_attn_3=attn_row[512:768].reshape(16,16) #Right

        # Ok Claude, I got a good one for you. I want to create a magenta square
        # on top of each image patch, with opacity controlled by cam_attn_1, cam_attn_2, etc.
        # The first and last 32 attention values should be ignored.
        # We may need to futz with scaling, let's make this parameterizable, 
        # with one param for each image, with a default of the max attention value for 
        # a given image, resulting in an opacity of 0.9. 

        # Per-image scaling: attention value that maps to opacity 0.9
        cam_attns = [cam_attn_1, cam_attn_2, cam_attn_3]

        attn_scales = [cam_attn_1[2:14].max(), cam_attn_2[2:14].max(), cam_attn_3[2:14].max()]
        max_opacities = [0.5, 0.95, 0.95]  # per-image max opacity

        magenta_overlays = Group()
        for k in range(3):
            cam = cam_attns[k][2:14, :]  # (12, 16)
            scale = attn_scales[k] if attn_scales[k] > 0 else 1.0
            mo = max_opacities[k]
            overlays_k = Group()
            for idx, patch_mob in enumerate(pixel_squares[k]):
                row = idx // 16
                col = idx % 16
                attn_val = cam[row, col]
                opacity = min(mo, mo * (attn_val / scale))

                sq = Square(side_length=patch_size)
                sq.set_fill(MAGENTA, opacity=float(opacity))
                sq.set_stroke(width=0)
                sq.move_to(patch_mob.get_center())
                overlays_k.add(sq)
            magenta_overlays.add(overlays_k)


        #Ok this looks great! Now, how do I animate these in/numbers comign over?
        self.wait()
        self.play(ReplacementTransform(attn_numbers[0].copy(), magenta_overlays[0][0].set_opacity(0.2)), 
                  ReplacementTransform(attn_numbers[1].copy(), magenta_overlays[1][85]), #.copy().set_opacity(0.5)), 
                  ReplacementTransform(attn_numbers[2].copy(), magenta_overlays[1][86]), #.copy().set_opacity(0.5)), 
                  ReplacementTransform(attn_numbers[3].copy(), magenta_overlays[1][87]), #.copy().set_opacity(0.5)), 
                  ReplacementTransform(attn_numbers[4].copy(), magenta_overlays[1][88]), #.copy().set_opacity(0.5)), 
                  self.frame.animate.reorient(0, 0, 0, (0.67, 0.02, 0.0), 8.60),
                  FadeIn(magenta_overlays[0][1:]), 
                  FadeIn(magenta_overlays[1][:85]), 
                  FadeIn(magenta_overlays[1][89:]), 
                  FadeIn(magenta_overlays[2]), 
                  run_time=5)
        # self.play(FadeIn(magenta_overlays[0][1:]), 
        #           FadeIn(magenta_overlays[1][:85]), 
        #           FadeIn(magenta_overlays[1][89:]), 
        #           FadeIn(magenta_overlays[2]), 
        #           run_time=3)

        # P43 Ok this next bit is going to be tricky, especially cleanup, but I think 
        # Claude can help!
        

        # self.add(magenta_overlays)
        # self.remove(magenta_overlays)

        self.wait()


        self.wait(20)
        self.embed()
































