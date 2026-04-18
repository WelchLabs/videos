from manimlib import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import colorsys
# import gc
# import resource
import shutil
import tempfile

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
BLUE2='#00AEEF'
GREEN='#00a14b' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'
PINK='#FAD0E2'

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/vla/graphics/to_manim/')
hacking_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/vla/hackin')

# Might need to run this to allow for opening enough files
# before rendering:
# ulimit -n 4096

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



class poster_1(InteractiveScene):
    def construct(self): 
        '''
        Clean break/pickup from P49.
        '''

        # ── Load SVGs (same as P31_49) ──
        svgs_to_skip=[0, 2, 3, 8, 12, 13, 20, 21]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(3.9)
            all_svgs.add(svg_image[1:])

        # ── Set frame to end-of-P49 state ──
        self.frame.reorient(0, 0, 0, (-2.03, -6.66, 0.0), 5.33)

        # ── Constants ──
        FRAME_IDX = 150
        total_height = 2.72
        grid_n = 16
        patch_size = total_height / grid_n
        gap_factor = 0.12
        vertical_spacing = 0.2

        # ═══════════════════════════════════════════════════
        # 1. IMAGE PATCHES (with gap expansion baked in)
        # ═══════════════════════════════════════════════════
        patch_centers = [
            [-5.23, 2.58, 0],      # overhead (no shift)
            [-5.23, 0.175, 0],     # left wrist (0.375 - 0.2 shift)
            [-5.23, -2.22, 0],     # right wrist (-1.82 - 0.4 shift)
        ]
        pixel_squares = Group()
        for k, image_name in enumerate(['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']):
            pixel_squares.add(Group())
            patch_dir = hacking_dir / f'p35/{FRAME_IDX}/{image_name}'
            for i in range(2, 14):
                for j in range(grid_n):
                    patch_mob = ImageMobject(str(patch_dir / f'patch_{i}_{j}.png'))
                    patch_mob.set_height(patch_size)
                    patch_mob.set_width(patch_size, stretch=True)
                    # Bake in gap expansion: offset * (1 + gap_factor)
                    x_pos = (j - grid_n/2 + 0.5) * patch_size * (1 + gap_factor)
                    y_pos = -(i - grid_n/2 + 0.5) * patch_size * (1 + gap_factor)
                    patch_mob.move_to([x_pos, y_pos, 0])
                    pixel_squares[-1].add(patch_mob)
            pixel_squares[k].move_to(patch_centers[k])

        # ═══════════════════════════════════════════════════
        # 2. SIGLIP ENCODERS + LIL ARROWS
        # ═══════════════════════════════════════════════════
        siglip_1 = all_svgs[2][:13]
        siglip_2 = all_svgs[2][13:26]
        siglip_3 = all_svgs[2][26:39]
        image_encoders_label = all_svgs[2][39:]

        siglip_1.scale(1.1).move_to([-3.0, 2.6, 0])
        siglip_2.scale(1.1).move_to([-3.0, 0.2, 0])
        siglip_3.scale(1.1).move_to([-3.0, -2.1, 0])
        image_encoders_label.scale(1.1).move_to([-3.0, 3.5, 0])

        lil_arrows_pair_1 = all_svgs[5]
        lil_arrows_pair_2 = lil_arrows_pair_1.copy()
        lil_arrows_pair_3 = lil_arrows_pair_1.copy()
        lil_arrows_pair_1.move_to([-3.0, 2.57, 0])
        lil_arrows_pair_2.move_to([-3.0, 0.18, 0])
        lil_arrows_pair_3.move_to([-3.0, -2.18, 0])

        # ═══════════════════════════════════════════════════
        # 3. EMBEDDING BRACKETS + ARROWS
        # ═══════════════════════════════════════════════════
        # all_svgs[6] remnants (indices that weren't ReplacementTransformed away)
        all_svgs[6].shift([0.08, 0, 0])
        bracket_remnants = all_svgs[6][1:7]

        # all_svgs[7] splits
        embedding_brackets_2 = all_svgs[7][2:8]
        embedding_brackets_2.shift([0.08, 0, 0])

        blue_text_embedding_arrow = all_svgs[7][:2]
        blue_text_embedding_arrow.shift([0.08, 0, 0])
        blue_text_embedding_arrow.shift([-0.1, 0.05, 0])
        blue_text_embedding_arrow.set_color(BLUE)

        embedding_out_arrow = all_svgs[7][-2:]
        # embedding_out_arrow.shift([-0.13, 0.18, 0])
        embedding_out_arrow.move_to([-0.64, 0.28, 0])

        # ═══════════════════════════════════════════════════
        # 4. EMBEDDING ROWS (colored barcode lines)
        # ═══════════════════════════════════════════════════
        overhead_im = np.load(hacking_dir / 'p35/150_overhead.npy')
        left_im = np.load(hacking_dir / 'p35/150_left.npy')
        right_im = np.load(hacking_dir / 'p35/150_right.npy')

        overhead_colors = patch_bright_average(overhead_im, exponent=2.0).reshape(-1, 3)
        left_colors = patch_bright_average(left_im, exponent=2.0).reshape(-1, 3)
        right_colors = patch_bright_average(right_im, exponent=2.0).reshape(-1, 3)

        def make_embedding_row(color_arr, patch_index, y_pos):
            bc = rgb_to_color(boost_colors_hsv(
                color_arr[patch_index + 32].reshape(1, 3) / 255.,
                saturation_boost=SATURATION_BOOST, min_saturation=MIN_SATURATION, min_value=MIN_VALUE).ravel())
            r = Rectangle(width=1.1, height=0.03)
            r.set_fill(bc, opacity=1).set_stroke(width=0)
            r.move_to([-1.5, y_pos, 0])
            return r

        # Overhead embeddings
        embedding_rows_1 = VGroup()
        for i, pi in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8]):
            embedding_rows_1.add(make_embedding_row(overhead_colors, pi, 3.15 - i * vertical_spacing))

        ellipsis_dots = VGroup(*[Dot(radius=0.025).set_color(CHILL_BROWN) for _ in range(3)])
        ellipsis_dots.arrange(DOWN, buff=0.035)
        ellipsis_dots.next_to(embedding_rows_1[-1], DOWN, buff=0.15)

        # Left wrist embeddings
        embedding_rows_2 = VGroup()
        for i, pi in enumerate([82, 83, 84, 85, 86, 87, 88, 89, 90, 91]):
            embedding_rows_2.add(make_embedding_row(left_colors, pi, 1.0 - i * vertical_spacing))

        ellipsis_dots_2 = VGroup(*[Dot(radius=0.025).set_color(CHILL_BROWN) for _ in range(3)])
        ellipsis_dots_2.arrange(DOWN, buff=0.035)
        ellipsis_dots_2.next_to(embedding_rows_2[-1], DOWN, buff=0.1)

        # Right wrist embeddings
        embedding_rows_3 = VGroup()
        for i, pi in enumerate([186, 186, 187, 188, 189, 190, 191]):
            embedding_rows_3.add(make_embedding_row(right_colors, pi, -1.3 - i * vertical_spacing))

        # Text token embeddings (blue)
        embedding_rows_4 = VGroup()
        for i in range(4):
            l = Line(LEFT * 0.55, RIGHT * 0.55)
            l.set_stroke(BLUE, width=4)
            l.move_to([-1.5, -2.75 - i * vertical_spacing, 0])
            embedding_rows_4.add(l)

        # ═══════════════════════════════════════════════════
        # 5. TOKENIZED PROMPT
        # ═══════════════════════════════════════════════════
        tokenized_prompt = Text('Un  cap  the  pen', font="Myriad Pro", weight='bold', font_size=25)
        tokenized_prompt.set_color(BLUE)
        tokenized_prompt.set_stroke(BLUE, width=0.1)
        # Position: started at [-5.5, -3.34], then shifted [0.2, -0.3]
        tokenized_prompt.move_to([-5.3, -3.64, 0])

        # ═══════════════════════════════════════════════════
        # 6. GEMMA NETWORK
        # ═══════════════════════════════════════════════════
        # Apply the [0.2, 0, 0] shift to all gemma internals (incl hidden ones)
        full_gemma = Group(all_svgs[8], all_svgs[9], all_svgs[10], 
                           all_svgs[11], all_svgs[12], all_svgs[13])
        full_gemma.shift([0.2, 0, 0])
        full_gemma_copy=Group(all_svgs[8].copy(), all_svgs[9].copy(), all_svgs[10].copy(), 
                           all_svgs[45], all_svgs[46], all_svgs[13].copy())
        all_svgs[26].shift([0.19, 0, 0])

        # Only these are visible at end of P49:
        gemma = Group(all_svgs[8], all_svgs[9], all_svgs[10], all_svgs[13], all_svgs[26])

        action_expert_full = full_gemma.copy()
        action_expert_full_copy = full_gemma_copy.copy()

        # ═══════════════════════════════════════════════════
        # 7. ATTENTION PATTERNS GRID
        # ═══════════════════════════════════════════════════
        all_attn_patterns = []
        for layer in range(18):
            all_attn_patterns.append([])
            for head in range(8):
                im = ImageMobject(str(hacking_dir / f'p47/attn_pattern_{layer}_{head}.png'))
                all_attn_patterns[-1].append(im)

        attn_patterns_to_show = Group()
        for i in range(8):
            all_attn_patterns[0][i].scale(0.096).move_to([0.25, 1.5 - 0.358 * i, 0])
            attn_patterns_to_show.add(all_attn_patterns[0][i])
        for i in range(8):
            if i == 6: continue  # slot reserved for the zoomed-in pattern
            all_attn_patterns[1][i].scale(0.096).move_to([2.65, 1.5 - 0.358 * i, 0])
            attn_patterns_to_show.add(all_attn_patterns[1][i])
        for i in range(8):
            all_attn_patterns[-1][i].scale(0.096).move_to([5.05, 1.5 - 0.358 * i, 0])
            attn_patterns_to_show.add(all_attn_patterns[-1][i])

        # The single pattern that was shrunk into the layer-1/head-6 slot
        attn_pattern = ImageMobject(str(hacking_dir / 'p44/attn_pattern_1.png'))
        attn_pattern.scale(1.3 * 0.41 * 0.18)  # cumulative relative scales
        attn_pattern.move_to([2.65, 1.5 - 0.358 * 6, 0])

        # ═══════════════════════════════════════════════════
        # 8. ACTION EXPERT AREA (bottom of canvas)
        # ═══════════════════════════════════════════════════
        all_svgs[27].move_to([-5.2, -6, 0])
        all_svgs[28].move_to([-5.2, -6, 0])
        all_svgs[29].move_to([-3.05, -4.7, 0])
        all_svgs[30].move_to([-1.53, -6.7, 0])
        all_svgs[31].move_to([-4.25, -6.765, 0])

        arm_img = ImageMobject(str(hacking_dir / 'arm_1.png'))
        arm_img.scale(0.52).move_to([-4.6, -5.97, 0])
        arm_img_flipped = ImageMobject(str(hacking_dir / 'arm_1_flipped.png'))
        arm_img_flipped.scale(0.52).move_to([-5.78, -5.97, 0])

        # Action expert embedding rows (red/pink)
        embedding_rows_action_expert = VGroup()
        for i in range(20):
            if i == 9:
                dots = VGroup(*[Dot(radius=0.02).set_color(CHILL_BROWN) for _ in range(3)])
                dots.arrange(DOWN, buff=0.03)
                dots.next_to(embedding_rows_action_expert[-1], DOWN, buff=0.09)
                embedding_rows_action_expert.add(dots)
                continue
            r = Rectangle(width=1.1, height=0.03)
            r.set_fill(RED if i == 0 else PINK, opacity=1)
            r.set_stroke(width=0)
            r.move_to([-1.5, -4.7 - i * vertical_spacing, 0])
            embedding_rows_action_expert.add(r)

        # ═══════════════════════════════════════════════════
        # ADD EVERYTHING TO SCENE
        # ═══════════════════════════════════════════════════
        self.add(
            # Image patches
            pixel_squares,
            # SigLIP encoders + arrows
            siglip_1, siglip_2, siglip_3, image_encoders_label,
            lil_arrows_pair_1, lil_arrows_pair_2, lil_arrows_pair_3,
            # Embedding column
            bracket_remnants, embedding_brackets_2,
            blue_text_embedding_arrow, embedding_out_arrow,
            embedding_rows_1, ellipsis_dots,
            embedding_rows_2, ellipsis_dots_2,
            embedding_rows_3, embedding_rows_4,
            # Prompt
            tokenized_prompt,
            # Gemma network
            gemma,
            # Attention patterns
            attn_patterns_to_show, attn_pattern,
            # Action expert area
            all_svgs[27], all_svgs[28], all_svgs[29], all_svgs[30], all_svgs[31],
            arm_img, arm_img_flipped,
            embedding_rows_action_expert,
        )
        # Occlusion fix: gemma connector labels on top
        self.remove(all_svgs[26]); self.add(all_svgs[26])

        # self.add(all_svgs[28])

        # Ok wow I think i can pickup here pretty cleanly! Dope. 
        # P53

        # Ok let me load up the action images.
        diffusion_images=Group()
        for i in range(11):
            im=ImageMobject(str(hacking_dir/('p51b/'+str(i).zfill(2)+'.png')))
            diffusion_images.add(im)


        action_expert_box = RoundedRectangle(
            width=1.9,
            height=1.5,
            corner_radius=0.1,
            stroke_color=CHILL_BROWN,
            stroke_width=2,
            fill_opacity=0,
        )

        manual_action_expert_box=all_svgs[4][-1].copy()
        manual_action_expert_box.scale(0.6)
        manual_action_expert_box.set_color(CHILL_BROWN)
        # manual_action_expert_box.move_to(action_expert_label)

        # self.add(manual_action_expert_box)

        action_expert_label=Text('ACTION EXPERT', font="Myriad Pro", weight='bold', font_size=22)
        action_expert_label.set_color(CHILL_BROWN)
        action_expert_label.move_to(manual_action_expert_box)
        action_expert_label.shift([0, -0.05, 0])

        action_expert_box_group=Group(manual_action_expert_box, action_expert_label)


        action_expert_box_group.move_to([0.57, -6.45, 0])
        

        # self.add(all_svgs[31])
        all_svgs[33].scale(0.7)
        all_svgs[33].move_to([-4.2, -6.9, 0])

        all_svgs[34].move_to([0.85, -6.5, 0])
        lil_arrow_2=all_svgs[34][0]
        # lil_arrow_2b=lil_arrow_2.copy()
        lil_arrow_2b=all_svgs[34][1]
        lil_arrow_2b.move_to([1.75, -6.5, 0])
        
        diffusion_images.scale(0.19)
        diffusion_images[0].move_to([-5.18, -8.0, 0])
        diffusion_images[1].move_to([3.53, -6.5, 0])

        all_svgs[35].move_to([3.2, -6.75, 0])
        all_svgs[35][5:].move_to([3.5, -7.05, 0]) #Timestemp label 

        diffusion_images_copy=diffusion_images.copy()
        diffusion_images_copy_2=diffusion_images.copy()
        diffusion_images_copy_3=diffusion_images.copy()
        diffusion_images_copy_4=diffusion_images.copy()


        self.remove(all_svgs[31])
        self.add(action_expert_box_group)
        self.add(all_svgs[33])
        self.add(diffusion_images[0])

        self.add(lil_arrow_2, lil_arrow_2b)
        self.frame.reorient(0, 0, 0, (-0.88, -7.24, 0.0), 6.82)

        self.wait()

        self.play(FadeIn(diffusion_images[1]),
                  FadeIn(all_svgs[35]), 
                  run_time=3)

        self.wait()
        self.play(diffusion_images[1].animate.move_to(diffusion_images[0]),
                  run_time=4)

        # Hmmmmm this is annoying - realizing that I don't have the sum/inputs
        # to actually show here. I could sorta fake this, hmm would this be insane:
        diffusion_images[1].set_opacity(0.5)

        diffusion_images[2].move_to([3.53, -6.5, 0])
        self.play(FadeIn(diffusion_images[2]), run_time=2)

        self.play(diffusion_images[2].animate.move_to(diffusion_images[0]),
                  run_time=4)
        diffusion_images[2].set_opacity(0.2)
        self.remove(diffusion_images[2]); self.add(diffusion_images[2])

        diffusion_images[3].move_to([3.53, -6.5, 0])
        self.play(FadeIn(diffusion_images[3]), run_time=2)

        tmp_diffusion_images=Group()
        for i in range(3, 10):
            tmp_1=diffusion_images[i].copy()
            tmp_1.move_to([-5.18, -8.0, 0])
            tmp_diffusion_images.add(tmp_1)
            diffusion_images[i+1].move_to([3.53, -6.5, 0])
            self.add(tmp_1, diffusion_images[i+1])
            self.wait(0.5)

        # P58
        # ok so were' not actualy showing the sum on the left, 
        # I'm going to let that go for now. 
        # Hmm can't see all my notes right now on the plane, 
        # but it looks like we jump right to P58 here?
        # Ok yeah that's what illustrator is suggesting to. 
        # Ok so my intuition here is that I should do a zoom out from 
        # this current position, or something pretty close to it, 
        # Expand the action expert box etc. 
        # Now, I think I do need to "shift up" all the action expert stuff 
        # a bit, and I need to make the whole LLM bigger. 
        # That's a lot of work, but we hitting the "final ascent here!"

        # self.add(action_expert_group_1[4])

        action_expert_group_1=Group(all_svgs[27], all_svgs[28], all_svgs[29], all_svgs[30], #all_svgs[31],
                                    # all_svgs[32], 
                                    all_svgs[33], all_svgs[34], #all_svgs[35],
                                    arm_img, arm_img_flipped, 
                                    tmp_diffusion_images,
                                    embedding_rows_action_expert, 
                                    # action_expert_box_group, 
                                    # diffusion_images, 
                                    )
        

        # self.add(action_expert_group_1)
        gemma_group=Group(gemma, attn_patterns_to_show, attn_pattern)
        gemma_group.scale(1.5)
        gemma_group.move_to([5.1, 0.6, 0])

        # ok so we'll do a zoom out and expand the action expert kinda deal. 
        # Do I want to draw in the full action exprt network first, and then 
        # Figure out how to animate to it? That's probably the move? 
        # I think at this pahse we just want the h1/h2 etc attention head 
        # labels. Let's try adopting the blue and yellow borders now too
        # I think that helps with the visual overwhelm, and we'll see how it
        # feels. 

        all_svgs[8].set_color(BLUE2)

        # action_expert_full.add(all_svgs[36])
        action_expert_full.scale(1.5)
        action_expert_full.move_to([5.1, -5.4, 0])
        action_expert_full_2=Group(*[action_expert_full[i] for i in [1, 2, 3, 4]])

        action_expert_full_copy[3].shift([0.2, 0, 0])
        action_expert_full_copy[4].shift([0.2, 0, 0])
        action_expert_full_copy.scale(1.5)
        action_expert_full_copy.move_to([5.1, -5.4, 0])
        

        all_svgs[36].scale([1.5, 1.4, 1])
        all_svgs[36].move_to([5.1, -5.5, 0])

        
        final_actions_text=Text('ACTIONS', font="Myriad Pro", weight='bold', font_size=26)
        final_actions_text.set_color(PINK)
        # final_actions_text.next_to(diffusion_images[10], DOWN, buff=0.2)
        final_actions_text.move_to([12.4, -5.8, 0])

        # Hmm ok running into kinda the same problem from earlier
        # I need an svg exported rounded box with a tiny slit 
        # for the action expert border to make the transition smooth. 
        # Ok yeah so I think like svg[5]?


        # self.add(all_svgs[4][-1])
        # all_svgs[4].move_to([ 0.6, -6.5,  0. ])

        # self.remove(all_svgs[4])

        # action_expert_label.set_color(YELLOW)
        # action_expert_label.scale(1.2)
        # action_expert_label.move_to([4.95, -8.28, 0])

        self.remove(diffusion_images[:10]) #Cleanup
        # self.remove(tmp_diffusion_images[:8])

        #Hmm just need to figure out how to nudge temp diffion image on left - 
        # which one it is?

        self.wait()
        self.remove(all_svgs[35])
        self.play(ReplacementTransform(action_expert_box_group[0], all_svgs[36][0]), 
                  action_expert_label.animate.scale(1.25).set_color(YELLOW).move_to([4.95, -8.28, 0]),
                  action_expert_group_1.animate.shift([0, 0.5, 0]),  #Nudge  right next to LLM
                  self.frame.animate.reorient(0, 0, 0, (4.34, -2.34, 0.0), 13.00),
                  all_svgs[34][1].animate.move_to([10.8, -5.2, 0]), #Arrow from action expert to output
                  diffusion_images[10].animate.move_to([12.4, -5.2, 0]), #ouput difussion image
                  run_time=7)
        self.add(final_actions_text)


        self.wait()
        self.play(Write(action_expert_full_2[0]), 
                  Write(action_expert_full_2[1]),
                  Write(action_expert_full_2[2]),
                  Write(action_expert_full_2[3]),
                  run_time=6
                  )

        # Ok we're into p58 now
        # I think a zoom into action expert?
        # might cut this in the edit, but probably not a terrible option to have
        self.play(self.frame.animate.reorient(0, 0, 0, (5.13, -5.51, 0.0), 6.41), run_time=6)
        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (4.34, -2.34, 0.0), 13.00), run_time=6)
        self.wait()

        # Ok so now I need to zoom into a head on the LLM, like we did before in P31_49
        # and then zoom into a head on the action expert. Let me see if Claude can help on 
        # replicating the Gemma attention head zoom in we did in P31_49. 
       

        # h6_label=all_svgs[12][1:]
        # self.wait()

        # Hmm ok yeah a little tricky here - 
        # Gotta figure out which box to zoom in on 
        # And what all I need to fade out. 
        # Some kinda SYSIWYG + manim deal would be nice woudln't it?


        # gemma
        # all_attn_patterns
        # attn_pattern


        # Ok cool so this nicely leaves us with just gemma_h6_border
        # Alright let me work on adding the attention head
        # then will switch to the action expert!
        # Should be able to follow a pretty similar approach!
        # self.play(ReplacementTransform(all_svgs[12][0], all_svgs[14][-1]),
        #           h6_label.animate.scale(1.5).move_to([6.4, -3.4, 0]),
        #           self.frame.animate.reorient(0, 0, 0, (0, 0, 0), 8),
        #           run_time=5)


        queries=Group(); keys=Group(); values=Group(); 
        attn_dots=VGroup()
        q_spacing=0.135
        for i in range(11):
            q=ImageMobject(str(hacking_dir/('p40_1/queries_'+ str(i).zfill(2) +'.png')))
            q.scale(0.022)
            q.move_to([3.5, 2.8-i*q_spacing, 0])
            queries.add(q)
            if i==1 or i==6:
                e=VGroup(*[Dot(radius=0.01).set_color(CHILL_BROWN) for _ in range(3)])
                e.arrange(DOWN, buff=0.012)
                e.move_to(q)
                attn_dots.add(e)

        for i in range(11):
            k=ImageMobject(str(hacking_dir/('p40_1/keys_'+ str(i).zfill(2) +'.png')))
            k.scale(0.022)
            k.move_to([3.5, 1.0-i*q_spacing, 0])
            keys.add(k)
            if i==1 or i==6:
                e=VGroup(*[Dot(radius=0.01).set_color(CHILL_BROWN) for _ in range(3)])
                e.arrange(DOWN, buff=0.012)
                e.move_to(k)
                attn_dots.add(e)

        for i in range(11):
            v=ImageMobject(str(hacking_dir/('p40_1/values_'+ str(i).zfill(2) +'.png')))
            v.scale(0.022)
            v.move_to([3.5, -0.8-i*q_spacing, 0])
            values.add(v)
            if i==1 or i==6:
                e=VGroup(*[Dot(radius=0.01).set_color(CHILL_BROWN) for _ in range(3)])
                e.arrange(DOWN, buff=0.012)
                e.move_to(v)
                attn_dots.add(e)

        all_svgs[17].next_to(q, DOWN, buff=0.05)
        all_svgs[18].next_to(k, DOWN, buff=0.05)
        all_svgs[19].next_to(v, DOWN, buff=0.05)
        all_svgs[37].move_to([0.2, 0.3, 0])

        # self.add(all_svgs[37]) #WQ QK QV Arrows
        all_svgs[14].scale([0.95, 0.77, 1]) #Attention head border
        all_svgs[14].move_to([2.88, 0.3, 0]) 


        gemma_h6_border=gemma_group[0][4][50].copy()
        self.add(gemma_h6_border)

        # Hmm ok so I think if I'm going to show the QKV zoom in on 
        # Gemma first I should just zoom in - it's just so little!

        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (5.15, 0.71, 0.0), 7.01), 
                  run_time=4)
        self.remove(final_actions_text)

        gemma_attn_head_group=Group(all_svgs[37], queries, all_svgs[17], keys, all_svgs[18], 
              values, all_svgs[19], attn_dots)
        
        gemma_attn_head_group.scale(0.08)
        gemma_attn_head_group.move_to([4.25, -0.48, 0])
        # self.add(gemma_attn_head_group)


        self.wait()
        self.play(FadeOut(gemma_group), 
                  FadeIn(gemma_attn_head_group), run_time=3)

        self.play(ReplacementTransform(gemma_h6_border, all_svgs[14][-1]), 
                  gemma_attn_head_group.animate.scale(1.0/0.08).move_to([2.7000403 , 0.22694729, 0.]),
                  run_time=5)

        #Scooch arrow over before we zoom out and it's visible again. 
        lil_arrow_2.shift([-0.05, 0, 0])

        #Pan down to action expert
        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (5.26, -5.43, 0.0), 6.99),
                  run_time=4)


        # Ok so now we gotta to the same zoom in action onto 
        # a single attention head
        # Let me work on the box first, then fill in the full size
        # QKV deal, then figure out the zoom in step. 
        action_expert_h6_border=action_expert_full_2[3][0].copy()
        self.add(action_expert_h6_border)

        action_expert_h6_border_copy=action_expert_h6_border.copy()

        #Action exper model and border - swtich to fade out. 
        action_expert_attn_head_border=all_svgs[14][-1].copy()
        action_expert_attn_head_border.move_to([2.88, -5.5, 0])

        action_expert_qkv_flow=all_svgs[38]
        action_expert_qkv_flow.move_to([0.2, -5.5, 0])

        # So i think the cool move here is probably bringing in 
        # Flow arrows, queries, keys, and values separately
        # Ok cool so still gotta figure out transition 
        # but yeah let's go ahead and build out keys queries and values here
        # Then come back and figure out the transition!

        queries_ae=Group(); keys_ae=Group(); values_ae=Group(); 
        attn_dots_ae=VGroup()
        q_spacing=0.135
        for i in range(7):
            q=ImageMobject(str(hacking_dir/('p58/queries_'+ str(i).zfill(2) +'.png')))
            q.scale(0.022)
            q.move_to([3.5, -i*q_spacing, 0])
            queries_ae.add(q)
            if i==3:
                e=VGroup(*[Dot(radius=0.01).set_color(CHILL_BROWN) for _ in range(3)])
                e.arrange(DOWN, buff=0.012)
                e.move_to(q)
                attn_dots_ae.add(e)

        for i in range(7):
            k=ImageMobject(str(hacking_dir/('p58/keys_'+ str(i).zfill(2) +'.png')))
            k.scale(0.022)
            k.move_to([3.5, -i*q_spacing, 0])
            keys_ae.add(k)
            if i==3:
                e=VGroup(*[Dot(radius=0.01).set_color(CHILL_BROWN) for _ in range(3)])
                e.arrange(DOWN, buff=0.012)
                e.move_to(k)
                attn_dots_ae.add(e)

        for i in range(7):
            v=ImageMobject(str(hacking_dir/('p58/values_'+ str(i).zfill(2) +'.png')))
            v.scale(0.022)
            v.move_to([3.5, -i*q_spacing, 0])
            values_ae.add(v)
            if i==3:
                e=VGroup(*[Dot(radius=0.01).set_color(CHILL_BROWN) for _ in range(3)])
                e.arrange(DOWN, buff=0.012)
                e.move_to(v)
                attn_dots_ae.add(e)


        queries_ae.move_to([3.5, -3.7, 0])
        attn_dots_ae[0].move_to([3.5, -3.7, 0])
        keys_ae.move_to([3.5, -5.5, 0])
        attn_dots_ae[1].move_to([3.5, -5.5, 0])
        values_ae.move_to([3.5, -7.3, 0])
        attn_dots_ae[2].move_to([3.5, -7.3, 0])

        all_svgs[39].scale(0.88)
        all_svgs[39].move_to([3.5, -4.3, 0])
        all_svgs[40].scale(0.88)
        all_svgs[40].move_to([3.5, -6.1, 0])
        all_svgs[41].scale(0.88)
        all_svgs[41].move_to([3.5, -7.9, 0])
        all_svgs[42].scale(0.89)
        all_svgs[42].move_to([6.08, -5.5, 0])

        # Ah yeah I want the little baby version of 
        # the attention head again. 

        ae_attn_head_group=Group(action_expert_qkv_flow, 
                                 # action_expert_attn_head_border,
                                 queries_ae, 
                                 keys_ae, 
                                 values_ae, 
                                 attn_dots_ae,
                                 all_svgs[39],
                                 all_svgs[40],
                                 all_svgs[41]
                                 )

        self.wait()

        #[ 2.7000403 ,-5.6124,  0.        ]

        # self.add(ae_attn_head_group)
        # self.add(attn_dots_ae)


        ae_attn_head_group.scale(0.08)
        ae_attn_head_group.move_to([4.3, -6.38, 0])

        self.wait()
        self.play(FadeOut(action_expert_full_2), 
                  FadeOut(action_expert_label),
                  FadeOut(all_svgs[36][0]),
                  FadeOut(all_svgs[34][1]),
                  FadeOut(diffusion_images[10]),
                  FadeIn(ae_attn_head_group),
                  run_time=2)

        self.wait()
        self.play(ReplacementTransform(action_expert_h6_border, action_expert_attn_head_border), 
                  ae_attn_head_group.animate.scale(1/0.08).move_to([ 2.7 , -5.6124,  0.]),
                  self.frame.animate.reorient(0, 0, 0, (-0.12, -5.59, 0.0), 7.51),
                  run_time=5)

        # Ok yeah I think i like this sliding to the side thing
        # Goes nicely with "different set of inputs"

        self.wait()
        self.play(Write(all_svgs[42]), 
                 self.frame.animate.reorient(0, 0, 0, (3.41, -5.6, 0.0), 5.31), 
                 run_time=5)

        self.wait()
        self.play(FadeOut(all_svgs[42]), run_time=2)

        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (-0.12, -5.59, 0.0), 7.51),
                  # FadeOut(all_svgs[42]),
                  run_time=5)

        # ok so I think zooming in and out here is reasonable/nice
        # I'l need to add some premiere arrows etc, but I think this can 
        # work!
        # Zoom out to setup P59!
        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (4.14, -2.4, 0.0), 12.75), 
                 run_time=6)


        # P59 - final push!
        # Ok so now we bring down a copy of the keys, and then values
        # A bit messy but we're really close!
        # Let me get everything positioned where we want here. 
        keys_to_move=Group(keys.copy(), attn_dots[2].copy(), attn_dots[3].copy())
        values_to_move=Group(values.copy(), attn_dots[4].copy(), attn_dots[5].copy())

        self.add(keys_to_move)
        self.add(values_to_move)

        
        #Hmm sliding a copy out to the right might be kinda nice?
        # keys_to_move.move_to([9, 0.325, 0])
        # values_to_move.move_to([9, -1.474, 0])

        self.wait()
        self.play(keys_to_move.animate.move_to([9, 0.325, 0]),
                  values_to_move.animate.move_to([9 ,-1.474, 0]),
                  run_time=4)

        # Ok I think simulatenously move down camera
        # move down keys/values copy, and makes space. 
        # Then we can slide things in nicely. 
        WV_box=action_expert_qkv_flow[33:]
        all_svgs[43].scale(0.89)
        # all_svgs[43].next_to(keys_to_move, DOWN, buff=0.08)
        all_svgs[43].move_to([ 3.49996001, -7.58181244,  0.])
        all_svgs[44].scale(0.89)
        # all_svgs[44].next_to(values_to_move, DOWN, buff=0.08)
        all_svgs[44].move_to([  3.5       , -10.48167058,   0.])
        self.wait()
        self.remove(all_svgs[40]) #AE keys label. 
        self.remove(all_svgs[41]) #AE values label. 
        self.play(keys_to_move.animate.move_to([9, -6.717, 0]),
                  values_to_move.animate.move_to([9 ,-9.62, 0]),
                  values_ae.animate.move_to([ 3.5, -8.4,  0. ]),
                  action_expert_attn_head_border.animate.scale([1, 1.35, 1]).move_to([2.88, -6.89, 0]),
                  WV_box.animate.move_to([0.416, -8.4, 0]),
                  action_expert_qkv_flow[31].animate.scale([1.0, 1.8, 1.0]).move_to([-0.2478, -6.83, 0]), 
                  action_expert_qkv_flow[32].animate.move_to([-0.134, -8.4, 0]),
                  self.frame.animate.reorient(0, 0, 0, (5.87, -6.61, 0.0), 8.18),
                  run_time=6)

        self.play(keys_to_move.animate.next_to(keys_ae, DOWN, buff=0.05), 
                  values_to_move.animate.move_to([3.5, -9.62,  0.]), 
                  run_time=3)
        self.add(all_svgs[43], all_svgs[44])

        # values_ae.move_to([ 3.5, -8.4,  0. ]) #Move AE values down

        # action_expert_attn_head_border.scale([1, 1.35, 1])
        # action_expert_attn_head_border.move_to([2.88, -6.89, 0])

        # keys_to_move.next_to(keys_ae, DOWN, buff=0.05)
        # values_to_move.next_to(values_ae, DOWN, buff=0.05)
        # values_to_move.move_to([3.5, -9.62,  0.])

        # Ok I think we do a little illustrator overlay at this point
        # Can add yellow row count brackets, and maybe call outs 
        # for which rows come from where

        # P60 
        # Ok so now we "zoom back out" to the full transformer
        # Just for action expert first, then zoom out to full system. 


        ae_attn_head_group_2=Group(action_expert_qkv_flow, 
                                 # action_expert_attn_head_border,
                                 queries_ae, 
                                 keys_ae, 
                                 values_ae, 
                                 values_to_move,
                                 keys_to_move,
                                 attn_dots_ae,
                                 all_svgs[39],
                                 all_svgs[43],
                                 all_svgs[44]
                                 )

        # self.add(action_expert_h6_border_copy)
        # ae_attn_head_group_2.move_to([4.3, -6.5, 0])

        self.wait()
        self.play(ReplacementTransform(action_expert_attn_head_border, action_expert_h6_border_copy), 
                  ae_attn_head_group_2.animate.scale(0.055).move_to([4.3, -6.47, 0]),
                  self.frame.animate.reorient(0, 0, 0, (5.18, -5.78, 0.0), 6.52),
                  run_time=6)

        # self.wait()
        self.play(FadeIn(action_expert_full_2), 
                  FadeIn(action_expert_label),
                  FadeIn(all_svgs[36][0]),
                  FadeIn(all_svgs[34][1]),
                  FadeIn(diffusion_images[10]),
                  FadeOut(ae_attn_head_group_2),
                  run_time=3)



        # Swap in full Gemma for attention head, then zoom out 
        # to reveal it. 
        self.remove(keys, queries, values, all_svgs[17], all_svgs[18], all_svgs[19], all_svgs[37], all_svgs[14][-1], attn_dots)
        self.add(full_gemma_copy)
        all_svgs[45].shift([0.2, 0, 0])
        all_svgs[46].shift([0.2, 0, 0])
        full_gemma_copy.scale(1.5)
        full_gemma_copy.move_to([5.1, 0.6, 0])
        full_gemma_copy[0].set_color(BLUE2)

        #Make outputs actions a little bigger and add title. 
        diffusion_images[-1].scale(1.3)
        diffusion_images[-1].move_to([12.8, -5.2,  0.])
        final_actions_text.scale(1.1)
        final_actions_text.next_to(diffusion_images[-1], DOWN, buff=0.1)

        self.add(final_actions_text)

        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (4.04, -2.31, 0.0), 13.17),
                  run_time=8)

        #Ok getting close!
        #P60 
        #Zoom in on Gemma, and add lines for QKVs
        #Hmm just dropping the letters is not trivial -> tempted to make 
        # another svg I guess...

        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (4.27, 0.69, 0.0), 7.52), 
                  FadeOut(all_svgs[46]), 
                  run_time=4)

        #Swap these while we're focused on Gemma
        action_expert_full_copy_2=VGroup(*[action_expert_full_copy[i] for i in [1, 2, 3, 4]])
        self.remove(action_expert_h6_border_copy) #, action_expert_full_2[3], action_expert_full_2[2])
        self.remove(action_expert_full)
        self.add(action_expert_full_copy_2) #[3], action_expert_full_copy[4])
        self.remove(action_expert_full_copy_2[3]) #h labels

        # VGroup(*[action_expert_full_copy[i] for i in [0, 1, 2, 3, 5]]).set_opacity(0.6)

        llm_attn_rows_tmp=VGroup() #Text ones
        for i in range(8):
            flat_line = Line(LEFT * 0.45, RIGHT * 0.45)
            flat_line.set_stroke(GREEN, width=3)
            llm_attn_rows_tmp.add(flat_line)

        llm_attn_rows_1=VGroup(llm_attn_rows_tmp, llm_attn_rows_tmp.copy(), llm_attn_rows_tmp.copy())
        
        llm_attn_rows_1[0].arrange(DOWN, buff=0.53) 
        llm_attn_rows_1[0].move_to([0.7, 0.98, 0])

        llm_attn_rows_1[1].arrange(DOWN, buff=0.53)
        llm_attn_rows_1[1].set_color(ORANGE) 
        llm_attn_rows_1[1].move_to([0.7, 0.86, 0])

        llm_attn_rows_1[2].arrange(DOWN, buff=0.53) 
        llm_attn_rows_1[2].set_color(BLUE) 
        llm_attn_rows_1[2].move_to([0.7, 0.74, 0])

        llm_attn_rows_2=llm_attn_rows_1.copy()
        llm_attn_rows_3=llm_attn_rows_1.copy()

        llm_attn_rows_2.move_to([4.3, 0.86, 0])
        llm_attn_rows_3.move_to([7.9, 0.85, 0])

        # Ok i think we make like the full 5 rows down here 
        # And then Replacement Transform
        ae_attn_rows_tmp=VGroup() #Text ones
        for i in range(8):
            flat_line = Line(LEFT * 0.45, RIGHT * 0.45)
            flat_line.set_stroke(RED, width=3)
            ae_attn_rows_tmp.add(flat_line)

        ae_attn_rows_1=VGroup(ae_attn_rows_tmp, ae_attn_rows_tmp.copy(), llm_attn_rows_tmp.copy(), 
                              ae_attn_rows_tmp.copy(), llm_attn_rows_tmp.copy())
        
        ae_attn_rows_1[0].arrange(DOWN, buff=0.53) 
        ae_attn_rows_1[0].move_to([0.7, -4.98, 0])

        ae_attn_rows_1[1].arrange(DOWN, buff=0.53) 
        ae_attn_rows_1[3].set_color(PINK) 
        ae_attn_rows_1[1].move_to([0.7, -5.06, 0])

        ae_attn_rows_1[2].arrange(DOWN, buff=0.53) 
        ae_attn_rows_1[3].set_color(GREEN) 
        ae_attn_rows_1[2].move_to([0.7, -5.14, 0])

        ae_attn_rows_1[3].arrange(DOWN, buff=0.53)
        ae_attn_rows_1[3].set_color(ORANGE) 
        ae_attn_rows_1[3].move_to([0.7, -5.22, 0])

        ae_attn_rows_1[4].arrange(DOWN, buff=0.53) 
        ae_attn_rows_1[4].set_color(BLUE) 
        ae_attn_rows_1[4].move_to([0.7, -5.30, 0])


        ae_attn_rows_2=ae_attn_rows_1.copy()
        ae_attn_rows_3=ae_attn_rows_1.copy()

        ae_attn_rows_2.move_to([4.3, -5.14, 0])
        ae_attn_rows_3.move_to([7.9, -5.15, 0])

        # VGroup(*[full_gemma_copy[i] for i in [0, 1, 2, 3, 5]]).set_opacity(0.6)
        # VGroup(*[action_expert_full_copy[i] for i in [0, 1, 2, 3, 5]]).set_opacity(0.6)
        # all_svgs[36][0].set_opacity(0.6)
        # action_expert_label.set_opacity(0.6)

        # self.add(action_expert_full_copy)
        # self.remove(action_expert_full_copy[5])


        self.wait()
        self.remove(action_expert_full_copy[4])
        all_svgs[27].set_opacity(0.85) #Lower opacity of robot background so it jumps out less
        self.play(VGroup(*[full_gemma_copy[i] for i in [0, 1, 2, 3, 5]]).animate.set_opacity(0.6), 
                  VGroup(*[action_expert_full_copy[i] for i in [1, 2, 3]]).animate.set_opacity(0.6),
                  all_svgs[36][0].animate.set_opacity(0.6),
                  action_expert_label.animate.set_opacity(0.6),
                  *[ShowCreation(llm_attn_rows_1[0][j]) for j in range(8)],
                  *[ShowCreation(llm_attn_rows_1[1][j]) for j in range(8)],
                  *[ShowCreation(llm_attn_rows_1[2][j]) for j in range(8)],
                  run_time=4)
        self.play(*[ShowCreation(llm_attn_rows_2[0][j]) for j in range(8)],
                  *[ShowCreation(llm_attn_rows_2[1][j]) for j in range(8)],
                  *[ShowCreation(llm_attn_rows_2[2][j]) for j in range(8)],
                  run_time=4)
        self.play(*[ShowCreation(llm_attn_rows_3[0][j]) for j in range(8)],
                  *[ShowCreation(llm_attn_rows_3[1][j]) for j in range(8)],
                  *[ShowCreation(llm_attn_rows_3[2][j]) for j in range(8)],
                  run_time=4)


        self.add(ae_attn_rows_1[0], ae_attn_rows_2[0], ae_attn_rows_3[0])
        self.add(ae_attn_rows_1[1], ae_attn_rows_2[1], ae_attn_rows_3[1])

        self.play(self.frame.animate.reorient(0, 0, 0, (4.29, -5.41, 0.0), 7.52),
                  *[ReplacementTransform(llm_attn_rows_1[0], ae_attn_rows_1[2])], 
                  *[ReplacementTransform(llm_attn_rows_1[1], ae_attn_rows_1[3])],
                  *[ReplacementTransform(llm_attn_rows_1[2], ae_attn_rows_1[4])],
                  *[ReplacementTransform(llm_attn_rows_2[0], ae_attn_rows_2[2])], 
                  *[ReplacementTransform(llm_attn_rows_2[1], ae_attn_rows_2[3])],
                  *[ReplacementTransform(llm_attn_rows_2[2], ae_attn_rows_2[4])],
                  *[ReplacementTransform(llm_attn_rows_3[0], ae_attn_rows_3[2])], 
                  *[ReplacementTransform(llm_attn_rows_3[1], ae_attn_rows_3[3])],
                  *[ReplacementTransform(llm_attn_rows_3[2], ae_attn_rows_3[4])],
                  run_time=7)

        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (4.16, -2.38, 0.0), 12.91), 
                  VGroup(*[full_gemma_copy[i] for i in [0, 1, 2, 3, 5]]).animate.set_opacity(1.0), 
                  VGroup(*[action_expert_full_copy[i] for i in [1, 2, 3]]).animate.set_opacity(1.0),
                  all_svgs[36][0].animate.set_opacity(1.0),
                  action_expert_label.animate.set_opacity(1.0),
                  run_time=8)


        # Hmm so for "running mulitple times" do I want to move outputs to 
        # inputs again?
        # Ok so getting really close here
        # I do think it makes sense to have the option in editing
        # to move outputs to inputs etc, and then update the whole 
        # flow one last time for P61 - includeing switching to a new image
        # and moving the cache over again!

        
        # diffusion_images_copy=diffusion_images.copy() #Actually we want to copy earlier

        diffusion_images_copy.set_opacity(1.0)
        for i in range(len(diffusion_images_copy)):
            diffusion_images_copy[i].move_to(tmp_diffusion_images[-1])

        diffusion_images_copy_2.set_opacity(1.0)
        for i in range(len(diffusion_images_copy_2)):
            diffusion_images_copy_2[i].scale(1.3)
            diffusion_images_copy_2[i].move_to(diffusion_images[-1])
        

        self.remove(tmp_diffusion_images, diffusion_images)
        self.add(diffusion_images_copy[0]) #, diffusion_images_copy_2[1])

        self.wait()
        self.play(FadeIn(diffusion_images_copy_2[1]))

        self.wait()
        self.play(diffusion_images_copy_2[1].animate.scale(1/1.3).move_to(diffusion_images_copy[0]),
                  run_time=4)

        self.play(FadeIn(diffusion_images_copy_2[2]))

        #Now just step through the rest:
        for i in range(2, len(diffusion_images_copy_2)):
            self.add(diffusion_images_copy[i-1])
            self.wait()
            self.add(diffusion_images_copy_2[i])
            self.wait()

        # Ok cool now I have the option to use that!
        # P61
        # Alright putting everthing together into a few steps now!
        # 1. Update input image patches 
        # 2. Draw all the llm_attn_rows -> this can just repeat exaclty what we did above
        # 3. Move down the attention rows to the action expert, as we did above
        # 4. Step throug the diffusion process -> probably skipping bring over the output. 
        # 5. Bonus points for doing all of this 2x. Then we're done!
        # Ok now leanign towards P61 in a separate scene if Claude can get 
        # me a clean starting point!
        # I think I'll kinda try both here - the thing I can't do well here is swap the image patches!

        self.wait()
        self.remove(diffusion_images_copy_2)
        self.remove(diffusion_images_copy)
        self.add(diffusion_images_copy[0])

        self.remove(ae_attn_rows_1[4], ae_attn_rows_1[3], ae_attn_rows_1[2])
        self.remove(ae_attn_rows_2[4], ae_attn_rows_2[3], ae_attn_rows_2[2])
        self.remove(ae_attn_rows_3[4], ae_attn_rows_3[3], ae_attn_rows_3[2])

    
        def _make_lines(color, n=8):
            return VGroup(*[
                Line(LEFT * 0.45, RIGHT * 0.45).set_stroke(color, width=3)
                for _ in range(n)
            ])

        _l1 = VGroup(_make_lines(GREEN), _make_lines(ORANGE), _make_lines(BLUE))
        _l1[0].arrange(DOWN, buff=0.53).move_to([0.7, 0.98, 0])
        _l1[1].arrange(DOWN, buff=0.53).move_to([0.7, 0.86, 0])
        _l1[2].arrange(DOWN, buff=0.53).move_to([0.7, 0.74, 0])
        _l2 = _l1.copy().move_to([4.3, 0.86, 0])
        _l3 = _l1.copy().move_to([7.9, 0.85, 0])

        self.wait()

        self.play(
            *[ShowCreation(_l1[c][j]) for c in range(3) for j in range(8)],
            run_time=3,
        )
        self.play(
            *[ShowCreation(_l2[c][j]) for c in range(3) for j in range(8)],
            run_time=3,
        )
        self.play(
            *[ShowCreation(_l3[c][j]) for c in range(3) for j in range(8)],
            run_time=3,
        )

        _a1 = VGroup(
            _make_lines(RED), _make_lines(PINK),
            _make_lines(GREEN), _make_lines(ORANGE), _make_lines(BLUE),
        )
        _a1[0].arrange(DOWN, buff=0.53).move_to([0.7, -4.98, 0])
        _a1[1].arrange(DOWN, buff=0.53).move_to([0.7, -5.06, 0])
        _a1[2].arrange(DOWN, buff=0.53).move_to([0.7, -5.14, 0])
        _a1[3].arrange(DOWN, buff=0.53).move_to([0.7, -5.22, 0])
        _a1[4].arrange(DOWN, buff=0.53).move_to([0.7, -5.30, 0])
        _a2 = _a1.copy().move_to([4.3, -5.14, 0])
        _a3 = _a1.copy().move_to([7.9, -5.15, 0])

        # Red + pink rows appear directly
        self.add(_a1[0], _a2[0], _a3[0])
        self.add(_a1[1], _a2[1], _a3[1])

        # Green/orange/blue slide down from LLM → action expert
        self.wait()
        self.play(
            *[ReplacementTransform(_l1[c], _a1[c + 2]) for c in range(3)],
            *[ReplacementTransform(_l2[c], _a2[c + 2]) for c in range(3)],
            *[ReplacementTransform(_l3[c], _a3[c + 2]) for c in range(3)],
            run_time=5,
        )



        # self.add(diffusion_images_copy[0]) #, diffusion_images_copy_2[1])
        # diffusion_images_copy_2.set_opacity(1.0)
        # self.remove(diffusion_images_copy)

        self.wait()

        diffusion_images_copy_3.set_opacity(1.0)
        for i in range(len(diffusion_images_copy)):
            diffusion_images_copy_3[i].move_to(tmp_diffusion_images[-1])

        diffusion_images_copy_4.set_opacity(1.0)
        for i in range(len(diffusion_images_copy_4)):
            diffusion_images_copy_4[i].scale(1.3)
            diffusion_images_copy_4[i].move_to(diffusion_images[-1])
    

        #Now just step through the rest:
        for i in range(1, len(diffusion_images_copy_4)):
            self.add(diffusion_images_copy_3[i-1])
            self.wait()
            self.add(diffusion_images_copy_4[i])
            self.wait()



        self.wait(20)
        self.embed()



class poster_2(InteractiveScene):
    def construct(self):
        hacking_dir = Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/vla/hackin')
        composited_dir = hacking_dir / 'p43_patchified'

        total_height = 2.72
        grid_n = 16
        patch_size = total_height / grid_n
        gap_factor = 0.12
        scale = 1 + gap_factor

        all_attn_values = np.load(hacking_dir / 'p42_1/p42_1.npy')
        max_opacities = [0.5, 0.95, 0.95]

        # Compute canvas dimensions in manim units (must match Jupyter export)
        xs = [(j - 8 + 0.5) * patch_size * scale for j in range(16)]
        ys = [-(i - 8 + 0.5) * patch_size * scale for i in range(2, 14)]
        canvas_w_manim = (max(xs) - min(xs)) + patch_size
        canvas_h_manim = (max(ys) - min(ys)) + patch_size

        # Create initial 3 images
        display_imgs = Group()
        img_gap = 0.3
        target_h = 3.5  # pick a nice display height
        for k in range(3):
            img = ImageMobject(str(composited_dir / f'0_{k}.png'))
            img.set_height(target_h)
            display_imgs.add(img)

        w = display_imgs[0].get_width()
        display_imgs[0].move_to([-(w + img_gap), 0, 0])
        display_imgs[1].move_to([0, 0, 0])
        display_imgs[2].move_to([(w + img_gap), 0, 0])

        # Scale factor: displayed image height / original canvas height in manim units
        s = target_h / canvas_h_manim
        scaled_patch = patch_size * s

        # Build magenta overlays aligned to actual image pixels
        magenta_overlays = Group()
        for k in range(3):
            overlays_k = Group()
            img_center = display_imgs[k].get_center()
            for i in range(2, 14):
                for j in range(16):
                    cx = (j - 8 + 0.5) * patch_size * scale * s
                    cy = -(i - 8 + 0.5) * patch_size * scale * s
                    sq = Square(side_length=scaled_patch)
                    sq.set_fill(MAGENTA, opacity=0)
                    sq.set_stroke(width=0)
                    sq.move_to(img_center + np.array([cx, cy, 0]))
                    overlays_k.add(sq)
            magenta_overlays.add(overlays_k)



        self.add(display_imgs, magenta_overlays)

        # self.frame.reorient(0, 0, 0, (-0.02, -0.07, 0.0), 5.62)
        self.frame.reorient(0, 0, 0, (0.01, 0.03, 0.0), 8.66)

        for frame_idx in range(150,300):
            attn_row = all_attn_values[frame_idx]
            cams = [attn_row[:256].reshape(16, 16), attn_row[256:512].reshape(16, 16), attn_row[512:768].reshape(16, 16)]

            for k in range(3):
                path = composited_dir / f'{frame_idx}_{k}.png'
                if not path.exists():
                    continue

                pos = display_imgs[k].get_center()
                h = display_imgs[k].get_height()
                old_img = display_imgs[k]
                
                # Close the PIL file handle before discarding
                if hasattr(old_img, 'image') and hasattr(old_img.image, 'close'):
                    old_img.image.close()
                self.remove(old_img)

                new_img = ImageMobject(str(path))
                new_img.set_height(h)
                new_img.move_to(pos)
                display_imgs.submobjects[k] = new_img
                self.add(new_img)
                self.remove(magenta_overlays); self.add(magenta_overlays)
                del old_img

                cam = cams[k][2:14, :]
                sc = cam.max() if cam.max() > 0 else 1.0
                mo = max_opacities[k]
                for idx in range(len(magenta_overlays[k])):
                    row, col = idx // 16, idx % 16
                    attn_val = cam[row, col]
                    opacity = min(mo, mo * (attn_val / sc))
                    magenta_overlays[k][idx].set_fill(opacity=float(opacity))

            self.wait(1 / 15)

        self.wait()


class poster_3(InteractiveScene):
    def construct(self):
        hacking_dir = Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/vla/hackin')
        composited_dir = hacking_dir / 'p43_patchified'

        total_height = 2.72
        grid_n = 16
        patch_size = total_height / grid_n
        gap_factor = 0.12
        scale = 1 + gap_factor

        all_attn_values = np.load(hacking_dir / 'p42_1/p42_1.npy')
        max_opacities = [0.5, 0.95, 0.95]

        # Compute canvas dimensions in manim units (must match Jupyter export)
        xs = [(j - 8 + 0.5) * patch_size * scale for j in range(16)]
        ys = [-(i - 8 + 0.5) * patch_size * scale for i in range(2, 14)]
        canvas_w_manim = (max(xs) - min(xs)) + patch_size
        canvas_h_manim = (max(ys) - min(ys)) + patch_size

        # Create initial 3 images
        display_imgs = Group()
        img_gap = 0.3
        target_h = 3.5  # pick a nice display height
        for k in range(3):
            img = ImageMobject(str(composited_dir / f'0_{k}.png'))
            img.set_height(target_h)
            display_imgs.add(img)

        w = display_imgs[0].get_width()
        display_imgs[0].move_to([-(w + img_gap), 0, 0])
        display_imgs[1].move_to([0, 0, 0])
        display_imgs[2].move_to([(w + img_gap), 0, 0])

        # Scale factor: displayed image height / original canvas height in manim units
        s = target_h / canvas_h_manim
        scaled_patch = patch_size * s

        # Build magenta overlays aligned to actual image pixels
        magenta_overlays = Group()
        for k in range(3):
            overlays_k = Group()
            img_center = display_imgs[k].get_center()
            for i in range(2, 14):
                for j in range(16):
                    cx = (j - 8 + 0.5) * patch_size * scale * s
                    cy = -(i - 8 + 0.5) * patch_size * scale * s
                    sq = Square(side_length=scaled_patch)
                    sq.set_fill(MAGENTA, opacity=0)
                    sq.set_stroke(width=0)
                    sq.move_to(img_center + np.array([cx, cy, 0]))
                    overlays_k.add(sq)
            magenta_overlays.add(overlays_k)

        self.frame.reorient(0, 0, 0, (0.01, 0.03, 0.0), 8.66)
        self.add(display_imgs, magenta_overlays)

        for frame_idx in range(0, 151):
            attn_row = all_attn_values[frame_idx]
            cams = [attn_row[:256].reshape(16, 16), attn_row[256:512].reshape(16, 16), attn_row[512:768].reshape(16, 16)]

            for k in range(3):
                path = composited_dir / f'{frame_idx}_{k}.png'
                if not path.exists():
                    continue

                pos = display_imgs[k].get_center()
                h = display_imgs[k].get_height()
                old_img = display_imgs[k]
                
                # Close the PIL file handle before discarding
                if hasattr(old_img, 'image') and hasattr(old_img.image, 'close'):
                    old_img.image.close()
                self.remove(old_img)

                new_img = ImageMobject(str(path))
                new_img.set_height(h)
                new_img.move_to(pos)
                display_imgs.submobjects[k] = new_img
                self.add(new_img)
                self.remove(magenta_overlays); self.add(magenta_overlays)
                del old_img

                cam = cams[k][2:14, :]
                sc = cam.max() if cam.max() > 0 else 1.0
                mo = max_opacities[k]
                for idx in range(len(magenta_overlays[k])):
                    row, col = idx // 16, idx % 16
                    attn_val = cam[row, col]
                    opacity = min(mo, mo * (attn_val / sc))
                    magenta_overlays[k][idx].set_fill(opacity=float(opacity))

            self.wait(1 / 15)

        self.wait()
