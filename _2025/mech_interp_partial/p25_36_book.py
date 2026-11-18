from welch_paths import welch_path
from pathlib import Path

import manimgx as gx
import numpy as np
from welch_geometry import DiagramMatrix, diagram_arrow
from matplotlib.colors import rgb_to_hex
from welch_scene import WelchScene, image_at_height

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import cm

import sys
sys.path.append(welch_path('dark_matter_of_ai/animations/videos'))
# /Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/dark_matter_of_ai/animations/videos/gemma_cache_dict_1.p
from helpers import (
    TOKENS_BY_LAYER,
    circle_plus,
    create_prism,
    get_image_and_border,
    get_transformer_block,
    show_top_tokens,
)
import pickle

data_dir=welch_path('dark_matter_of_ai/animations/videos')

with open(data_dir+'/gemma_cache_dict_1.p', 'rb') as f:
    cache=pickle.load(f)

with open(data_dir+'/w_u_reduced.p', 'rb') as f:
    w_u_reduced=pickle.load(f)

with open(data_dir+'/example_output.p', 'rb') as f: #logits, probs, for wikipedia example. 
    example_output=pickle.load(f)

with open(data_dir+'/top_activating_text_dec_19_2.p', 'rb') as f:
    top_activating_words,top_activating_word_activations=pickle.load(f)


CHILL_BROWN='#948979'
YELLOW='#ffd35a'

class P25b(WelchScene):
    def construct(self):

        layer_id = 25
        sampled_layer_out_matrix = np.hstack((
            cache['blocks.'+str(layer_id)+'.hook_resid_post'][1:, :3], 
            cache['blocks.'+str(layer_id)+'.hook_resid_post'][1:, -2:]
        ))
        layer_out_matrix = DiagramMatrix(sampled_layer_out_matrix.round(2), ellipses_col=3).scale(0.6)
        self.add(layer_out_matrix)

        self.set_view(0, 0, 0, (0,0,0), 8)

        # Create rectangle with rounded corners
        last_row = layer_out_matrix.get_entries()[-len(sampled_layer_out_matrix[0])+1:]
        surrounding_rect = gx.RoundedRectangle(
            corner_radius=0.1,
            stroke_color='#ffd35a',
            stroke_width=3,
            width=last_row[-1].get_center()[0] - last_row[0].get_center()[0] + 1,
            height=0.6,
            fill_opacity=0
        )
        
        surrounding_rect.move_to(last_row[len(last_row)//2])
        
        self.play(gx.Create(surrounding_rect, run_time=1.5), gx.FadeIn(surrounding_rect, rate_func=lambda t: gx.smooth(t), run_time=1.2))

        last_row=layer_out_matrix[-7:-2].copy()
        self.remove(layer_out_matrix[-7:-2])
        self.add(last_row)

        #Ok, just checked, reshaping operation is "row first"
        sampled_layer_out_matrix_longer = np.hstack((
            cache['blocks.'+str(layer_id)+'.hook_resid_post'][-1:, :200], #Legit tempted to do more than 400 elements now!
            cache['blocks.'+str(layer_id)+'.hook_resid_post'][-1:, -200:] #Let's revisit this later when I have time, keep rolling for now. 
        ))
        last_row_longer=DiagramMatrix(sampled_layer_out_matrix_longer.round(2), ellipses_col=200).scale(0.6)

        ## -- Testing
        # self.add(last_row_longer)
        # last_row_longer.scale(0.03)
        ## ---


        diff=last_row[0].get_center()-last_row_longer[0].get_center()
        last_row_longer.shift(diff)
        # self.add(last_row_longer)

        self.play(gx.FadeOut(surrounding_rect), gx.FadeOut(layer_out_matrix), run_time=1)
        self.wait()

        self.play(gx.FadeOut(last_row), gx.FadeIn(last_row_longer), run_time=1)
        self.wait()

        # self.play(self.animate_view(0, 0, 0, (230.04, 1.93, 0.0), 276.47), run_time=5)
        # Instead of moving camera, might be better to make vector small. 
        # self.play(AnimationGroup(last_row_longer.animate.scale(0.03), last_row_longer.animate.move_to(ORIGIN),lag_ratio=0.8),  run_time=4)
        self.play(last_row_longer.animate.scale(0.03).move_to(gx.ORIGIN),run_time=4)
        self.wait()

        b=gx.Brace(last_row_longer, gx.UP, buff=0.1, stroke_width=0)
        b.set_color(CHILL_BROWN)
        bt=gx.Text('2,304').scale(0.6).next_to(b, gx.UP, buff=0.1).set_color(CHILL_BROWN)
        self.add(b, bt)
        self.wait()

        self.remove(b,bt)
        self.wait()

        #Create groups for each row
        groups=[]
        for i in range(20):
            groups.append(gx.VGroup([*last_row_longer[i*20:(i+1)*20]]))

        initial_vertical_spacing=0.3
        self.remove(last_row_longer[-3:]) #Remove brackets
        self.play(*[g.animate.scale(8).move_to(gx.UP*initial_vertical_spacing*(10-i)) for i,g in enumerate(groups)], run_time=4)
        self.wait()

        #Now Fade in the colormap!
        input_matrix=get_image_and_border(path='gemma_cached_images_dec_16_1/hook_embed_1.png', scale=1.5)
        input_matrix.shift(0.15*gx.UP) #Nudge to align with numbers
        input_matrix_color_bar=image_at_height(Path(data_dir) / 'gemma_cached_images_dec_16_1/hook_embed_1_colorbar.png')
        input_matrix_color_bar.to_edge(gx.RIGHT, buff=1.0)
        
        self.play(gx.FadeIn(input_matrix))
        self.add(input_matrix_color_bar)
        self.wait()

        # self.add(input_matrix)
        # input_matrix.set_opacity(0.5)
        # self.remove(input_matrix)

class P25bMoreElements(WelchScene):
    def construct(self):

        layer_id = 25
        sampled_layer_out_matrix = np.hstack((
            cache['blocks.'+str(layer_id)+'.hook_resid_post'][1:, :3], 
            cache['blocks.'+str(layer_id)+'.hook_resid_post'][1:, -2:]
        ))
        layer_out_matrix = DiagramMatrix(sampled_layer_out_matrix.round(2), ellipses_col=3).scale(0.6)
        self.add(layer_out_matrix)

        self.set_view(0, 0, 0, (0,0,0), 8)

        # Create rectangle with rounded corners
        last_row = layer_out_matrix.get_entries()[-len(sampled_layer_out_matrix[0])+1:]
        surrounding_rect = gx.RoundedRectangle(
            corner_radius=0.1,
            stroke_color='#ffd35a',
            stroke_width=3,
            width=last_row[-1].get_center()[0] - last_row[0].get_center()[0] + 1,
            height=0.6,
            fill_opacity=0
        )
        
        surrounding_rect.move_to(last_row[len(last_row)//2])
        
        self.play(gx.Create(surrounding_rect, run_time=1.5), gx.FadeIn(surrounding_rect, rate_func=lambda t: gx.smooth(t), run_time=1.2))

        last_row=layer_out_matrix[-7:-2].copy()
        self.remove(layer_out_matrix[-7:-2])
        self.add(last_row)

        #Ok, just checked, reshaping operation is "row first"
        # sampled_layer_out_matrix_longer = np.hstack((
        #     cache['blocks.'+str(layer_id)+'.hook_resid_post'][-1:, :450], #Legit tempted to do more than 400 elements now!
        #     cache['blocks.'+str(layer_id)+'.hook_resid_post'][-1:, -450:] #Let's revisit this later when I have time, keep rolling for now. 
        # ))
        sampled_layer_out_matrix_longer=cache['blocks.'+str(layer_id)+'.hook_resid_post'][-1:, :] #Dare we try the whole thing?
        last_row_longer=DiagramMatrix(sampled_layer_out_matrix_longer.round(2), ellipses_col=1152).scale(0.6)

        ## -- Testing
        # self.add(last_row_longer)
        # # last_row_longer.scale(0.03)
        # self.remove(last_row_longer)
        ## ---


        diff=last_row[0].get_center()-last_row_longer[0].get_center()
        last_row_longer.shift(diff)
        # self.add(last_row_longer)

        self.play(gx.FadeOut(surrounding_rect), gx.FadeOut(layer_out_matrix), run_time=1)
        self.wait()

        self.add(last_row_longer)
        self.play(gx.FadeOut(last_row), gx.FadeIn(last_row_longer), run_time=1)
        self.wait()

        # self.play(self.animate_view(0, 0, 0, (230.04, 1.93, 0.0), 276.47), run_time=5)
        # Instead of moving camera, might be better to make vector small. 
        # self.play(AnimationGroup(last_row_longer.animate.scale(0.03), last_row_longer.animate.move_to(ORIGIN),lag_ratio=0.8),  run_time=4)
        # self.play(last_row_longer.animate.scale(0.03).move_to(ORIGIN),run_time=4) #Length 400
        self.play(last_row_longer.animate.scale(0.00458).move_to(gx.ORIGIN),run_time=4) #Length 3204
        self.wait()

        b=gx.Brace(last_row_longer, gx.UP, buff=0.1, stroke_width=0)
        b.set_color(CHILL_BROWN)
        bt=gx.Text('2,304').scale(0.6).next_to(b, gx.UP, buff=0.1).set_color(CHILL_BROWN)
        self.add(b, bt)
        self.wait()

        self.remove(b,bt)
        self.wait()

        #Create groups for each row
        groups=[]
        for i in range(48):
            groups.append(gx.VGroup([*last_row_longer[i*48:(i+1)*48]]))

        initial_vertical_spacing=0.143 #0.3*(20./48)
        self.remove(last_row_longer[-3:]) #Remove brackets
        self.play(*[g.animate.scale(24).move_to(gx.UP*initial_vertical_spacing*(24-i)) for i,g in enumerate(groups)], run_time=4)
        self.wait()

        #Now Fade in the colormap!
        input_matrix=get_image_and_border(path='gemma_cached_images_dec_16_1/hook_embed_1.png')
        input_matrix.scale(1.135)
        input_matrix.shift(0.09*gx.UP) #Nudge to align with numbers
        input_matrix.set_opacity(0.5)

        self.add(input_matrix)
        self.remove(input_matrix)

        input_matrix_color_bar=image_at_height(Path(data_dir) / 'gemma_cached_images_dec_16_1/hook_embed_1_colorbar.png')
        input_matrix_color_bar.to_edge(gx.RIGHT, buff=1.0)
        
        self.play(gx.FadeIn(input_matrix))
        self.add(input_matrix_color_bar)
        self.wait()

class P26_P27(WelchScene):
    def construct(self):
        #Pick up where we left off in P26
        input_matrix=get_image_and_border(path='gemma_cached_images_dec_16_1/hook_embed_1.png', scale=1.5)
        input_matrix.shift(0.15*gx.UP) #Nudge to align with numbers
        input_matrix_color_bar=image_at_height(Path(data_dir) / 'gemma_cached_images_dec_16_1/hook_embed_1_colorbar.png')
        input_matrix_color_bar.to_edge(gx.RIGHT, buff=1.0)
        
        self.add(input_matrix, input_matrix_color_bar)
        self.wait()

        self.remove(input_matrix_color_bar)
        self.play(input_matrix.animate.move_to(3*gx.LEFT).scale(0.3), run_time=2.0)
        self.wait()

        square = gx.RoundedRectangle(width=2, height=2, corner_radius=0.2, fill_opacity=0.0, stroke_color='#ffd35a', stroke_width=4)
        square_text = gx.Text("Unembed", font="Myriad Pro").set_color('#ffd35a')
        square_text.scale(0.8)  # Adjust size as needed

        # Add smaller gray text below Gemma
        subtext = gx.Text("norm>unembed \n >softcap>softmax", font="Myriad Pro").set_color('#555555')
        subtext.scale(0.3)  # Make it smaller than Gemma text
        subtext.next_to(square_text, gx.DOWN, buff=0.1)  # Position it below Gemma

        # Group square and text together
        square_group = gx.VGroup(square, square_text, subtext)
        square_group.next_to(input_matrix, gx.RIGHT, buff=1.0)
        self.add(square_group)


        arrow_1=diagram_arrow(start=input_matrix.get_right(), end=square_group.get_left(), buff=0.2, thickness=4).set_color(YELLOW)
        self.add(arrow_1)
        self.wait(1)

        #Ok, now I need to plot words/probs with very at 100% first, then transition, maybe smootly, maybe not to the actually probs. 
        top_tokens={' very': 1.0,
                     '': 0.0,
                     '': 0.0,
                     '': 0.0,
                     '': 0.0,
                     '': 0.0,
                     '': 0.0,
                     '': 0.0,
                     '': 0.0,
                     '': 0.0}

        # Get viridis colors
        cmap = plt.cm.viridis
        max_value = max(top_tokens.values()) #*1.2 #Dull down top word a bit, lol who am I softcap?
        scale_factor = 1  # Adjust this to change bar lengths

        #  word_group=VGroup(); bar_group=VGroup(); prob_group=VGroup()

        i=0
        word=list(top_tokens.keys())[i]
        hex_color=rgb_to_hex(cmap(top_tokens[word]/max_value)[:3])
        first_word = gx.Text(word, font="Myriad Pro").scale(0.8)
        first_word.set_color(hex_color)
        first_word_hex_color=hex_color #Need dis later
        first_word.next_to(square_group, gx.RIGHT, buff=1.0)
        first_word.shift(0.8*gx.UP)

        # Create first bar and value separately
        first_bar = gx.Rectangle(
            height=0.14,
            width=top_tokens[word]*scale_factor,
            fill_opacity=1,
            color=hex_color
        )
        first_bar.next_to(first_word, gx.RIGHT, buff=0.3)
        first_bar.shift(0.05*gx.UP)
        
        first_value = gx.Text(f"{top_tokens[word]:.4f}").set_color('#FFFFFF').scale(0.38)
        first_value.next_to(first_bar, gx.RIGHT, buff=0.4)
        self.add(first_word, first_bar, first_value)

        arrow_2=arrow_1.copy()
        arrow_2.move_to(square_group.get_right()+0.5*gx.RIGHT)
        self.add(arrow_2)
        self.wait(1)

        words = ["The", "reliability", "of", "Wikipedia", "is", "very"]
        vertical_text = gx.VGroup(*[
            gx.Text(word, font="Myriad Pro").scale(0.8)
            for word in words
        ]).arrange(gx.DOWN, aligned_edge=gx.LEFT, buff=0.3)

        #Add arrows pointing to tokens and token column label
        word_to_vector_arrows=gx.VGroup()
        for i in range(len(vertical_text)):
            vert_pos=vertical_text[i].get_center()[1]
            arrow = diagram_arrow(start=np.array([-4.5,vert_pos,0]), end=np.array([-3.8,vert_pos,0]), buff=0, thickness=3, tip_width_ratio=4)
            arrow.set_color('#948979')
            word_to_vector_arrows.add(arrow)

        vertical_text[2].shift([0,0.07,0.00]) #move of up a little
        vertical_text[4].shift([0,0.07,0.00]) #move is up a little
        vertical_text.to_edge(gx.LEFT)

        sampled_embedding_matrix=np.hstack((cache['hook_embed'][1:, :3],cache['hook_embed'][1:, -2:]))
        embedding_matrix=DiagramMatrix(sampled_embedding_matrix.round(2), ellipses_col=3).scale(0.73)
        embedding_matrix.next_to(word_to_vector_arrows[i], gx.RIGHT, buff=0.6)
        embedding_matrix.shift([0,1.5,0])

        last_row = embedding_matrix.get_entries()[-len(sampled_embedding_matrix[0])+1:]
        surrounding_rect = gx.RoundedRectangle(
            corner_radius=0.1,
            stroke_color='#ffd35a',
            stroke_width=3,
            width=last_row[-1].get_center()[0] - last_row[0].get_center()[0] + 1,
            height=0.6,
            fill_opacity=0
        )
        surrounding_rect.move_to(last_row[len(last_row)//2])

        text_matrix_arrows=gx.VGroup(vertical_text, word_to_vector_arrows, embedding_matrix, surrounding_rect)
        text_matrix_arrows.scale(0.55)
        text_matrix_arrows.next_to(input_matrix, gx.UP, buff=0.6)
        text_matrix_arrows.shift(gx.LEFT*(embedding_matrix.get_center()[0]-input_matrix.get_center()[0]))

        self.add(text_matrix_arrows)

        arrow_3=diagram_arrow(start=embedding_matrix.get_bottom(), end=input_matrix.get_top(), buff=0.1, thickness=4).set_color(YELLOW)
        self.add(arrow_3)
        self.wait(1)

        #I want to do some cool shake animation on very here, but I need to keep moving and dont' see how to do it in 3b1b manim. 
        self.remove(first_word, first_bar, first_value)

        top_tokens={"_very": 0.006,
                    "very": 0.006,
                    "_Very": 0.006,
                    "Very": 0.006,
                    "_VERY": 0.006,
                    "_très": 0.006,
                    "_muy": 0.006,
                    "VERY": 0.006,
                    "_extremely": 0.006,
                    "_sehr": 0.0059}

        # Get viridis colors
        cmap = plt.cm.viridis
        max_value = max(top_tokens.values())*3 #Dull down top word a bit, lol who am I softcap?
        scale_factor = 15  # Adjust this to change bar lengths

        word_group=gx.VGroup(); bar_group=gx.VGroup(); prob_group=gx.VGroup()

        i=0
        word=list(top_tokens.keys())[i]
        hex_color=rgb_to_hex(cmap(top_tokens[word]/max_value)[:3])
        first_word = gx.Text(word, font="Myriad Pro").scale(0.8)
        first_word.set_color(hex_color)
        first_word_hex_color=hex_color #Need dis later
        first_word.next_to(square_group, gx.RIGHT, buff=1.2)
        first_word.shift(2.5*gx.UP)

        # Create first bar and value separately
        first_bar = gx.Rectangle(
            height=0.14,
            width=top_tokens[word]*scale_factor,
            fill_opacity=1,
            color=hex_color
        )
        first_bar.next_to(first_word, gx.RIGHT, buff=1.2)
        # first_bar.shift(0.05*UP)
        
        first_value = gx.Text(f"{top_tokens[word]:.4f}").set_color('#FFFFFF').scale(0.38)
        first_value.next_to(first_bar, gx.RIGHT, buff=0.4)
        # self.add(first_word, first_bar, first_value)


        word_group.add(first_word); bar_group.add(first_bar); prob_group.add(first_value)

        # Create group for animations
        word_bar_groups = gx.VGroup()
        vertical_spacing=0.55

        # Create remaining words, bars, and values
        for i, (word, value) in enumerate(list(top_tokens.items())[1:], 1):
            # Create word text
            word_text = gx.Text(word, font="Myriad Pro", stroke_width=0).scale(0.8)
            hex_color=rgb_to_hex(cmap(top_tokens[word]/max_value)[:3])
            word_text.set_color(hex_color)
            word_text.next_to(square_group, gx.RIGHT, buff=1.2)
            word_text.shift(2.5*gx.UP)
            word_text.shift(gx.DOWN * i * vertical_spacing)
            
            # Create bar
            bar = gx.Rectangle(
                height=0.14,
                width=value * scale_factor,
                fill_opacity=1,
                color=hex_color
            )
            bar.next_to(first_word, gx.RIGHT, buff=1.2)
            bar.shift(gx.DOWN * i * vertical_spacing)
            
            # # Create value text
            value_text = gx.Text(f"{value:.4f}").set_color('#FFFFFF').scale(0.38)
            value_text.next_to(first_bar, gx.RIGHT, buff=0.4)
            value_text.shift(gx.DOWN * i * vertical_spacing)
            
            # word_bar_groups.add(VGroup(word_text, bar, value_text))
            word_group.add(word_text); bar_group.add(bar); prob_group.add(value_text)

        #Animate bars growing from left
        for bar in bar_group:
            bar.save_state()
            bar.stretch(0, 0, about_edge=gx.LEFT)  # Shrink to width 0 from left edge

        self.play(*[gx.Restore(bar) for bar in bar_group]+[gx.FadeIn(word_group)], run_time=2.0, rate_func=gx.smooth)
        self.add(prob_group)
        self.wait()
        # self.play(FadeOut([word_group, bar_group, prob_group, arrow_1, arrow_2, arrow_3, square_group]), run_time=1.0)
        self.play(gx.FadeOut(gx.VGroup(word_group, bar_group, prob_group, arrow_1, arrow_2, arrow_3, square_group, text_matrix_arrows)), run_time=1.0)
        
        #Ok cool so that's a pretty clean ending I can use to start P28
        self.wait()


class P28_P30(WelchScene):
    def construct(self):
        #Pick up where we left off in P27 ish, since the whole reference frame is shifting though, I'm not going to worry about 
        #nailing it here, I'll stitch it better in Premiere, here I just need to make sure we're square-ish to the image. 

        input_matrix=get_image_and_border(path='gemma_cached_images_dec_16_1/hook_embed_1.png', scale=1.5)
        input_matrix.rotate(90*gx.DEGREES, np.array([1,0,0]))
        self.add(input_matrix)

        #Go ahead and add a copy that we're going to move later
        input_matrix_2=input_matrix.copy()
        self.add(input_matrix_2)
        self.set_view(0, 90, 0, (4.5, -0.06, -0.18), 13.47)
        self.wait(1)


        attention_block_width=8
        attention_block_height=6
        attention_block_depth=1
        mlp_block_width=8
        mlp_block_height=6
        mlp_block_depth=1
        block_orig=np.array([0,0,0])
        line_padding=0.3
        residual_compute_block_spacing=11
        line_thickness=6
        circle_stroke_width=3
        plus_stroke_width=3

        attention_block_1=create_prism(center=block_orig+np.array([11,-4,0]), height=attention_block_depth, width=attention_block_width, depth=attention_block_height, 
                                    face_colors=gx.BLUE, opacity=0.2, label_text='Attention',
                                    label_size=80, label_opacity=0.5, label_face="bottom")

        #Input to Attention
        a0=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing+line_padding,0,0]),
                  end=block_orig+np.array([3.5,0,0]), #Special offset from matrix here, not sure how to handle this generally yet
                  fill_color=gx.WHITE, thickness=line_thickness, tip_width_ratio=0)
        a1=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing, 0.3,0]),
                 end=block_orig+np.array([residual_compute_block_spacing, -3.0,0]),fill_color=gx.WHITE, thickness=line_thickness)

        self.play(gx.FadeIn(attention_block_1), gx.FadeIn(a0), gx.FadeIn(a1), self.animate_view(-43, 64, 0, (5.25, -5.09, -1.39), 17.08), run_time=2.0)


        input_matrix.set_opacity(0.3)
        self.play(input_matrix_2.animate.move_to(np.array([residual_compute_block_spacing, 0,0])), run_time=2)
        self.play(input_matrix_2.animate.move_to(np.array([residual_compute_block_spacing, -3,0])), run_time=1, rate_func=gx.linear)
        attention_out_matrix=get_image_and_border(path='gemma_cached_images_dec_16_1/hook_attn_out_1.png', scale=1.5)
        attention_out_matrix.rotate(90*gx.DEGREES, np.array([1,0,0]))
        attention_out_matrix.move_to(np.array([residual_compute_block_spacing, -3, 0]))
        self.play(gx.FadeOut(input_matrix_2),
                  gx.FadeIn(attention_out_matrix),
                  input_matrix_2.animate.move_to(np.array([residual_compute_block_spacing, -5, 0])),
                  attention_out_matrix.animate.move_to(np.array([residual_compute_block_spacing, -5, 0])),
                  run_time=1,
                  rate_func=gx.linear)

        self.wait(1)

        #Attention to Residual
        a2=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing, -5.0, 0]),
                   end=block_orig+np.array([residual_compute_block_spacing, -6.3, 0]),fill_color=gx.WHITE, thickness=line_thickness, tip_width_ratio=0)
        a3=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing+line_padding,-6,0]),
                   end=block_orig+np.array([0.5,-6,0]),fill_color=gx.WHITE, thickness=line_thickness)

        #First Addition
        c=circle_plus(circle_stroke_width=circle_stroke_width, plus_stroke_width=plus_stroke_width, overall_scale=0.6, position=block_orig+np.array([0,-6,0]))

        #Input to First Addition
        a4=diagram_arrow(start=block_orig+np.array([0,-0.75,0]),
                   end=block_orig+np.array([0,-5.5,0]),fill_color=gx.WHITE, thickness=line_thickness)
        
        self.play(gx.FadeIn(a2), gx.FadeIn(a3), gx.FadeIn(c), gx.FadeIn(a4))
        self.wait(1)

        #Move forward residual stream
        self.play(input_matrix.animate.move_to(np.array([0,-6,0])), run_time=2)
        self.wait()

        #Move attention output to summation point
        self.remove(input_matrix_2)
        self.play(attention_out_matrix.animate.move_to(np.array([residual_compute_block_spacing, -6, 0])), run_time=1, rate_func=gx.smooth)
        self.play(attention_out_matrix.animate.move_to(np.array([0, -6, 0])), run_time=2, rate_func=gx.smooth)

        #Add Attention back to residual       
        residual_matrix_1=get_image_and_border(path='gemma_cached_images_dec_16_1/hook_resid_mid_1.png', scale=1.5)
        residual_matrix_1.rotate(90*gx.DEGREES, np.array([1,0,0]))
        residual_matrix_1.move_to(np.array([0, -6, 0]))
        self.add(residual_matrix_1)
        self.remove(input_matrix)
        self.remove(attention_out_matrix)
        self.wait(1)
        
        self.play(self.animate_view(0, 78, 0, (0.45, -4.73, 0.05), 17.08))
        self.wait()

        ## Now add in tokens/probs! Probably time to put this in some kinda function or class?
        top_tokens= {'_very': 0.0117,
                    'very': 0.0117,
                    'Very': 0.0117,
                    '_Very': 0.0117,
                    '_VERY': 0.0117,
                    'VERY': 0.0117,
                    '_très': 0.0117,
                    '_muy': 0.0117,
                    '_sehr': 0.0116,
                    '_extremely': 0.0115}
        word_group, bar_group, prob_group=show_top_tokens(top_tokens)
        top_tokens_group=gx.VGroup(word_group, bar_group, prob_group)
        top_tokens_group.rotate(90*gx.DEGREES, np.array([1,0,0]))
        top_tokens_group.scale(1.8)
        top_tokens_group.next_to(residual_matrix_1, gx.LEFT, buff=2.2)

        self.add(word_group, bar_group, prob_group)

        arrow_3=diagram_arrow(start=residual_matrix_1.get_left(), end=top_tokens_group.get_right(), buff=0.4, thickness=8).set_color(YELLOW)
        arrow_3.rotate(90*gx.DEGREES, np.array([1,0,0]))
        self.add(arrow_3)
        self.wait(1)

        t=gx.Text('The reliability of Wikipedia is very very ...', font="Myriad Pro").set_color(CHILL_BROWN)
        t.rotate(90*gx.DEGREES, np.array([1,0,0]))
        t.scale(1.8)
        t.next_to(residual_matrix_1, gx.OUT, buff=2.5)
        self.play(gx.Write(t))
        self.wait(1)

        self.remove(t, top_tokens_group, arrow_3)
        self.play(self.animate_view(-27, 58, 0, (5.02, -7.35, -0.51), 16.90), residual_matrix_1.animate.move_to(np.array([0, -7.5, 0])), run_time=2)

        #Residual to MLP
        a6=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing+line_padding,-7.5,0]),
                   end=block_orig+np.array([3.5,-7.5,0]), fill_color=gx.WHITE, thickness=line_thickness, tip_width_ratio=0)
        a7=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing, -7.2,0]),
                   end=block_orig+np.array([residual_compute_block_spacing, -9,0]), fill_color=gx.WHITE, thickness=line_thickness)
        self.add(a6,a7)
        mlp_block_1=create_prism(center=block_orig+np.array([11,-10,0]), height=mlp_block_depth, width=mlp_block_width, depth=mlp_block_height, 
                            face_colors=gx.GREEN, opacity=0.2, label_text='MLP',
                            label_size=80, label_opacity=0.5, label_face="bottom")
        self.add(mlp_block_1)

        residual_matrix_1b=residual_matrix_1.copy()
        self.add(residual_matrix_1b)

        residual_matrix_1.set_opacity(0.3)

        self.play(residual_matrix_1b.animate.move_to(np.array([residual_compute_block_spacing, -7.5,0])), run_time=2)
        self.play(residual_matrix_1b.animate.move_to(np.array([residual_compute_block_spacing, -9,0])), run_time=1, rate_func=gx.linear)

        mlp_out_matrix=get_image_and_border(path='gemma_cached_images_dec_18_1/hook_mlp_out_1.png', scale=1.5)
        mlp_out_matrix.rotate(90*gx.DEGREES, np.array([1,0,0]))
        mlp_out_matrix.move_to(np.array([residual_compute_block_spacing, -9, 0]))


        self.play(gx.FadeOut(residual_matrix_1b),
                  gx.FadeIn(mlp_out_matrix),
                  residual_matrix_1b.animate.move_to(np.array([residual_compute_block_spacing, -12, 0])),
                  mlp_out_matrix.animate.move_to(np.array([residual_compute_block_spacing, -12, 0])),
                  run_time=1,
                  rate_func=gx.linear)

        self.wait(1)

        #MLP to residual
        a8=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing, -11,0]),
                   end=block_orig+np.array([residual_compute_block_spacing, -12.3,0]), fill_color=gx.WHITE, thickness=line_thickness, tip_width_ratio=0)
        a9=diagram_arrow(start=block_orig+np.array([residual_compute_block_spacing+line_padding,-12,0]),
                   end=block_orig+np.array([0.5,-12,0]), fill_color=gx.WHITE, thickness=line_thickness)
        self.add(a8,a9)


        #Input to Second Addition
        a5=diagram_arrow(start=block_orig+np.array([0,-6.5,0]),
                   end=block_orig+np.array([0,-11.5,0]),fill_color=gx.WHITE, thickness=line_thickness)
        self.add(a5)


        #Second Addition
        c2=circle_plus(circle_stroke_width=circle_stroke_width, plus_stroke_width=plus_stroke_width, overall_scale=0.6, position=block_orig+np.array([0,-12,0]))
        self.add(c2)

        #Move forward residual stream
        self.play(residual_matrix_1.animate.move_to(np.array([0,-12,0])), run_time=2)
        self.wait()

        #Move mlp output to summation point
        self.remove(residual_matrix_1b)
        # self.play(attention_out_matrix.animate.move_to(np.array([residual_compute_block_spacing, -6, 0])), run_time=1, rate_func=smooth)
        self.play(mlp_out_matrix.animate.move_to(np.array([0, -12, 0])), run_time=2, rate_func=gx.smooth)

        #Add Attention back to residual       
        residual_matrix_2=get_image_and_border(path='gemma_cached_images_dec_18_1/hook_resid_post_1.png', scale=1.5)
        residual_matrix_2.rotate(90*gx.DEGREES, np.array([1,0,0]))
        residual_matrix_2.move_to(np.array([0, -12, 0]))
        self.add(residual_matrix_2)
        self.remove(mlp_out_matrix)
        self.remove(residual_matrix_1)
        self.wait(1)         

        self.play(self.animate_view(-1, 71, 0, (1.53, -5.14, -1.39), 22.37))
        self.wait()

        top_tokens={'_very': 0.0161,
                  'very': 0.0161,
                  'Very': 0.0161,
                  '_Very': 0.0161,
                  '_VERY': 0.016,
                  'VERY': 0.0159,
                  '_très': 0.0158,
                  '_muy': 0.0158,
                  '_sehr': 0.0154,
                  '_extremely': 0.0151}
        word_group, bar_group, prob_group=show_top_tokens(top_tokens)
        top_tokens_group=gx.VGroup(word_group, bar_group, prob_group)
        top_tokens_group.rotate(90*gx.DEGREES, np.array([1,0,0]))
        top_tokens_group.scale(1.8)
        top_tokens_group.next_to(residual_matrix_1, gx.LEFT, buff=2.2)

        self.add(word_group, bar_group, prob_group)

        arrow_3=diagram_arrow(start=residual_matrix_1.get_left(), end=top_tokens_group.get_right(), buff=0.4, thickness=8).set_color(YELLOW)
        arrow_3.rotate(90*gx.DEGREES, np.array([1,0,0]))
        self.add(arrow_3)
        self.wait(1)

        #Ok done I think! Let's plan on a clean break to the next scene - if I time to come back and animate, great, it not - that's ok - gotta keep rollin. 

        #Out Arrow
        # a10=Arrow(start=block_orig+np.array([0,-12.5,0]),
        #            end=block_orig+np.array([0,-14.5,0]),fill_color=WHITE, thickness=line_thickness)
        # self.add(a10)



        #Animate bars growing from left
        # for bar in bar_group:
        #     bar.save_state()
        #     bar.stretch(0, 0, about_edge=LEFT)  # Shrink to width 0 from left edge


        # self.play(*[Restore(bar) for bar in bar_group]+[FadeIn(word_group), probs_matrix[3].animate.move_to(first_value.get_center()),
        #             probs_matrix[5].animate.move_to(prob_group[1].get_center())],
        #            run_time=2.0, rate_func=smooth)


class P30_P31(WelchScene):
    def construct(self): 

        # Parse the data
        data = TOKENS_BY_LAYER
        
        # Calculate cell dimensions
        cell_height = 0.4
        cell_width = 1.0
        
        # Create table background
        table = gx.VGroup()
        
        # Get global min and max for color scaling
        all_values = [value for d in data for value in d.values()]
        min_val = min(all_values)
        max_val = max(all_values)
        
        colormap_scale = 1.0
        cmap = plt.cm.viridis
        scaled_colormap_max = max_val * colormap_scale

        # Create headers first
        header_row = gx.VGroup()
        
        # Layer header for first column
        layer_header = gx.Text("Layer", font="Helvetica", font_size=20).set_color(CHILL_BROWN)
        layer_cell = gx.Rectangle(height=cell_height, width=0.6*cell_width, fill_opacity=0.0, stroke_width=0)
        layer_header_group = gx.VGroup(layer_cell, layer_header)
        header_row.add(layer_header_group)
        
        # Top Next Token Predictions header spanning other columns
        predictions_header = gx.Text("Top Next Token Predictions", font="Helvetica", font_size=20).set_color(CHILL_BROWN)
        predictions_cell = gx.Rectangle(height=cell_height, width=cell_width*10, fill_opacity=0.0, stroke_width=0)
        predictions_header_group = gx.VGroup(predictions_cell, predictions_header)
        header_row.add(predictions_header_group)
        
        header_row.arrange(gx.RIGHT, buff=0)
        
        # Create cells
        # for i in range(len(data)):
        for i in range(16): #Start with first 16 layers
            row = gx.VGroup()
            
            # Add index column
            index_cell = gx.Rectangle(height=cell_height, width=0.6*cell_width, fill_opacity=0.0, stroke_width=0)
            index_text = gx.Text(str(i+1), font_size=20, font="Helvetica").set_color(CHILL_BROWN)
            index_cell = gx.VGroup(index_cell, index_text)
            row.add(index_cell)
            
            # Add data columns
            dict_items = list(data[i].items())[:10]
            for key, value in dict_items:
                cell = gx.Rectangle(height=cell_height, width=cell_width, stroke_width=0)
                
                hex_color = rgb_to_hex(cmap(value/scaled_colormap_max)[:3])
                cell.set_fill(hex_color, opacity=0.7)
                
                word_text = gx.Text(key.strip(), font_size=16, font="Helvetica")
                value_text = gx.Text(f"{value:.4f}", font_size=9).set_color('#CCCCCC')
                text_group = gx.VGroup(word_text, value_text).arrange(gx.DOWN, buff=0.05)
                
                cell_group = gx.VGroup(cell, text_group)
                row.add(cell_group)
            
            row.arrange(gx.RIGHT, buff=0)
            table.add(row)
        
        # Arrange rows vertically
        table.arrange(gx.DOWN, buff=0)
        
        # Create horizontal line exactly at top of table
        line = gx.Line(
            start=table.get_left(),
            end=table.get_right(),
            stroke_color=CHILL_BROWN,
            stroke_width=1
        ).move_to(table.get_top(), aligned_edge=gx.DOWN)
        
        # Position headers above line with some spacing
        header_row.next_to(line, gx.UP, buff=0.1)  # Adjust this value to move headers up/down
        
        # Create final group with all elements
        full_visualization = gx.VGroup(header_row, line, table)
        
        # Scale everything
        # full_visualization.scale(1.0)
        
        self.add(full_visualization)
        self.wait(1)

        self.play(full_visualization.animate.shift(gx.RIGHT))

        #Quick addition of two images on left. 
        residual_matrix_1=get_image_and_border(path='gemma_cached_images_dec_19_1/hook_resid_post_1.png')
        residual_matrix_1.scale(0.35)
        residual_matrix_1.to_edge(gx.LEFT, buff=0.5)
        residual_matrix_1.shift(2*gx.UP)
        residual_matrix_1_title=gx.Text("Layer 1 Residual Stream", font='Myriad Pro', font_size=15).set_color(CHILL_BROWN)
        residual_matrix_1_title.next_to(residual_matrix_1, gx.UP, buff=0.1)

        residual_matrix_2=get_image_and_border(path='gemma_cached_images_dec_19_1/hook_resid_post_15.png')
        residual_matrix_2.scale(0.35)
        residual_matrix_2.to_edge(gx.LEFT, buff=0.5)
        residual_matrix_2.shift(2*gx.DOWN)
        residual_matrix_2_title=gx.Text("Layer 15 Residual Stream", font='Myriad Pro', font_size=15).set_color(CHILL_BROWN)
        residual_matrix_2_title.next_to(residual_matrix_2, gx.UP, buff=0.1)

        arrow_1=diagram_arrow(start=residual_matrix_1.get_bottom(), end=residual_matrix_2.get_top(), buff=0.6, thickness=4, tip_width_ratio=3.5).set_color(CHILL_BROWN)
        
        self.add(residual_matrix_1, residual_matrix_2, residual_matrix_1_title, residual_matrix_2_title, arrow_1)
        self.wait(1)

        self.remove(residual_matrix_1, residual_matrix_2, residual_matrix_1_title, residual_matrix_2_title, arrow_1)

        ## -- Ok this is hacky, but need to move fast, I'll wrap up table into a method later if it makes sense. 
        # Create table background
        table = gx.VGroup()
        header_row = gx.VGroup()
        
        # Layer header for first column
        layer_header = gx.Text("Layer", font="Helvetica", font_size=20).set_color(CHILL_BROWN)
        layer_cell = gx.Rectangle(height=cell_height, width=0.6*cell_width, fill_opacity=0.0, stroke_width=0)
        layer_header_group = gx.VGroup(layer_cell, layer_header)
        header_row.add(layer_header_group)
        
        # Top Next Token Predictions header spanning other columns
        predictions_header = gx.Text("Top Next Token Predictions", font="Helvetica", font_size=20).set_color(CHILL_BROWN)
        predictions_cell = gx.Rectangle(height=cell_height, width=cell_width*10, fill_opacity=0.0, stroke_width=0)
        predictions_header_group = gx.VGroup(predictions_cell, predictions_header)
        header_row.add(predictions_header_group)
        
        header_row.arrange(gx.RIGHT, buff=0)
        
        # Create cells
        # for i in range(len(data)):
        for i in range(22): #Start with first 16 layers
            row = gx.VGroup()
            
            # Add index column
            index_cell = gx.Rectangle(height=cell_height, width=0.6*cell_width, fill_opacity=0.0, stroke_width=0)
            index_text = gx.Text(str(i+1), font_size=20, font="Helvetica").set_color(CHILL_BROWN)
            index_cell = gx.VGroup(index_cell, index_text)
            row.add(index_cell)
            
            # Add data columns
            dict_items = list(data[i].items())[:10]
            for key, value in dict_items:
                cell = gx.Rectangle(height=cell_height, width=cell_width, stroke_width=0)
                
                hex_color = rgb_to_hex(cmap(value/scaled_colormap_max)[:3])
                cell.set_fill(hex_color, opacity=0.7)
                
                word_text = gx.Text(key.strip(), font_size=16, font="Helvetica")
                value_text = gx.Text(f"{value:.4f}", font_size=9).set_color('#CCCCCC')
                text_group = gx.VGroup(word_text, value_text).arrange(gx.DOWN, buff=0.05)
                
                cell_group = gx.VGroup(cell, text_group)
                row.add(cell_group)
            
            row.arrange(gx.RIGHT, buff=0)
            table.add(row)
        
        # Arrange rows vertically
        table.arrange(gx.DOWN, buff=0)
        
        # Create horizontal line exactly at top of table
        line = gx.Line(
            start=table.get_left(),
            end=table.get_right(),
            stroke_color=CHILL_BROWN,
            stroke_width=1
        ).move_to(table.get_top(), aligned_edge=gx.DOWN)
        
        # Position headers above line with some spacing
        header_row.next_to(line, gx.UP, buff=0.1)  # Adjust this value to move headers up/down
        
        # Create final group with all elements
        full_visualization_2 = gx.VGroup(header_row, line, table)
        full_visualization_2.scale(0.82)
        full_visualization_2.move_to(gx.ORIGIN)

        #Move samller table over and replace with bigger table. 
        # full_visualization.scale(0.82)
        # full_visualization.move_to(1*UP) #May need to tune this to get smooth transition between tablels. 

        self.play(full_visualization.animate.move_to(0.98*gx.UP).scale(0.82))  #May need to tune this to get smooth transition between tablels.
        self.remove(full_visualization)
        self.add(full_visualization_2)
        self.wait()

        #ZOOM in on layer 21
        self.play(self.animate_view(0, 0, 0, (0.08, -1.3, 0.0), 5.40))
        self.wait()

class P32(WelchScene):
    def construct(self): 

        all_transformer_blocks=gx.VGroup()
        transformer_block_lines=gx.VGroup()

        for i in range(26): #Scale up to 26 once i have stuff kidna working. 
            tb, tb_lines=get_transformer_block(block_orig=np.array([0,i*-14.5,0]),  attention_block_width=8,
                                              attention_block_height=6, attention_block_depth=1, mlp_block_width=8, 
                                              mlp_block_height=6, mlp_block_depth=1)
            all_transformer_blocks.add(tb)
            transformer_block_lines.add(tb_lines)
        self.add(all_transformer_blocks)
        self.set_view(-89, 59, 0, (47.24, -192.2, -24.03), 284.51)
        # self.wait(1)

        self.play(self.animate_view(-35, 66, 0, (30.84, -266.0, -17.62), 62.17), run_time=4)
        # self.wait(1)
        for i in range(21,26): self.remove(all_transformer_blocks[i])

        residual_compute_block_spacing=11
        mlp_out_matrix=get_image_and_border(path='gemma_cached_images_dec_19_2/hook_mlp_out_21.png', scale=1.5)
        mlp_out_matrix.rotate(90*gx.DEGREES, np.array([1,0,0]))
        mlp_out_matrix.move_to(np.array([residual_compute_block_spacing, -14.5*20-12, 0]))
        self.add(mlp_out_matrix)
        # self.wait()
        self.play(self.animate_view(-0.8890242898859254, 78.324221446455, -1.6896696745274932e-15, (11.4431, -260.7836, -8.4289), 45.2850) , run_time=2)
        self.wait()

        # Ok at this point, I'll call out invididual pixel indices with an illustrator overlay
        # And maybe cut to live action. 
        # Ok so atually now that I'm thinking about it, I think i cut to a new 2d scenet
        # That will save a bunch of time
        # And I can transition to it in Premiere - just moving the image over. I'll just do this to make it easier:
        # self.play(FadeOut(all_transformer_blocks[:21]), run_time=2)
        self.remove(all_transformer_blocks)
        self.wait()


        #Get more precise cameara position - should make this into a helper
        # center = self.camera.frame.get_center()
        # height = self.camera.frame.height
        # angles = self.view_angles() * gx.DEGREES

        # call = f"reorient("
        # theta, phi, gamma = (angles / DEGREES) #.astype(int)
        # call += f"{theta}, {phi}, {gamma}"
        # if any(center != 0):
        #     call += f", {tuple(np.round(center, 4))}"
        # if height != FRAME_HEIGHT:
        #     call += ", {:.4f}".format(height)
        # call += ")"
        # print(call)
        # pyperclip.copy(call)

class P33(WelchScene):
    def construct(self): 

        top_tokens_default={'important': 0.2021,
                             'much': 0.125,
                             'high': 0.1116,
                             'low': 0.108,
                             'questionable': 0.0948,
                             'poor': 0.0547,
                             'good': 0.0455,
                             'well': 0.0196,
                             'controversial': 0.0187,
                             'often': 0.0142}
        top_tokens_clamp={'important': 0.2263,
                         'high': 0.125,
                         'much': 0.1136,
                         'low': 0.0902,
                         'questionable': 0.0757,
                         'good': 0.0572,
                         'poor': 0.0366,
                         'well': 0.0226,
                         'controversial': 0.0196,
                         'often': 0.0138
                         }
        top_tokens_reverse_clamp={'much': 0.1862,
                         'important': 0.151,
                         'low': 0.1136,
                         'questionable': 0.1066,
                         'poor': 0.0887,
                         'high': 0.0692,
                         'good': 0.0299,
                         'often': 0.0186,
                         'well': 0.0165,
                         'dependent': 0.0142}


        t=gx.Text('The reliability of Wikipedia is very', font="Myriad Pro").set_color(CHILL_BROWN)
        t.scale(0.7)
        t.to_edge(gx.LEFT, buff=0.5)
        t.shift(1.5*gx.UP)
        self.add(t)

        word_group, bar_group, prob_group=show_top_tokens(top_tokens_default, bar_length_scale = 3, colormap_scale=3, word_bar_buffer=0.8, bar_prob_buffer=0.3)
        default=gx.VGroup(word_group, bar_group, prob_group)
        default.next_to(t, gx.RIGHT, buff=0.2, aligned_edge=(gx.config.frame_height / 2 * gx.UP)).shift(0.01*gx.DOWN)
        self.add(default)

        #Ok, now I think it's going to be dope to add the colormap aboce the defualt group, maybe colorbar, we'll see, and a label. 
        # "Default layer 21 outputs"
        mlp_out_matrix=get_image_and_border(path='gemma_cached_images_dec_19_3/hook_mlp_out_21.png')
        mlp_out_matrix_colorbar=image_at_height(Path(data_dir) / 'gemma_cached_images_dec_19_3/hook_mlp_out_21_colorbar.png')
        mlp_out_matrix.scale(0.4).next_to(default, gx.UP, buff=0.5)
        mlp_out_matrix_colorbar.scale(0.55).next_to(mlp_out_matrix, gx.RIGHT, buff=0.1)
        self.add(mlp_out_matrix, mlp_out_matrix_colorbar)

        title_1=gx.Text("DEFAULT LAYER 21 OUT", font="Myriad Pro").set_color(CHILL_BROWN)
        title_1.scale(0.55)
        title_1.next_to(mlp_out_matrix, gx.UP, buff=0.25)
        self.add(title_1)

        self.set_view(0, 0, 0, (1.0, 0.29, 0.0), 10.00)
        self.wait()


        word_group, bar_group, prob_group=show_top_tokens(top_tokens_clamp, bar_length_scale = 3, colormap_scale=3, word_bar_buffer=0.8, bar_prob_buffer=0.3)
        clamped=gx.VGroup(word_group, bar_group, prob_group)
        clamped.next_to(default, gx.RIGHT, buff=1.0, aligned_edge=(gx.config.frame_height / 2 * gx.UP)).shift(0.01*gx.DOWN)
        self.add(clamped)

        mlp_out_matrix=get_image_and_border(path='gemma_cached_images_dec_19_3/hook_mlp_out_clamped_21.png')
        mlp_out_matrix_colorbar=image_at_height(Path(data_dir) / 'gemma_cached_images_dec_19_3/hook_mlp_out_clamped_21_colorbar.png')
        mlp_out_matrix.scale(0.4).next_to(clamped, gx.UP, buff=0.5)
        mlp_out_matrix_colorbar.scale(0.55).next_to(mlp_out_matrix, gx.RIGHT, buff=0.1)
        self.add(mlp_out_matrix, mlp_out_matrix_colorbar)

        title_2=gx.Text("CLAMPED LAYER 21 OUT", font="Myriad Pro").set_color(CHILL_BROWN)
        title_2.scale(0.55)
        title_2.next_to(mlp_out_matrix, gx.UP, buff=0.25)
        self.add(title_2)
        self.wait()


        self.play(self.animate_view(0, 0, 0, (2.71, 0.62, 0.0), 10.89), run_time=2)

        word_group, bar_group, prob_group=show_top_tokens(top_tokens_reverse_clamp, bar_length_scale = 3, colormap_scale=3, word_bar_buffer=1.4, bar_prob_buffer=0.3)
        reverse_clamped=gx.VGroup(word_group, bar_group, prob_group)
        reverse_clamped.next_to(clamped, gx.RIGHT, buff=1.0, aligned_edge=(gx.config.frame_height / 2 * gx.UP)).shift(0.01*gx.DOWN)
        self.add(reverse_clamped)

        mlp_out_matrix=get_image_and_border(path='gemma_cached_images_dec_19_3/hook_mlp_out_reverse_clamped_21.png')
        mlp_out_matrix_colorbar=image_at_height(Path(data_dir) / 'gemma_cached_images_dec_19_3/hook_mlp_out_reverse_clamped_21_colorbar.png')
        mlp_out_matrix.scale(0.4).next_to(reverse_clamped, gx.UP, buff=0.5)
        mlp_out_matrix_colorbar.scale(0.55).next_to(mlp_out_matrix, gx.RIGHT, buff=0.1)
        self.add(mlp_out_matrix, mlp_out_matrix_colorbar)

        title_3=gx.Text("REVERSE CLAMPED LAYER 21 OUT", font="Myriad Pro").set_color(CHILL_BROWN)
        title_3.scale(0.55)
        title_3.next_to(mlp_out_matrix, gx.UP, buff=0.25)
        self.add(title_3)
        self.wait()


class P35(WelchScene):
    def construct(self): 
        ### ok I'm descoping this one, we're going to pan to layer 21 out, and then just show the resulting text highlighted. 

        all_transformer_blocks=gx.VGroup()
        transformer_block_lines=gx.VGroup()

        for i in range(26): #Scale up to 26 once i have stuff kidna working. 
            tb, tb_lines=get_transformer_block(block_orig=np.array([0,i*-14.5,0]),  attention_block_width=8,
                                              attention_block_height=6, attention_block_depth=1, mlp_block_width=8, 
                                              mlp_block_height=6, mlp_block_depth=1)
            all_transformer_blocks.add(tb)
            transformer_block_lines.add(tb_lines)
        self.add(all_transformer_blocks)
        self.set_view(-89, 59, 0, (47.24, -192.2, -24.03), 284.51)
        self.wait(1)

        self.play(self.animate_view(-35, 66, 0, (30.84, -266.0, -17.62), 62.17), run_time=4)
        # self.wait(1)
        for i in range(21,26): self.remove(all_transformer_blocks[i])

        residual_compute_block_spacing=11
        mlp_out_matrix=get_image_and_border(path='gemma_cached_images_dec_19_2/hook_mlp_out_21.png', scale=1.5)
        mlp_out_matrix.rotate(90*gx.DEGREES, np.array([1,0,0]))
        mlp_out_matrix.move_to(np.array([residual_compute_block_spacing, -14.5*20-12, 0]))
        self.add(mlp_out_matrix)
        self.wait()

        self.play(self.animate_view(-0.8890242898859254, 78.324221446455, -1.6896696745274932e-15, (11.4431, -260.7836, -8.4289), 45.2850) , run_time=2)
        self.wait() 

class P36(WelchScene):
    def construct(self): 
        #This might end up being fairly static, and I think that's ok!
        #Could just look at ways to "animate this one in"

        # print(top_activating_words[:2],top_activating_word_activations[:2])
        words=top_activating_words
        activations=top_activating_word_activations


       # Normalize activations to [0, 1] range for color mapping
        flat_activations = [item for sublist in activations for item in sublist]
        min_act = min(flat_activations)
        max_act = max(flat_activations)
        normalized_activations = [
            [(x - min_act) / (max_act - min_act) for x in row]
            for row in activations
        ]

        # Create text lines with colored backgrounds
        all_line_groups = gx.VGroup()
        
        for line_index in range(1,11):
            text_line=gx.Text(''.join(words[line_index]).replace("\n", " "), font="Helvetica").scale(0.5).set_color('#000000')
            # self.add(text_line)

            starting_char_index=0
            line_highlights=gx.VGroup()
            for i, word in enumerate(words[line_index]):
                # print(word)
                num_chars=len(''.join(word).replace("\n", " ").replace(" ", "")) #This got tricky
                # print(num_chars)
                word_chars=text_line[starting_char_index:starting_char_index+num_chars]
                # print(len(word_chars))
                # print(word_chars.get_left(), word_chars.get_right())
                # print(word_chars.width)
                activation=normalized_activations[line_index][i]
                color=rgb_to_hex(cm.viridis_r(activation)[:3]) #Hec color
                background = gx.Rectangle( height=0.3, width=word_chars.width+0.00, fill_opacity=0.6, stroke_width=0)
                background.set_fill(color)
                background.move_to([word_chars.get_center()[0], 0,0])
                line_highlights.add(background)

                starting_char_index+=num_chars

            line_group=gx.VGroup(line_highlights, text_line)
            # line_group.move_to(UP*(3-0.5*line_index))
            line_group.move_to(gx.DOWN*(5+1.0*line_index)) #Move down below and then we animate up.
            all_line_groups.add(line_group)


        title_1=gx.Text("Gemma 2B Layer 21 MLP Neuron 1393 Top Negative Activating Examples from The Pile Dataset", font="Myriad Pro").set_color(CHILL_BROWN)
        title_1.scale(0.55)
        title_1.to_edge(gx.UP, buff=0.5)
        self.add(title_1)

        color_bar=image_at_height(data_dir+'/top_activating_example_colorbar_dec_19_2.png')
        color_bar.scale(0.65)
        color_bar.to_edge(gx.LEFT, buff=0.4)
        self.add(color_bar)

        color_bar_title=gx.Text('Activation \n Value', font="Myriad Pro").set_color(CHILL_BROWN)
        color_bar_title.scale(0.45)
        color_bar_title.next_to(color_bar, gx.DOWN, buff=0.1)
        color_bar_title.shift(0.2*gx.LEFT)
        self.add(color_bar_title)
        self.wait()

        # self.add(all_line_groups)
        # self.play(ShowCreation(all_line_groups), run_time=5)
        # self.add(all_line_groups)

        # self.play(all_line_groups[0].animate.move_to(UP), run_time=4)

        self.play(*[all_line_groups[line_index].animate.move_to(gx.UP*(3-0.6*(line_index+1))) for line_index in range(10)], run_time=10)
        self.wait(30)

        #not quite what i wanted - but it will work!



        #ok ok ok ok the baseline shift stuff was an annoying time stuck, I think I see a good way forwward
        #Giving bud his medicine and then going for it!


        
        # # Add min/max labels
        # min_label = Text(f"{min_act:.1f}", font_size=20).next_to(color_bar, LEFT)
        # max_label = Text(f"{max_act:.1f}", font_size=20).next_to(color_bar, RIGHT)
        
        # color_bar_group = VGroup(color_bar, min_label, max_label)
        # color_bar_group.to_edge(DOWN, buff=0.5)

        # self.add(color_bar_group)
        



        ## ----


        # starting_char_index=0
        # line_highlights=VGroup()
        # for i, word in enumerate(words[line_index]):
        #     print(word)
        #     num_chars=len(''.join(word).replace("\n", " ").replace(" ", "")) #This got tricky
        #     print(num_chars)
        #     word_chars=text_line[starting_char_index:starting_char_index+num_chars]
        #     # print(len(word_chars))
        #     # print(word_chars.get_left(), word_chars.get_right())
        #     print(word_chars.width)
        #     activation=normalized_activations[line_index][i]
        #     color=rgb_to_hex(cm.viridis_r(activation)[:3]) #Hec color
        #     background = Rectangle( height=0.3, width=word_chars.width+0.00, fill_opacity=0.6, stroke_width=0)
        #     background.set_fill(color)
        #     background.move_to([word_chars.get_center()[0], 0,0])
        #     line_highlights.add(background)

        #     starting_char_index+=num_chars

        # line_group=VGroup(line_highlights, text_line)
        # self.add(line_group)
        # self.wait()


        # for line_idx, (word_line, activation_line) in enumerate(zip(words[1:3], normalized_activations[1:3])):
        #     line_group = VGroup()
            
        #     for word, activation in zip(word_line, activation_line):
        #         # Create text
        #         text = Text(word, font='Helvetica', font_size=24)
        #         has_descenders=any([o in word for o in ['g', 'j', 'p', 'q', 'y']]) #Aww shit we got descenders!
        #         # print(word)
        #         has_ascenders = not has_no_ascenders(word)
        #         # Adjust vertical position based on character composition
        #         if has_descenders and has_ascenders:
        #             # Word has both - might need different adjustment
        #             # text.shift(0.04 * DOWN)
        #             pass
        #         elif has_descenders:
        #             # Only descenders
        #             text.shift(0.2 * UP)
        #         elif has_ascenders:
        #             # No ascenders (and no descenders)
        #             text.shift(0.2 * DOWN)

        #         # if descenders_flag and not no_acscenders_flag:
        #         #     print('#Aww shit we got descenders!')
        #         #     text.shift(0.04*DOWN) #Tuned
                
        #         # if no_acscenders_flag and not descenders_flag:
        #         #     print('#Aww shit we got descenders!')
        #         #     text.shift(0.022*DOWN) #Tuned  

        #         # Create background rectangle
        #         # color = rgb_to_color(cm.viridis(activation)[:3])
        #         color=rgb_to_hex(cm.viridis_r(activation)[:3]) #Hec color
        #         background = Rectangle(
        #             height=0.3, #text.height + 0.2,
        #             width=text.width + 0.1,
        #             fill_opacity=0.6,
        #             stroke_width=0,
        #         )
        #         background.set_fill(color)
                
        #         # Group text and background
        #         word_group = VGroup(background, text)
        #         # word_group.arrange(ORIGIN)
                
        #         line_group.add(word_group)
            
        #     # Arrange words horizontally
        #     line_group.arrange(RIGHT, buff=0.0)
            
        #     # Position the line vertically
        #     line_group.shift(0.4*UP * (1 - line_idx))
            
        #     all_text_groups.add(line_group)
        
        # # Center everything
        # all_text_groups.move_to(ORIGIN)

        # self.add(all_text_groups)

        # ## Ok so manim is losing the leading spaces, not sure how much to worry about that just yet
        # ## Bigger problem is DEFINITELY vertical alignment, let me see if i can find a solution

        
        # # Animation
        # # self.play(
        # #     Write(all_text_groups),
        # #     run_time=2
        # # )
        # # self.wait(2)

        # # Add a color bar
        # color_bar = Rectangle(
        #     height=0.3,
        #     width=4,
        #     fill_opacity=1,
        #     stroke_width=1
        # )
        # color_bar.set_shading_gradient(
        #     color1=rgb_to_color(cm.viridis(0)[:3]),
        #     color2=rgb_to_color(cm.viridis(1)[:3])
        # )
        
        # # Add min/max labels
        # min_label = Text(f"{min_act:.1f}", font_size=20).next_to(color_bar, LEFT)
        # max_label = Text(f"{max_act:.1f}", font_size=20).next_to(color_bar, RIGHT)
        
        # color_bar_group = VGroup(color_bar, min_label, max_label)
        # color_bar_group.to_edge(DOWN, buff=0.5)

        # self.add(color_bar_group)
        
        # self.play(
        #     ShowCreation(color_bar_group),
        #     run_time=1
        # )
        # self.wait(2)




# class P30_P31(InteractiveScene):
#     def construct(self): 
#         # Parse the data
#         #Not sure if I want to include the underscores here or not. 
#         data=TOKENS_BY_LAYER

#         # Create the table structure
#         num_rows = len(data)
#         num_cols = 11  # Index + 10 entries per dictionary
        
#         # Calculate cell dimensions
#         cell_height = 0.4
#         cell_width = 1.0
        
#         # Create table background
#         table = VGroup()
        
#         # Get global min and max for color scaling
#         all_values = [value for d in data for value in d.values()]
#         min_val = min(all_values)
#         max_val = max(all_values)
        
#         colormap_scale=1.0
#         cmap = plt.cm.viridis
#         scaled_colormap_max = max_val*colormap_scale #Dull down top word a bit, lol who am I softcap?

#         # Create headers first
#         header_row = VGroup()
        
#         # Layer header for first column
#         layer_header = Text("Layer", font="Helvetica", font_size=26).set_color(CHILL_BROWN)
#         layer_cell = Rectangle(height=cell_height, width=0.6*cell_width, fill_opacity=0.0, stroke_width=0)
#         layer_header_group = VGroup(layer_cell, layer_header)
#         header_row.add(layer_header_group)
        
#         # Top Next Token Predictions header spanning other columns
#         predictions_header = Text("Top Next Token Predictions", font="Helvetica", font_size=26).set_color(CHILL_BROWN)
#         predictions_cell = Rectangle(height=cell_height, width=cell_width*10, fill_opacity=0.0, stroke_width=0)
#         predictions_header_group = VGroup(predictions_cell, predictions_header)
#         header_row.add(predictions_header_group)
        
#         header_row.arrange(RIGHT, buff=0)
        
#         # Create horizontal line
#         line = Line(
#             start=header_row.get_left(),
#             end=header_row.get_right(),
#             stroke_color=CHILL_BROWN,
#             stroke_width=1
#         )
#         line.shift(UP * cell_height/2)  # Move line to top of header row
        

#         # Create cells
#         for i in range(num_rows):
#             row = VGroup()
            
#             # Add index column
#             index_cell = Rectangle(height=cell_height, width=0.6*cell_width, fill_opacity=0.0, stroke_width=0)
#             index_text = Text(str(i+1), font_size=20, font="Helvetica").set_color(CHILL_BROWN)
#             index_cell = VGroup(index_cell, index_text)
#             row.add(index_cell)
            
#             # Add data columns
#             dict_items = list(data[i].items())[:10]  # Take first 10 items
#             for key, value in dict_items:
#                 # Create cell
#                 cell = Rectangle(height=cell_height, width=cell_width, stroke_width=0)
                
#                 # Color cell using viridis colormap
#                 # norm_value = (value - min_val) / (max_val - min_val)
#                 # rgb_color = cm.viridis(norm_value)[:3]
#                 hex_color=rgb_to_hex(cmap(value/scaled_colormap_max)[:3])
#                 cell.set_fill(hex_color, opacity=0.7) #COME BACK AND FIX COLOR
                
#                 # Add text
#                 word_text = Text(key.strip(), font_size=16, font="Helvetica") #Issues with russian in myriad pro in russian
#                 value_text = Text(f"{value:.4f}", font_size=9).set_color('#CCCCCC')
#                 text_group = VGroup(word_text, value_text).arrange(DOWN, buff=0.05)
                
#                 cell_group = VGroup(cell, text_group)
#                 row.add(cell_group)
            
#             # Arrange row horizontally
#             row.arrange(RIGHT, buff=0)
#             table.add(row)
        
#         # Arrange rows vertically
#         table.arrange(DOWN, buff=0)
#          # Create final group with headers, line, and table
#         full_visualization = VGroup(header_row, line, table)
#         full_visualization.arrange(DOWN, buff=0.2)  # Add some space between headers and table
        
#         # Scale everything
#         full_visualization.scale(0.6)
        
#         self.add(full_visualization)
#         self.wait()




        # import matplotlib.font_manager

        # # Get all font names
        # font_names = [f.name for f in matplotlib.font_manager.fontManager.ttflist]

        # # Print them in a sorted list
        # for font in sorted(set(font_names)):
        #     print(font)













