import manimgx as gx
from welch_scene import WelchScene, image_at_height
from welch_text import math_tex

import numpy as np
import pandas as pd
import ast
import random

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
CHILL_PURPLE='#a9a1c8'
FRESH_TAN='#dfd0b9'

def line_intersection(p1, p2, p3, p4):
    A = np.array([
        [p2[0] - p1[0], p3[0] - p4[0]],
        [p2[1] - p1[1], p3[1] - p4[1]]
    ])
    b = np.array([p3[0] - p1[0], p3[1] - p1[1]])
    
    try:
        t, s = np.linalg.solve(A, b)
        intersection = p1 + t * (p2 - p1)
        return intersection
    except np.linalg.LinAlgError:
        return None

class P24(WelchScene):
    def construct(self):
        image_path='/Users/stephen/manim/videos/_2025/sora/'
        svg = gx.SVGMobject(image_path+"p17_25_10.svg")[1:].scale(4)
        # svg_arrow_fix = SVGMobject(image_path+"p24_background_2.svg")[1:].scale(4)
        

        top_left_arrow = gx.VGroup(svg[0], svg[1])
        top_right_arrow = gx.VGroup(svg[2], svg[3])
        bottom_left_arrow = gx.VGroup(svg[4], svg[5])
        bottom_right_arrow = gx.VGroup(svg[6], svg[7])

        arrows = gx.VGroup(
            top_left_arrow,
            top_right_arrow,
            bottom_left_arrow,
            bottom_right_arrow
        )

        top_curve_arrow = gx.VGroup(svg[13], svg[14])
        bottom_curve_arrow = gx.VGroup(svg[9], svg[10])

        # self.wait()
        # top_curve_arrow=svg_arrow_fix[4]
        # self.add(top_curve_arrow)


        curve_arrows = gx.VGroup(
            top_curve_arrow,
            bottom_curve_arrow
        )

        top_matches = gx.VGroup(*svg[32:42])
        top_matches_line = svg[31]
        top_matches.move_to([top_matches_line.get_x(), top_matches.get_y(), 0])
        top_matches_group = gx.VGroup(top_matches, top_matches_line)
        top_matches_group.move_to([0, top_matches_group.get_y(), 0])

        cosine_similarity = gx.VGroup(*svg[15:31]).scale(0.9)

        border_box = svg[11]

        gx.VGroup(arrows, curve_arrows, cosine_similarity, border_box).shift(gx.RIGHT)

        cat = image_at_height(image_path+"n02123045_1955.jpg").scale(0.5).next_to(top_left_arrow, gx.LEFT, buff=0.25).shift(gx.LEFT * 0.5)

        buff = 0.25
        cat_list = math_tex(r"[0.21,\ -0.12,\ \ldots\ 0.71]").scale_to_fit_width(((top_right_arrow.get_right()[0] - top_curve_arrow.get_left()[0]) * -1) - buff * 2).set_color(CHILL_GREEN)
        midpoint_x = (top_right_arrow.get_right()[0] + top_curve_arrow.get_left()[0]) / 2
        cat_list.move_to([midpoint_x, top_right_arrow.get_y(), 0])

        df = pd.read_csv(image_path+"cat_similarity.csv")
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        cosine_similarity = df[['text', 'similarity', 'embedding']].values.tolist() 

        np.random.seed(2)
        np.random.shuffle(cosine_similarity)

        left_align = gx.Line(top_matches_line.get_center(), top_matches_line.get_center() + gx.UP * 50000).shift(gx.DOWN * 2).shift(gx.LEFT * 0.9)
        right_align = gx.Line(top_matches_line.get_center(), top_matches_line.get_center() + gx.UP * 50000).shift(gx.DOWN * 2).shift(gx.RIGHT * 0.9)

        gap = 0.48
        text_points = []
        similarity_points = []

        for i in range(300):
            text_point = gx.Point().move_to(line_intersection(top_matches_line.get_left(), top_matches_line.get_right(), left_align.get_top(), left_align.get_bottom())).shift(gx.DOWN * (gap * (i + 1)))
            similarity_point = gx.Point().move_to(line_intersection(top_matches_line.get_left(), top_matches_line.get_right(), right_align.get_top(), right_align.get_bottom())).shift(gx.DOWN * (gap * (i + 1)))
            text_points.append(text_point)
            similarity_points.append(similarity_point)

        labels = gx.VGroup(
            gx.Text("1.", font='Myriad Pro', font_size=36).set_color(CHILL_BROWN).move_to([top_matches_line.get_x(), text_points[0].get_y(), 0]).scale(0.8),
            gx.Text("2.", font='Myriad Pro', font_size=36).set_color(CHILL_BROWN).move_to([top_matches_line.get_x(), text_points[1].get_y(), 0]).scale(0.8),
            gx.Text("3.", font='Myriad Pro', font_size=36).set_color(CHILL_BROWN).move_to([top_matches_line.get_x(), text_points[2].get_y(), 0]).scale(0.8)
        ).shift(gx.LEFT * 2)

        self.set_view(0, 0, 0, (-0.13, -0.06, 0.0), 8.30)
        self.add(svg, cat, cat_list) #, labels)


        
        count = 0
        seen = []  # Now stores [main_word_mobject, similarity_mobject, original_text]
        current_similarity_mobject = None

        cosine_similarity[0][0] = 'dog'

        for pair in cosine_similarity: #[:6]:
            text, similarity, embedding = pair
            count += 1
            article = "an" if str(text).strip().lower()[0] in "aeiou" else "a"
            if str(text).strip().lower()[-1] == 's': article=""
            prefix = f'"A photo of {article} '
            main_word = str(text).strip()
            full_sentence = prefix + main_word + '"'

            full_text = gx.Text(full_sentence, font='Myriad Pro', font_size=40).set_color(CHILL_BROWN).scale_to_fit_width(3)
            main_word_mobject = gx.Text(main_word, font='Myriad Pro').set_color(CHILL_BROWN).scale(0.8)
            full_text.move_to([cat.get_x(), bottom_left_arrow.get_y(), 0])
            main_word_mobject.move_to([full_text.get_right()[0], bottom_left_arrow.get_y(), 0], aligned_edge=gx.RIGHT)

            similarity_mobject = math_tex(f"{similarity:.3f}", font_size=38).set_color(FRESH_TAN).move_to(border_box.get_center()).scale_to_fit_width(0.88)
            rand_vals = [round(random.uniform(-1, 1), 2) for _ in range(3)]
            embedding_mobject = math_tex(rf"[{rand_vals[0]},\ {rand_vals[1]},\ \ldots\ {rand_vals[2]}]", font_size=32).set_color(CHILL_PURPLE).move_to([cat_list.get_x(), bottom_right_arrow.get_y(), 0])

            # Remove the previous similarity mobject if it exists
            if current_similarity_mobject is not None:
                self.remove(current_similarity_mobject)
            
            # Add new elements
            self.add(full_text)
            self.add(embedding_mobject)
            self.add(similarity_mobject)
            current_similarity_mobject = similarity_mobject
            self.wait(0.2)

            # Add to seen list - now include the original text for comparison
            seen.append([main_word_mobject, similarity_mobject, main_word])
            ordered = sorted(seen, key=lambda x: float(x[1].get_tex()), reverse=True)
            
            # Check if current word will be in top 3 - now use the stored text
            top3_words = [item[2] for item in ordered[:3]]  # item[2] is the original text
            current_will_stay_in_top3 = main_word in top3_words
            
            print(f"Current word: {main_word}, Top 3: {top3_words}, Will stay: {current_will_stay_in_top3}")
            
            # Set up targets for top 3
            for idx, (w, s, _) in enumerate(ordered[:3]):  # Unpack the third element too
                target_word_point = text_points[idx].get_center()
                target_sim_point = similarity_points[idx].get_center()
                w.generate_target()
                s.generate_target()
                w.target.move_to(target_word_point)
                s.target.move_to(target_sim_point)

            # Remove the 4th item if it exists (the one that gets bumped out)
            if len(ordered) > 3:
                w4, s4, _ = ordered[3]  # Unpack the third element
                self.remove(w4, s4)
            
            # Move top 3 to their new spots
            if current_will_stay_in_top3:
                self.play(
                    *[gx.MoveToTarget(w) for w, s, _ in ordered[:3]],  # Unpack properly
                    *[gx.MoveToTarget(s) for w, s, _ in ordered[:3]]
                )
                
            # Remove temporary elements
            self.remove(embedding_mobject, full_text)
            
            # If current similarity will stay in top 3, don't mark it for removal
            if current_will_stay_in_top3:
                current_similarity_mobject = None  # It's now managed by the ranking system

        # Clean up any remaining similarity mobject at the end
        if current_similarity_mobject is not None:
            self.remove(current_similarity_mobject)

        self.wait()


        self.wait(5)
        # Scene authoring stops here in the interactive editor.

        # for pair in cosine_similarity:
        #     text, similarity, embedding = pair
        #     count += 1
        #     article = "an" if str(text).strip().lower()[0] in "aeiou" else "a"
        #     prefix = f'"A photo of {article} '
        #     main_word = str(text).strip()
        #     full_sentence = prefix + main_word + '"'

        #     full_text = Text(full_sentence, font='Myriad Pro').set_color(CHILL_BROWN).scale_to_fit_width(3)
        #     main_word_mobject = Text(main_word, font='Myriad Pro').set_color(CHILL_BROWN).scale(0.8)
        #     full_text.move_to([cat.get_x(), bottom_left_arrow.get_y(), 0])
        #     main_word_mobject.move_to([cat.get_x(), bottom_left_arrow.get_y(), 0])

        #     similarity_mobject = Tex(str(round(similarity, 3))).set_color(FRESH_TAN).move_to(border_box.get_center()).scale_to_fit_width(1)
        #     rand_vals = [round(random.uniform(-1, 1), 2) for _ in range(3)]
        #     embedding_mobject = Tex(rf"[{rand_vals[0]},\ {rand_vals[1]},\ \ldots\ {rand_vals[2]}]").set_color(CHILL_PURPLE).move_to([cat_list.get_x(), bottom_right_arrow.get_y(), 0]).scale_to_fit_width(cat_list.width)
            
        #     if remove:
        #         self.remove(last_pair)
                
        #     last_pair = VGroup(main_word_mobject, similarity_mobject, embedding_mobject)
            
        #     if count < 4:
        #         self.play(FadeIn(main_word_mobject))
        #         self.play(ReplacementTransform(main_word_mobject.copy(), embedding_mobject))
        #         self.play(ReplacementTransform(VGroup(embedding_mobject.copy(), cat_list.copy()), similarity_mobject))
                
        #     else:
        #         self.add(main_word_mobject, similarity_mobject, embedding_mobject)

        #     seen.append([main_word_mobject, similarity_mobject])
        #     ordered = sorted(seen, key=lambda x: float(x[1].get_tex()), reverse=True)
            
        #     top5_words = [w[0].text for w in ordered[:3]]
        #     if text not in top5_words:
        #         remove = True

        #     for idx, (w, s) in enumerate(ordered[:3]):
        #         target_word_point = text_points[idx].get_center()
        #         target_sim_point = similarity_points[idx].get_center()
        #         w.generate_target()
        #         s.generate_target()
        #         w.target.move_to(target_word_point)
        #         s.target.move_to(target_sim_point)

        #     # Move top 4 to their new spots
            
            
            
        #     # Move top 5 to their new spots (including the 5th)
        #     # Remove the previous 5th spot if it exists (before moving)
        #     if len(ordered) > 3:
        #         w5, s5 = ordered[3]
        #         self.remove(w5, s5)
        #     self.play(
        #         *[MoveToTarget(w) for w, s in ordered[:3]],
        #         *[MoveToTarget(s) for w, s in ordered[:3]]
        #     )

        #     # Remove the previous 5th spot if it exists
            

        #     self.wait(0.2)




