from welch_paths import welch_path
from pathlib import Path

import manimgx as gx
from welch_scene import WelchScene, image_at_height
from welch_text import math_tex

import numpy as np
import pandas as pd
 
HERE = Path(__file__).resolve().parent

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
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

class P20(WelchScene):
    def construct(self):
        stephen_no_hat_p20 = image_at_height(HERE / 'me_no_hat_cropped_1.jpeg').scale(0.55)
        stephen_hat_p20 = image_at_height(HERE / 'me_with_hat.jpeg').scale(0.55)

        gx.Group(stephen_no_hat_p20, stephen_hat_p20).arrange(gx.DOWN, buff=1).shift(gx.LEFT * 5.5)
        
        arrows = gx.SVGMobject(HERE / 'p_20_21_to_manim-03.svg')[1:].scale(6)
        top_left_arrow = arrows[0].next_to(stephen_no_hat_p20, gx.RIGHT)
        bottom_left_arrow = arrows[1].next_to(stephen_hat_p20, gx.RIGHT)
        
        top_image_encoder_text = gx.SVGMobject(HERE / 'top_image_encoder.svg')[3:].scale(5).next_to(top_left_arrow, gx.RIGHT)
        bottom_image_encoder_text = gx.SVGMobject(HERE / 'bottom_image_encoder.svg')[3:].scale(5).next_to(bottom_left_arrow, gx.RIGHT)
        
        
        
        scale_factor = 0.002
        points = [
            scale_factor * np.array([493.37, 196.07, 0]),
            scale_factor * np.array([493.37, 458.95, 0]),
            scale_factor * np.array([670.56, 415.78, 0]),
            scale_factor * np.array([670.56, 240.21, 0]),
        ]

        rhombus = gx.Polygon(*points,
                          fill_color="#6c946f",
                          fill_opacity=0.2,
                          stroke_color="#6c946f",
                          stroke_opacity=1,
                          stroke_width=2).scale(5)
        
        top_image_encoder_outer = rhombus.copy().move_to(top_image_encoder_text.get_center())

        bottom_image_encoder_outer = rhombus.copy().move_to(bottom_image_encoder_text.get_center())
        
        top_image_encoder = gx.VGroup(top_image_encoder_outer, top_image_encoder_text)

        bottom_image_encoder = gx.VGroup(bottom_image_encoder_outer, bottom_image_encoder_text)
        
        top_right_arrow = arrows[2].next_to(top_image_encoder, gx.RIGHT)
        bottom_right_arrow = arrows[3].next_to(bottom_image_encoder, gx.RIGHT)
        
        stephen_no_hat_equation = math_tex(r"I_{man} = [-0.13\ -0.10\ \cdots\ -0.56]").set_color(GREEN).next_to(top_right_arrow, gx.RIGHT).scale(0.9).shift(gx.LEFT * 0.3)

        stephen_hat_equation = math_tex(r"\hat{I}_{man} = [-0.13\ -0.10\ \cdots\ -0.56]").set_color(GREEN).next_to(bottom_right_arrow, gx.RIGHT).scale(0.9).shift(gx.LEFT * 0.3)

        self.play(gx.FadeIn(stephen_hat_p20), gx.FadeIn(stephen_no_hat_p20))
        self.play(gx.GrowFromEdge(top_left_arrow, gx.LEFT), gx.GrowFromEdge(bottom_left_arrow, gx.LEFT))
        self.play(gx.Create(top_image_encoder), gx.Create(bottom_image_encoder))
        self.play(gx.GrowFromEdge(top_right_arrow, gx.LEFT), gx.GrowFromEdge(bottom_right_arrow, gx.LEFT))
        self.play(gx.Write(stephen_no_hat_equation), gx.Write(stephen_hat_equation))
        # Scene authoring stops here in the interactive editor.

class P21(WelchScene):
    def construct(self):
        x_axis = WelchXAxis(0, 8).move_to(gx.ORIGIN)
        y_axis = WelchYAxis(0, 8).move_to(gx.ORIGIN)
        x_axis.ticks.set_opacity(0)
        x_axis.labels.set_opacity(0)
        y_axis.ticks.set_opacity(0)
        y_axis.labels.set_opacity(0)
        
        axes = gx.VGroup(x_axis, y_axis)
        axes.shift(gx.LEFT * 4.5)
        
        stephen_no_hat = image_at_height(HERE / 'me_no_hat_cropped_1.jpeg').scale(0.3)
        stephen_hat = image_at_height(HERE / 'me_with_hat.jpeg').scale(0.3)
        
        stephen_hat.move_to((y_axis.get_center() + y_axis.get_top())/2).shift(gx.LEFT * 0.6)
        stephen_no_hat.move_to((x_axis.get_center() + x_axis.get_right())/2).shift(gx.DOWN * 0.6)
        
        # Define the angle (in radians) to rotate the arrow up (e.g., 15 degrees)

        # Create the arrows
        stephen_hat_arrow = WelchXAxis(x_min=0, x_max=2.25, axis_length_on_canvas=2.25, stroke_width=5).set_color(GREEN)
        stephen_no_hat_arrow = WelchXAxis(x_min=0, x_max=2.25, axis_length_on_canvas=2.25, stroke_width=5).set_color(GREEN)
        
        
        
        
        stephen_hat_arrow.ticks.set_opacity(0)
        stephen_hat_arrow.labels.set_opacity(0)
        stephen_no_hat_arrow.ticks.set_opacity(0)
        stephen_no_hat_arrow.labels.set_opacity(0)
        
        axes_intersection = line_intersection(
            x_axis.get_axis_line().get_left(), x_axis.get_axis_line().get_right(),
            y_axis.get_axis_line().get_bottom(), y_axis.get_axis_line().get_top()
        )

        stephen_hat_arrow.shift(axes_intersection - stephen_hat_arrow.get_axis_line().get_left())
        stephen_no_hat_arrow.shift(axes_intersection - stephen_no_hat_arrow.get_axis_line().get_left())
        
        stephen_hat_arrow.rotate(65 * gx.DEGREES, about_point=axes_intersection)
        stephen_no_hat_arrow.rotate(15 * gx.DEGREES, about_point=axes_intersection)
        
        stephen_delta_arrow_line = gx.Line(stephen_hat_arrow.arrow.get_all_points()[0], stephen_no_hat_arrow.arrow.get_all_points()[0], stroke_width=5, color=YELLOW)
        stephen_delta_arrow_arrow = gx.SVGMobject(HERE / 'welch_arrow_tip_1.svg').scale(0.1).next_to(stephen_delta_arrow_line.get_end()).rotate(gx.DEGREES * 310).set_color(YELLOW)
        stephen_delta_arrow_arrow.shift(stephen_no_hat_arrow.arrow.get_all_points()[0] - stephen_delta_arrow_arrow.get_all_points()[0])
        
        stephen_hat_arrow_label = math_tex(r"I_{man}").set_color(GREEN).next_to(stephen_hat_arrow.arrow, gx.UP)
        stephen_no_hat_arrow_label = math_tex(r"\hat{I}_{man}").set_color(GREEN).next_to(stephen_no_hat_arrow.arrow, gx.RIGHT)
                
        start = stephen_hat_arrow.arrow.get_all_points()[0]
        end = stephen_no_hat_arrow.arrow.get_all_points()[0]

        # Vector from start to end
        direction = end - start
        length = np.linalg.norm(direction)

        # Normalize and trim
        trim_amount = 0.1  # Adjust this value as needed
        new_end = end - (direction / length) * trim_amount

        # Create the trimmed line
        stephen_delta_arrow_line = gx.Line(start, new_end, stroke_width=5, color=YELLOW)
        
        stephen_delta_arrow = gx.VGroup(stephen_delta_arrow_line, stephen_delta_arrow_arrow)
        
        stephen_delta_arrow_label = math_tex(r"\hat{I}_{man}-I_{man}").set_color(YELLOW).next_to(stephen_delta_arrow.get_center(), gx.RIGHT).shift(gx.UP * 0.25)

        top_header_line = gx.Line(stroke_width=2).scale_to_fit_width(5.5).set_color(CHILL_BROWN).shift(gx.RIGHT * 3.25 + gx.UP * 2.5)
        
        bottom_header_line = gx.Line(stroke_width=2).scale_to_fit_width(5.5).set_color(CHILL_BROWN).shift(gx.RIGHT * 3.25 + gx.UP * 0.5)
        bottom_header_title = gx.Text('TOP MATCHES', font='Myriad Pro').next_to(bottom_header_line, gx.UP, buff=0.1).set_color(CHILL_BROWN).scale(0.6)
        
        left_align = gx.Line(top_header_line.get_center(), bottom_header_line.get_center(), stroke_width=2).shift(gx.LEFT * 1.4).set_length(15)
        right_align = gx.Line(top_header_line.get_center(), bottom_header_line.get_center(), stroke_width=2).shift(gx.RIGHT * 1.4).set_length(15)
        

        top_header_word_column = gx.Text('WORD', font='Myriad Pro').set_color(CHILL_BROWN).scale(0.6).next_to(line_intersection(top_header_line.get_left(), top_header_line.get_right(), left_align.get_top(), left_align.get_bottom()), gx.UP, buff=0.1)
        top_header_cosine_similarity_column = gx.Text('COSINE SIMILARITY', font='Myriad Pro').set_color(CHILL_BROWN).scale(0.6).next_to(line_intersection(top_header_line.get_left(), top_header_line.get_right(), right_align.get_top(), right_align.get_bottom()), gx.UP, buff=0.1)
        
        df = pd.read_csv("cosine_similarities.csv")
        
        cosine_similarity = df[['text', 'similarity']].values.tolist()
        np.random.shuffle(cosine_similarity)
        
        seen = []
        ordered = seen.copy()
        
        gap = 0.8
        max_width = 2.56
        
        start = gx.Point().move_to(line_intersection(top_header_line.get_left(), top_header_line.get_right(), left_align.get_top(), left_align.get_bottom())).shift(gx.DOWN* gap)

        
        points = []
        for i in range(200):
            point = gx.Point()
            point.move_to(line_intersection(bottom_header_line.get_left(), bottom_header_line.get_right(), left_align.get_top(), left_align.get_bottom())).shift(gx.DOWN * (gap * (i+1)))

            points.append(point)
            
        label_2 = gx.Text("2.", font='Myriad Pro').set_color(CHILL_BROWN).move_to([bottom_header_line.get_x(), points[1].get_y(), 0])
        label_3 = gx.Text("3.", font='Myriad Pro').set_color(CHILL_BROWN).move_to([bottom_header_line.get_x(), points[2].get_y(), 0])
        label_4 = gx.Text("4.", font='Myriad Pro').set_color(CHILL_BROWN).move_to([bottom_header_line.get_x(), points[3].get_y(), 0])
        label_5 = gx.Text("5.", font='Myriad Pro').set_color(CHILL_BROWN).move_to([bottom_header_line.get_x(), points[4].get_y(), 0])
        label_1 = gx.Text("1.", font='Myriad Pro').set_color(CHILL_BROWN).move_to([bottom_header_line.get_x(), points[0].get_y(), 0])
        labels = gx.VGroup(label_1, label_2, label_3, label_4, label_5).shift(gx.LEFT * 3)
            
        similarity_points = []
        for i in range(200):
            point = gx.Point()
            point.move_to(line_intersection(bottom_header_line.get_left(), bottom_header_line.get_right(), right_align.get_top(), right_align.get_bottom())).shift(gx.DOWN * (gap * (i+1)))

            similarity_points.append(point)


        self.add(top_header_line)
        self.add(top_header_word_column)
        self.add(top_header_cosine_similarity_column)
        self.add(bottom_header_line)
        self.add(bottom_header_title)
        self.add(labels)
        self.add(top_header_line, bottom_header_line)
        self.add(axes)
        self.add(stephen_hat, stephen_no_hat)
        self.add(stephen_hat_arrow, stephen_no_hat_arrow)
        self.add(stephen_hat_arrow_label, stephen_no_hat_arrow_label, stephen_delta_arrow_label)
        self.add(stephen_delta_arrow)
        count = 0
        seen = []
        remove = False
        last_pair = None
        for pair in cosine_similarity:
            count += 1
            word, similarity = pair

            word_mobject = gx.Text(word, font='Myriad Pro').set_color(FRESH_TAN).scale(0.9)
            similarity_mobject = math_tex(str(round(similarity, 3))).set_color(FRESH_TAN).scale(0.9).move_to([right_align.get_x(), start.get_y(), 0])
            word_mobject.move_to(start.get_center())
            
            if remove:
                self.remove(last_pair)
                
            last_pair = gx.VGroup(word_mobject, similarity_mobject)
            
            if count < 6:
                self.play(gx.FadeIn(word_mobject))
                self.play(gx.ReplacementTransform(word_mobject.copy(), similarity_mobject))
            else:
                self.add(word_mobject, similarity_mobject)

            seen.append([word_mobject, similarity_mobject])
            ordered = sorted(seen, key=lambda x: float(x[1].get_tex()), reverse=True)
            
            top5_words = [w[0].text for w in ordered[:5]]
            if word not in top5_words:
                remove = True

            for idx, (w, s) in enumerate(ordered[:5]):
                target_word_point = points[idx].get_center()
                target_sim_point = similarity_points[idx].get_center()
                w.generate_target()
                s.generate_target()
                w.target.move_to(target_word_point)
                s.target.move_to(target_sim_point)

            # Move top 4 to their new spots
            
            
            
            # Move top 5 to their new spots (including the 5th)
            # Remove the previous 5th spot if it exists (before moving)
            if len(ordered) > 5:
                w5, s5 = ordered[5]
                self.remove(w5, s5)
            self.play(
                *[gx.MoveToTarget(w) for w, s in ordered[:5]],
                *[gx.MoveToTarget(s) for w, s in ordered[:5]]
            )

            # Remove the previous 5th spot if it exists
            

            self.wait(0.2)



            

                
            
            
        # Scene authoring stops here in the interactive editor.

class P20_21v2(WelchScene):
    def construct(self):
        image_path='/Users/stephen/manim/videos/_2025/sora/'
        svg_path=welch_path('sora/graphics/to_manim')
        stephen_no_hat_p20 = image_at_height(image_path+"me_no_hat_cropped_1.jpeg").scale(0.55)
        stephen_hat_p20 = image_at_height(image_path+"me_with_hat.jpeg").scale(0.55)

        # Place both together on the left
        gx.Group(stephen_no_hat_p20, stephen_hat_p20).arrange(gx.DOWN, buff=1).shift(gx.LEFT * 5.5)

        # Save the final position of the hat image
        initial_point = gx.Point([-5.5, -1.6,  0. ])

        # Move hat image directly behind no-hat image and make it invisible
        stephen_hat_p20.move_to(stephen_no_hat_p20.get_center()).set_opacity(0)
        
        
        
        arrows = gx.SVGMobject(image_path+"p_20_21_to_manim-03.svg")[1:].scale(6)
        top_left_arrow = arrows[0].next_to(stephen_no_hat_p20, gx.RIGHT)
        bottom_left_arrow = arrows[1].move_to([top_left_arrow.get_x(), initial_point.get_y(), 0])
        
        top_image_encoder_text = gx.SVGMobject(image_path+"top_image_encoder.svg")[3:].scale(5).next_to(top_left_arrow, gx.RIGHT)
        bottom_image_encoder_text = gx.SVGMobject(image_path+"bottom_image_encoder.svg")[3:].scale(5).next_to(bottom_left_arrow, gx.RIGHT)
        
        
        
        scale_factor = 0.002
        points = [
            scale_factor * np.array([493.37, 196.07, 0]),
            scale_factor * np.array([493.37, 458.95, 0]),
            scale_factor * np.array([670.56, 415.78, 0]),
            scale_factor * np.array([670.56, 240.21, 0]),
        ]

        rhombus = gx.Polygon(*points,
                          fill_color="#6c946f",
                          fill_opacity=0.2,
                          stroke_color="#6c946f",
                          stroke_opacity=1,
                          stroke_width=2).scale(5)
        
        top_image_encoder_outer = rhombus.copy().move_to(top_image_encoder_text.get_center())

        bottom_image_encoder_outer = rhombus.copy().move_to(bottom_image_encoder_text.get_center())
        
        top_image_encoder = gx.VGroup(top_image_encoder_outer, top_image_encoder_text)

        bottom_image_encoder = gx.VGroup(bottom_image_encoder_outer, bottom_image_encoder_text)
        
        top_right_arrow = arrows[2].next_to(top_image_encoder, gx.RIGHT)
        bottom_right_arrow = arrows[3].next_to(bottom_image_encoder, gx.RIGHT)
        
        stephen_no_hat_equation = math_tex(r"I_{man} = [-0.13\ -0.10\ \cdots\ -0.56]").set_color(GREEN).next_to(top_right_arrow, gx.RIGHT).scale(0.9).shift(gx.LEFT * 0.3)

        stephen_hat_equation = math_tex(r"\hat{I}_{man} = [-0.13\ -0.10\ \cdots\ -0.50]").set_color(GREEN).next_to(bottom_right_arrow, gx.RIGHT).scale(0.9).shift(gx.LEFT * 0.3)
        
        x_axis = WelchXAxis(0, 8).move_to(gx.ORIGIN)
        y_axis = WelchYAxis(0, 8).move_to(gx.ORIGIN)
        x_axis.ticks.set_opacity(0)
        x_axis.labels.set_opacity(0)
        y_axis.ticks.set_opacity(0)
        y_axis.labels.set_opacity(0)
        
        axes = gx.VGroup(x_axis, y_axis)
        axes.shift(gx.LEFT * 4.5)
        
        stephen_no_hat = image_at_height(image_path+"me_no_hat_cropped_1.jpeg").scale(0.3)
        stephen_hat = image_at_height(image_path+"me_with_hat.jpeg").scale(0.3)
        
        stephen_hat.move_to((y_axis.get_center() + y_axis.get_top())/2).shift(gx.LEFT * 0.6)
        stephen_no_hat.move_to((x_axis.get_center() + x_axis.get_right())/2).shift(gx.DOWN * 0.6)
        
        # Define the angle (in radians) to rotate the arrow up (e.g., 15 degrees)

        # Create the arrows
        stephen_hat_arrow = WelchXAxis(x_min=0, x_max=2.25, axis_length_on_canvas=2.25, stroke_width=5).set_color(GREEN)
        stephen_no_hat_arrow = WelchXAxis(x_min=0, x_max=2.25, axis_length_on_canvas=2.25, stroke_width=5).set_color(GREEN)
        
        
        
        
        stephen_hat_arrow.ticks.set_opacity(0)
        stephen_hat_arrow.labels.set_opacity(0)
        stephen_no_hat_arrow.ticks.set_opacity(0)
        stephen_no_hat_arrow.labels.set_opacity(0)
        
        axes_intersection = line_intersection(
            x_axis.get_axis_line().get_left(), x_axis.get_axis_line().get_right(),
            y_axis.get_axis_line().get_bottom(), y_axis.get_axis_line().get_top()
        )

        stephen_hat_arrow.shift(axes_intersection - stephen_hat_arrow.get_axis_line().get_left())
        stephen_no_hat_arrow.shift(axes_intersection - stephen_no_hat_arrow.get_axis_line().get_left())
        
        stephen_hat_arrow.rotate(65 * gx.DEGREES, about_point=axes_intersection)
        stephen_no_hat_arrow.rotate(15 * gx.DEGREES, about_point=axes_intersection)
        
        #SW - this arrows direction was backwards
        stephen_delta_arrow_line = gx.Line(stephen_hat_arrow.arrow.get_all_points()[0]+np.array([0.05, -0.05, 0]), stephen_no_hat_arrow.arrow.get_all_points()[0], stroke_width=5, color=YELLOW)
        
        # stephen_delta_arrow_line = Line(stephen_no_hat_arrow.arrow.get_all_points()[0], stephen_hat_arrow.arrow.get_all_points()[0], stroke_width=5, color=YELLOW)
        
        #stephen_delta_arrow_arrow = SVGMobject(image_path+"welch_arrow_tip_1.svg").scale(0.1).next_to(stephen_delta_arrow_line.get_end()).rotate(DEGREES * 310).set_color(YELLOW)
        #stephen_delta_arrow_arrow.shift(stephen_no_hat_arrow.arrow.get_all_points()[0] - stephen_delta_arrow_arrow.get_all_points()[0])
        
        stephen_delta_arrow_arrow = gx.SVGMobject(image_path+"welch_arrow_tip_1.svg").scale(0.1).set_color(YELLOW)
        stephen_delta_arrow_arrow.rotate(gx.DEGREES * 126)
        stephen_delta_arrow_arrow.move_to(stephen_delta_arrow_line.get_start()).shift([0.065, -0.06, 0])

        # stephen_delta_arrow_arrow.shift(stephen_delta_arrow_arrow.get_all_points()[0] - stephen_no_hat_arrow.arrow.get_all_points()[0])

        stephen_hat_arrow_label = math_tex(r"\hat{I}_{man}").set_color(GREEN).next_to(stephen_hat_arrow.arrow, gx.UP)
        stephen_no_hat_arrow_label = math_tex(r"I_{man}").set_color(GREEN).next_to(stephen_no_hat_arrow.arrow, gx.RIGHT)
                
        start = stephen_hat_arrow.arrow.get_all_points()[0]
        end = stephen_no_hat_arrow.arrow.get_all_points()[0]

        # Vector from start to end
        direction = end - start
        length = np.linalg.norm(direction)

        # Normalize and trim
        trim_amount = 0.1  # Adjust this value as needed
        new_end = end - (direction / length) * trim_amount

        # Create the trimmed line
        # stephen_delta_arrow_line = Line(start, new_end, stroke_width=5, color=YELLOW)
        
        stephen_delta_arrow = gx.VGroup(stephen_delta_arrow_line, stephen_delta_arrow_arrow)
        
        stephen_delta_arrow_label = math_tex(r"\hat{I}_{man}-I_{man}").set_color(YELLOW).next_to(stephen_delta_arrow.get_center(), gx.RIGHT).shift(gx.UP * 0.25)

        top_header_line = gx.Line(stroke_width=2).scale_to_fit_width(5.5).set_color(CHILL_BROWN).shift(gx.RIGHT * 3.25 + gx.UP * 2.5)
        
        bottom_header_line = gx.Line(stroke_width=2).scale_to_fit_width(5.5).set_color(CHILL_BROWN).shift(gx.RIGHT * 3.25 + gx.UP * 0.5)
        bottom_header_title = gx.Text('TOP MATCHES', font='Myriad Pro').next_to(bottom_header_line, gx.UP, buff=0.1).set_color(CHILL_BROWN).scale(0.6)
        
        left_align = gx.Line(top_header_line.get_center(), bottom_header_line.get_center(), stroke_width=2).shift(gx.LEFT * 1.4).set_length(15)
        right_align = gx.Line(top_header_line.get_center(), bottom_header_line.get_center(), stroke_width=2).shift(gx.RIGHT * 1.4).set_length(15)
        

        top_header_word_column = gx.Text('WORD', font='Myriad Pro').set_color(CHILL_BROWN).scale(0.6).next_to(line_intersection(top_header_line.get_left(), top_header_line.get_right(), left_align.get_top(), left_align.get_bottom()), gx.UP, buff=0.1)
        top_header_cosine_similarity_column = gx.Text('COSINE SIMILARITY', font='Myriad Pro').set_color(CHILL_BROWN).scale(0.6).next_to(line_intersection(top_header_line.get_left(), top_header_line.get_right(), right_align.get_top(), right_align.get_bottom()), gx.UP, buff=0.1)
        
        df = pd.read_csv(image_path+"cosine_similarities.csv")
        
        cosine_similarity = df[['text', 'similarity']].values.tolist()
        np.random.shuffle(cosine_similarity)
        
        seen = []
        ordered = seen.copy()
        
        gap = 0.8
        max_width = 2.56
        
        start = gx.Point().move_to(line_intersection(top_header_line.get_left(), top_header_line.get_right(), left_align.get_top(), left_align.get_bottom())).shift(gx.DOWN* gap)

        
        points = []
        for i in range(200):
            point = gx.Point()
            point.move_to(line_intersection(bottom_header_line.get_left(), bottom_header_line.get_right(), left_align.get_top(), left_align.get_bottom())).shift(gx.DOWN * (gap * (i+1)))

            points.append(point)
            
        label_2 = gx.Text("2.", font='Myriad Pro').set_color(CHILL_BROWN).move_to([bottom_header_line.get_x(), points[1].get_y(), 0])
        label_3 = gx.Text("3.", font='Myriad Pro').set_color(CHILL_BROWN).move_to([bottom_header_line.get_x(), points[2].get_y(), 0])
        label_4 = gx.Text("4.", font='Myriad Pro').set_color(CHILL_BROWN).move_to([bottom_header_line.get_x(), points[3].get_y(), 0])
        label_5 = gx.Text("5.", font='Myriad Pro').set_color(CHILL_BROWN).move_to([bottom_header_line.get_x(), points[4].get_y(), 0])
        label_1 = gx.Text("1.", font='Myriad Pro').set_color(CHILL_BROWN).move_to([bottom_header_line.get_x(), points[0].get_y(), 0])
        labels = gx.VGroup(label_1, label_2, label_3, label_4, label_5).shift(gx.LEFT * 3)
            
        similarity_points = []
        for i in range(200):
            point = gx.Point()
            point.move_to(line_intersection(bottom_header_line.get_left(), bottom_header_line.get_right(), right_align.get_top(), right_align.get_bottom())).shift(gx.DOWN * (gap * (i+1)))

            similarity_points.append(point)

        self.add(stephen_hat_p20)
        self.play(gx.FadeIn(stephen_no_hat_p20))

        self.wait()

        self.play(
            stephen_hat_p20.animate.move_to([-5.5, -1.6,  0. ]).set_opacity(1)
        )

        self.wait()
        self.add(top_left_arrow, bottom_left_arrow, top_image_encoder, bottom_image_encoder)
        self.wait()
        self.add(top_right_arrow, bottom_right_arrow, stephen_no_hat_equation, stephen_hat_equation)
        self.wait()
        self.play(gx.FadeOut(arrows), gx.FadeOut(top_image_encoder), gx.FadeOut(bottom_image_encoder))
        self.wait()
        self.play(gx.FadeIn(axes),
                  gx.ReplacementTransform(stephen_hat_p20, stephen_hat),
                  gx.ReplacementTransform(stephen_hat_equation[0:5], stephen_hat_arrow_label),
                  gx.ReplacementTransform(stephen_hat_equation[6:], stephen_hat_arrow[0]),
                  gx.FadeOut(stephen_hat_equation[5]),
                  stephen_no_hat_p20.animate.shift(gx.DOWN * 4 + gx.LEFT * 0.3).set_opacity(0.5), run_time = 3)

        # self.play(ReplacementTransform(stephen_hat_equation[6:].copy(), stephen_hat_arrow[0]), run_time=3)

        self.wait()
        self.play(gx.ReplacementTransform(stephen_no_hat_p20, stephen_no_hat),
                  gx.FadeOut(stephen_no_hat_equation[4]),
                  gx.ReplacementTransform(stephen_no_hat_equation[0:4],  stephen_no_hat_arrow_label),
                  gx.ReplacementTransform(stephen_no_hat_equation[5:], stephen_no_hat_arrow[0]), #Moving to WelchAxis is weird, first index helps with animation
                  run_time=3)

        
        self.wait()
        self.play(gx.FadeIn(stephen_delta_arrow))
        self.wait()
        self.play(gx.Write(stephen_delta_arrow_label))
        self.wait()
        self.play(gx.FadeIn(top_header_line), gx.DrawBorderThenFill(top_header_word_column), gx.DrawBorderThenFill(top_header_cosine_similarity_column))
        self.wait()
        self.play(gx.ReplacementTransform(top_header_line.copy(), bottom_header_line), gx.ReplacementTransform(gx.VGroup(top_header_word_column.copy(), top_header_cosine_similarity_column.copy()), bottom_header_title)),
        self.wait()
        self.play(gx.Write(labels))
        self.wait()
        
        count = 0
        seen = []
        remove = False
        last_pair = None
        for pair in cosine_similarity:
            count += 1
            word, similarity = pair

            word_mobject = gx.Text(word, font='Myriad Pro').set_color(FRESH_TAN).scale(0.9)
            similarity_mobject = math_tex(str(round(similarity, 3))).set_color(FRESH_TAN).scale(0.9).move_to([right_align.get_x(), start.get_y(), 0])
            word_mobject.move_to(start.get_center())
            
            if remove:
                self.remove(last_pair)
                
            last_pair = gx.VGroup(word_mobject, similarity_mobject)
            
            if count < 6:
                self.play(gx.FadeIn(word_mobject))
                self.play(gx.ReplacementTransform(word_mobject.copy(), similarity_mobject))
            else:
                self.add(word_mobject, similarity_mobject)

            seen.append([word_mobject, similarity_mobject])
            ordered = sorted(seen, key=lambda x: float(x[1].get_tex()), reverse=True)
            
            top5_words = [w[0].text for w in ordered[:5]]
            if word not in top5_words:
                remove = True

            for idx, (w, s) in enumerate(ordered[:5]):
                target_word_point = points[idx].get_center()
                target_sim_point = similarity_points[idx].get_center()
                w.generate_target()
                s.generate_target()
                w.target.move_to(target_word_point)
                s.target.move_to(target_sim_point)

            # Move top 4 to their new spots
            
            
            
            # Move top 5 to their new spots (including the 5th)
            # Remove the previous 5th spot if it exists (before moving)
            if len(ordered) > 5:
                w5, s5 = ordered[5]
                self.remove(w5, s5)
            self.play(
                *[gx.MoveToTarget(w) for w, s in ordered[:5]],
                *[gx.MoveToTarget(s) for w, s in ordered[:5]]
            )

            # Remove the previous 5th spot if it exists
            

            self.wait(0.2)
        


def generate_nice_ticks(min_val, max_val, min_ticks=3, max_ticks=16, ignore=[0]):
    """
    Generate a list of nice-looking tick values between min_val and max_val,
    and return extended range values for the full axis.
    
    Args:
        min_val (float): Minimum value for the data range
        max_val (float): Maximum value for the data range
        min_ticks (int): Minimum number of ticks desired
        max_ticks (int): Maximum number of ticks desired
        ignore (list): List of values to exclude from the ticks
        
    Returns:
        tuple: (tick_values, axis_min, axis_max)
            - tick_values (list): A list of tick values
            - axis_min (float): Suggested minimum value for the axis (one tick before min_val)
            - axis_max (float): Suggested maximum value for the axis (one tick after max_val)
    """
    # Ensure min_val < max_val
    if min_val > max_val:
        min_val, max_val = max_val, min_val
        
    # Handle case where min_val and max_val are equal or very close
    if abs(max_val - min_val) < 1e-10:
        # Create a small range around the value
        min_val = min_val - 1
        max_val = max_val + 1
    
    # Find the appropriate order of magnitude for the tick spacing
    range_val = max_val - min_val
    power = np.floor(np.log10(range_val))
    
    # Try different multiples of the base power of 10
    possible_step_sizes = [10**power, 5 * 10**(power-1), 2 * 10**(power-1), 10**(power-1)]
    
    # Find the first step size that gives us fewer than max_ticks
    chosen_step = possible_step_sizes[0]  # Default to the largest step
    
    for step in possible_step_sizes:
        # Calculate how many ticks we'd get with this step size
        first_tick = np.ceil(min_val / step) * step
        last_tick = np.floor(max_val / step) * step
        
        # Count ticks, excluding ignored values
        num_ticks = 0
        current = first_tick
        while current <= last_tick * (1 + 1e-10):
            if not any(abs(current - ignored_val) < 1e-10 for ignored_val in ignore):
                num_ticks += 1
            current += step
        
        if min_ticks <= num_ticks <= max_ticks:
            chosen_step = step
            break
        elif num_ticks > max_ticks:
            # If we have too many ticks, stop and use the previous step size
            break
    
    # Calculate the first tick at or below min_val
    first_tick = np.floor(min_val / chosen_step) * chosen_step
    
    # Calculate the last tick at or above max_val
    last_tick = np.ceil(max_val / chosen_step) * chosen_step
    
    # Calculate one tick before first_tick for axis_min
    axis_min = first_tick - chosen_step
    
    # Calculate one tick after last_tick for axis_max
    axis_max = last_tick + chosen_step
    
    # Generate the tick values that fall within the data range, excluding ignored values
    ticks = []
    current = np.ceil(min_val / chosen_step) * chosen_step
    
    while current <= max_val * (1 + 1e-10):  # Add a small epsilon to handle floating point errors
        # Only add the tick if it's not in the ignore list
        if not any(abs(current - ignored_val) < 1e-10 for ignored_val in ignore):
            ticks.append(float(current))  # Convert to float to avoid numpy types
        current += chosen_step
    
    # If we still have too few ticks, try the next smaller step size
    if len(ticks) < min_ticks and possible_step_sizes.index(chosen_step) < len(possible_step_sizes) - 1:
        return generate_nice_ticks(min_val, max_val, min_ticks, max_ticks, ignore)
    
    return ticks, float(axis_min), float(axis_max)

class WelchXAxis(gx.VGroup):
    def __init__(
        self,
        x_min=0,
        x_max=6, 
        x_ticks=[1, 2, 3, 4, 5],  # Default tick values
        x_tick_height=0.15,        # Default tick height
        x_label_font_size=24,     # Default font size
        stroke_width=3,           # Default stroke width
        color=CHILL_BROWN,        # Default color (using predefined BROWN)
        arrow_tip_scale=0.1, 
        axis_length_on_canvas=5,
        include_tip=True,
        **kwargs
    ):
        
        gx.VGroup.__init__(self, **kwargs)

        # Store parameters
        self.x_ticks = x_ticks
        self.x_tick_height = x_tick_height
        self.x_label_font_size = x_label_font_size
        self.stroke_width = stroke_width
        self.axis_color = color
        self.arrow_tip_scale=arrow_tip_scale
        self.x_min = x_min
        self.x_max = x_max
        self.axis_length_on_canvas=axis_length_on_canvas
        self.include_tip=include_tip

        self.axis_to_canvas_scale=(self.x_max-self.x_min)/axis_length_on_canvas
        self.x_ticks_scaled=(np.array(x_ticks)-self.x_min)/self.axis_to_canvas_scale

        # Create the basic components
        self._create_axis_line()
        self._create_ticks()
        self._create_labels()
        
    def _create_axis_line(self):
        
        # Create a line for the x-axis
        axis_line = gx.Line(
            start=np.array([0, 0, 0]),
            end=np.array([self.axis_length_on_canvas, 0, 0]),
            color=self.axis_color,
            stroke_width=self.stroke_width
        )
        if self.include_tip:
            arrow_tip=gx.SVGMobject('/Users/stephen/manim/videos/_2025/sora/welch_arrow_tip_1.svg') #Dumb hardcode for now
            arrow_tip.scale(self.arrow_tip_scale)
            arrow_tip.move_to([self.axis_length_on_canvas, 0, 0])
            axis_line = gx.VGroup(axis_line, arrow_tip)

        self.axis_line = axis_line
        self.arrow = arrow_tip
        self.add(axis_line)
        
    def _create_ticks(self):
        self.ticks = gx.VGroup()
        
        for x_val in self.x_ticks_scaled:
            tick = gx.Line(
                start=np.array([x_val, 0, 0]),
                end=np.array([x_val, -self.x_tick_height, 0]),  # Ticks extend downward
                color=self.axis_color,
                stroke_width=self.stroke_width
            )
            self.ticks.add(tick)
            
        self.add(self.ticks)
        
    def _create_labels(self):
        self.labels = gx.VGroup()
        
        for x_val, x_val_label in zip(self.x_ticks_scaled, self.x_ticks):
            # In 3B1B's manim, use TexMobject instead of MathTex
            label = math_tex(str(round(x_val_label, 4)))
            label.scale(self.x_label_font_size / 48)  # Approximate scaling
            label.set_color(self.axis_color)
            label.next_to(
                np.array([x_val, -self.x_tick_height, 0]),
                gx.DOWN,
                buff=0.1
            )
            self.labels.add(label)
            
        self.add(self.labels)
    
    # Helper method to get the axis line
    def get_axis_line(self):
        return self.axis_line
    
    # Helper method to get ticks
    def get_ticks(self):
        return self.ticks
    
    # Helper method to get labels
    def get_labels(self):
        return self.labels

    def map_to_canvas(self, value, axis_start=0):
        value_scaled=(value-self.x_min)/(self.x_max-self.x_min)
        return (value_scaled+axis_start)*self.axis_length_on_canvas

class WelchYAxis(gx.VGroup):
    def __init__(
        self,
        y_min=0,
        y_max=6, 
        y_ticks=[1, 2, 3, 4, 5],  # Default tick values
        y_tick_width=0.15,        # Default tick width
        y_label_font_size=24,     # Default font size
        stroke_width=3,           # Default stroke width
        color=CHILL_BROWN,        # Default color
        arrow_tip_scale=0.1,
        axis_length_on_canvas=5,
        include_tip=True,
        **kwargs
    ):
        gx.VGroup.__init__(self, **kwargs)
        
        # Store parameters
        self.y_ticks = y_ticks
        self.y_tick_width = y_tick_width
        self.y_label_font_size = y_label_font_size
        self.stroke_width = stroke_width
        self.axis_color = color
        self.arrow_tip_scale = arrow_tip_scale
        self.y_min = y_min
        self.y_max = y_max
        self.axis_length_on_canvas = axis_length_on_canvas
        self.include_tip=include_tip
        
        self.axis_to_canvas_scale = (self.y_max - self.y_min) / axis_length_on_canvas
        self.y_ticks_scaled = (np.array(y_ticks)-self.y_min)/ self.axis_to_canvas_scale
        
        # Create the basic components
        self._create_axis_line()
        self._create_ticks()
        self._create_labels()
    def _create_axis_line(self):
        # Create a line for the y-axis
        axis_line = gx.Line(
            start=np.array([0, 0, 0]),
            end=np.array([0, self.axis_length_on_canvas, 0]),
            color=self.axis_color,
            stroke_width=self.stroke_width
        )
        
        # Add SVG arrow tip at the end
        if self.include_tip:
            arrow_tip = gx.SVGMobject('/Users/stephen/manim/videos/_2025/sora/welch_arrow_tip_1.svg')
            arrow_tip.scale(self.arrow_tip_scale)
            arrow_tip.move_to([0, self.axis_length_on_canvas, 0])
            # Rotate the arrow tip to point upward
            arrow_tip.rotate(gx.PI/2)  # Rotate 90 degrees to point up
            axis_line = gx.VGroup(axis_line, arrow_tip)

        self.axis_line = axis_line
        self.add(axis_line)
        self.add(axis_line)
        
    def _create_ticks(self):
        self.ticks = gx.VGroup()
        
        for y_val in self.y_ticks_scaled:
            tick = gx.Line(
                start=np.array([0, y_val, 0]),
                end=np.array([-self.y_tick_width, y_val, 0]),  # Ticks extend to the left
                color=self.axis_color,
                stroke_width=self.stroke_width
            )
            self.ticks.add(tick)
            
        self.add(self.ticks)
        
    def _create_labels(self):
        self.labels = gx.VGroup()
        
        for y_val, y_val_label in zip(self.y_ticks_scaled, self.y_ticks):
            # Use Tex for labels
            label = math_tex(str(round(y_val_label,5)))
            label.scale(self.y_label_font_size / 48)  # Approximate scaling
            label.set_color(self.axis_color)
            label.next_to(
                np.array([-self.y_tick_width, y_val, 0]),
                gx.LEFT,
                buff=0.1
            )
            self.labels.add(label)
            
        self.add(self.labels)
    
    # Helper method to get the axis line
    def get_axis_line(self):
        return self.axis_line
    
    # Helper method to get ticks
    def get_ticks(self):
        return self.ticks
    
    # Helper method to get labels
    def get_labels(self):

        return self.labels
    def map_to_canvas(self, value, axis_start=0):
        value_scaled=(value-self.y_min)/(self.y_max-self.y_min)
        return (value_scaled+axis_start)*self.axis_length_on_canvas

class WelchArrow(gx.VGroup):
    pass