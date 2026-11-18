from welch_paths import welch_path
import manimgx as gx
from curve_fitting import eval_legendre_poly, f, fit_legendre_pinv, get_noisy_data
import numpy as np
from welch_geometry import diagram_axes
from welch_scene import WelchScene
from welch_text import math_tex

import os
import scipy.special

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' #6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
TEST_BLUE='#008080'  # Teal color for all test/blue elements

# graphics_dir = os.path.expanduser('~/Stephencwelch Dropbox/welch_labs/double_descent/graphics/')
# svg_dir = os.path.expanduser('~/Downloads')
graphics_dir=welch_path('double_descent/graphics/')
svg_dir=welch_path('double_descent/graphics/to_manim')


# def f(x): return 0.5*(x**4-3*x**2)
# def f(x): return np.add(2.0 * x, np.cos(x * 25)) #[:, 0]


class P8_15V2(WelchScene):
    def construct(self):
        # Configuration Parameters
        random_seed = 428
        n_points = 10
        noise_level = 0.2
        
        # Load SVG Axes
        curve_fit_axis_svg = gx.SVGMobject(svg_dir + '/p8_15_2a.svg')[1:]
        curve_fit_axis_svg.scale(4.0)
        curve_fit_axis_svg.move_to([-2.86, 0.6, 0])
        
        # Generate Training and Testing Data
        all_x = np.linspace(-2, 2, 128)
        all_y = f(all_x)
        
        n_train_points = int(np.floor(n_points * 0.5))
        n_test_points = n_points - n_train_points
        x, y = get_noisy_data(n_points, noise_level, random_seed)
        
        x_train, y_train = x[:n_train_points], y[:n_train_points]
        x_test, y_test = x[n_train_points:], y[n_train_points:]
        
        # Create First Axes (Curve Fitting Plot)
        axes_1 = diagram_axes(
            x_range=[-2.0, 2.0, 1],
            y_range=[-0.5, 2.0, 1],
            width=6,
            height=5,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": True,
                "include_numbers": True,
                "include_tip": True,
                "stroke_width": 3,
                "tip_config": {"width": 0.02, "length": 0.02}
            }
        )
        axes_1.move_to([-3, 0, 0])
        
        # Create Parabola and Data Points
        parabola = axes_1.plot(
            lambda x: f(x),
            x_range=[-2, 2],
            color=CHILL_BROWN
        )
        parabola.set_stroke(width=3)
        
        train_dots = gx.VGroup(*[
            gx.Dot(axes_1.c2p(x_train[i], y_train[i]), radius=0.08)
            for i in range(len(x_train))
        ])
        test_dots = gx.VGroup(*[
            gx.Dot(axes_1.c2p(x_test[i], y_test[i]), radius=0.08)
            for i in range(len(x_test))
        ])
        all_dots = gx.VGroup(test_dots, train_dots)
        all_dots.set_color(YELLOW)
        
        # Sort dots by x-coordinate for smooth animation
        dots_with_x = []
        for i, dot in enumerate(train_dots):
            dots_with_x.append((x_train[i], dot, 'train'))
        for i, dot in enumerate(test_dots):
            dots_with_x.append((x_test[i], dot, 'test'))
        dots_with_x.sort(key=lambda item: item[0])
        sorted_dots = [item[1] for item in dots_with_x]
        
        '''# Create Legend
        # Legend for Degree 1
        legend_train_dot_1 = Dot(radius=0.06).set_color(YELLOW)
        legend_train_text_1 = Text("Training Data", font_size=22, font='myraid-pro').set_color(CHILL_BROWN)
        # Arrange without aligned_edge so items are vertically centered
        legend_train_1 = VGroup(legend_train_dot_1, legend_train_text_1)
        
        legend_test_dot_1 = Dot(radius=0.06).set_color(TEST_BLUE)
        legend_test_text_1 = Text("Testing Data", font_size=22, font='myraid-pro').set_color(CHILL_BROWN)
        legend_test_1 = VGroup(legend_test_dot_1, legend_test_text_1)
        
        legend_line_1_1 = Line(LEFT * 0.15, RIGHT * 0.15, color=GREEN, stroke_width=3)
        legend_line_1_text_1 = Tex(r"y = ax+b", font_size=26).set_color(GREEN)
        # Adjust legend_line_1_item_1 to align equation to the right
        legend_line_1_item_1 = VGroup(legend_line_1_1, legend_line_1_text_1)

        legend_items_1 = VGroup(legend_train_1, legend_test_1, legend_line_1_item_1).arrange(RIGHT, buff=0.5)

        legend_box_1 = RoundedRectangle(
            width=legend_items_1.width + 0.8,
            height=legend_items_1.height + 0.4,
            corner_radius=0.08,
            stroke_color=CHILL_BROWN,
            stroke_width=2,
            fill_color=None,
            fill_opacity=0.0
        )
        legend_box_1.set_stroke(opacity=0.7)
        legend_items_1.move_to(legend_box_1.get_center())
        
        legend_1 = VGroup(legend_box_1, legend_items_1)
        legend_1.scale(0.9)
        legend_1.move_to(ORIGIN + DOWN * 3.2)

        # Legend for Degree 2
        legend_train_dot_2 = Dot(radius=0.06).set_color(YELLOW)
        legend_train_text_2 = Text("Training Data", font_size=22, font='myraid-pro').set_color(CHILL_BROWN)
        legend_train_2 = VGroup(legend_train_dot_2, legend_train_text_2).arrange(RIGHT, buff=0.2)
        
        legend_test_dot_2 = Dot(radius=0.06).set_color(TEST_BLUE)
        legend_test_text_2 = Text("Testing Data", font_size=22, font='myraid-pro').set_color(CHILL_BROWN)
        legend_test_2 = VGroup(legend_test_dot_2, legend_test_text_2).arrange(RIGHT, buff=0.2)
        
        legend_line_2_2 = Line(LEFT * 0.15, RIGHT * 0.15, color=YELLOW, stroke_width=3)
        legend_line_2_text_2 = Tex(r"y = ax^2+bx+c", font_size=26).set_color(YELLOW)
        # Adjust legend_line_2_item_2 to align equation to the right
        legend_line_2_item_2 = VGroup(legend_line_2_2, legend_line_2_text_2).arrange(RIGHT, buff=0.1, aligned_edge=RIGHT)

        legend_items_2 = VGroup(legend_train_2, legend_test_2, legend_line_1_item_1, legend_line_2_item_2).arrange(RIGHT, buff=0.5)

        legend_box_2 = RoundedRectangle(
            width=legend_items_2.width + 0.8,
            height=legend_items_2.height + 0.4,
            corner_radius=0.08,
            stroke_color=CHILL_BROWN,
            stroke_width=2,
            fill_color=None,
            fill_opacity=0.0
        )
        legend_box_2.set_stroke(opacity=0.7)
        legend_items_2.move_to(legend_box_2.get_center())
        
        legend_2 = VGroup(legend_box_2, legend_items_2)
        legend_2.scale(0.9)
        legend_2.move_to(ORIGIN + DOWN * 3.2)

        # Legend for Degree 3
        legend_train_dot_3 = Dot(radius=0.06).set_color(YELLOW)
        legend_train_text_3 = Text("Training Data", font_size=22, font='myraid-pro').set_color(CHILL_BROWN)
        legend_train_3 = VGroup(legend_train_dot_3, legend_train_text_3).arrange(RIGHT, buff=0.2)
        
        legend_test_dot_3 = Dot(radius=0.06).set_color(TEST_BLUE)
        legend_test_text_3 = Text("Testing Data", font_size=22, font='myraid-pro').set_color(CHILL_BROWN)
        legend_test_3 = VGroup(legend_test_dot_3, legend_test_text_3).arrange(RIGHT, buff=0.2)
        
        legend_line_3_3 = Line(LEFT * 0.15, RIGHT * 0.15, color=ORANGE, stroke_width=3)
        legend_line_3_text_3 = Tex(r"y = ax^3+bx^2+cx+d", font_size=26).set_color(ORANGE)
        # Adjust legend_line_3_item_3 to align equation to the right
        legend_line_3_item_3 = VGroup(legend_line_3_3, legend_line_3_text_3).arrange(RIGHT, buff=0.1, aligned_edge=RIGHT)

        legend_items_3 = VGroup(legend_train_3, legend_test_3, legend_line_1_item_1, legend_line_2_item_2, legend_line_3_item_3).arrange(RIGHT, buff=0.5)

        legend_box_3 = RoundedRectangle(
            width=legend_items_3.width + 0.8,
            height=legend_items_3.height + 0.4,
            corner_radius=0.08,
            stroke_color=CHILL_BROWN,
            stroke_width=2,
            fill_color=None,
            fill_opacity=0.0
        )
        legend_box_3.set_stroke(opacity=0.7)
        legend_items_3.move_to(legend_box_3.get_center())
        
        legend_3 = VGroup(legend_box_3, legend_items_3)
        legend_3.scale(0.9)
        legend_3.move_to(ORIGIN + DOWN * 3.2)

        # Legend for Degree 4
        legend_train_dot_4 = Dot(radius=0.06).set_color(YELLOW)
        legend_train_text_4 = Text("Training Data", font_size=22, font='myraid-pro').set_color(CHILL_BROWN)
        legend_train_4 = VGroup(legend_train_dot_4, legend_train_text_4).arrange(RIGHT, buff=0.2)
        
        legend_test_dot_4 = Dot(radius=0.06).set_color(TEST_BLUE)
        legend_test_text_4 = Text("Testing Data", font_size=22, font='myraid-pro').set_color(CHILL_BROWN)
        legend_test_4 = VGroup(legend_test_dot_4, legend_test_text_4).arrange(RIGHT, buff=0.2)
        
        legend_line_4_4 = Line(LEFT * 0.15, RIGHT * 0.15, color=MAROON_B, stroke_width=3)
        legend_line_4_text_4 = Tex(r"y = ax^4+bx^3+cx^2+dx+e", font_size=26).set_color(MAROON_B)
        # Adjust legend_line_4_item_4 to align equation to the right
        legend_line_4_item_4 = VGroup(legend_line_4_4, legend_line_4_text_4).arrange(RIGHT, buff=0.1, aligned_edge=RIGHT)

        legend_items_4 = VGroup(legend_train_4, legend_test_4, legend_line_1_item_1, legend_line_2_item_2, legend_line_3_item_3, legend_line_4_item_4).arrange(RIGHT, buff=0.5)

        legend_box_4 = RoundedRectangle(
            width=legend_items_4.width + 0.8,
            height=legend_items_4.height + 0.4,
            corner_radius=0.08,
            stroke_color=CHILL_BROWN,
            stroke_width=2,
            fill_color=None,
            fill_opacity=0.0
        )
        legend_box_4.set_stroke(opacity=0.7)
        legend_items_4.move_to(legend_box_4.get_center())
        
        legend_4 = VGroup(legend_box_4, legend_items_4)
        legend_4.scale(0.9)
        legend_4.move_to(ORIGIN + DOWN * 3.2)'''
        
        # Create simplified legend with just Training Data, Testing Data, and Target Function
        legend = gx.VGroup()
        
        # Training Data item
        legend_training_dot = gx.Dot(radius=0.06).set_color(YELLOW)
        legend_training_text = gx.Text("Training Data", font_size=20, font='myraid-pro').set_color(CHILL_BROWN)
        legend_training = gx.VGroup(legend_training_dot, legend_training_text).arrange(gx.RIGHT, buff=0.15, aligned_edge=gx.ORIGIN)
        
        # Testing Data item
        legend_testing_dot = gx.Dot(radius=0.06).set_color(TEST_BLUE)
        legend_testing_text = gx.Text("Testing Data", font_size=20, font='myraid-pro').set_color(CHILL_BROWN)
        legend_testing = gx.VGroup(legend_testing_dot, legend_testing_text).arrange(gx.RIGHT, buff=0.15, aligned_edge=gx.ORIGIN)
        
        # Target Function item
        legend_target_line = gx.Line(gx.LEFT * 0.2, gx.RIGHT * 0.2, color=CHILL_BROWN, stroke_width=3)
        legend_target_text = gx.Text("Target Function", font_size=20, font='myraid-pro').set_color(CHILL_BROWN)
        legend_target = gx.VGroup(legend_target_line, legend_target_text).arrange(gx.RIGHT, buff=0.15, aligned_edge=gx.ORIGIN)
        
        # Arrange all items in the legend
        legend_items = gx.VGroup(legend_training, legend_testing, legend_target).arrange(gx.RIGHT, buff=0.4)
        
        # Create box around legend
        legend_box = gx.RoundedRectangle(
            width=legend_items.width + 0.5,
            height=legend_items.height + 0.35,
            corner_radius=0.08,
            stroke_color=CHILL_BROWN,
            stroke_width=2,
            fill_color=None,
            fill_opacity=0.0
        )
        legend_box.set_stroke(opacity=0.7)
        legend_items.move_to(legend_box.get_center())
        
        legend.add(legend_box, legend_items)
        legend.scale(0.85)
        # Position under the first plot - x-aligned with axes_1 at x=-3
        legend.move_to([-3, -3.2, 0])
        


        # Initial Animation: Show Curve and Data
        self.set_view(0, 0, 0, (-2.94, 0.08, 0.0), 7.45)
        
        self.play(gx.Write(curve_fit_axis_svg), run_time=3)
        self.play(gx.Create(parabola), run_time=2)  # Show the target function (brown parabola)
        self.play(
            gx.LaggedStart(*[gx.FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15),
            run_time=2
        )
        self.wait()
        
        self.play(
            test_dots.animate.set_color(TEST_BLUE),
            gx.FadeIn(legend)
        )
        self.wait()
        
        # Fit Linear Model (Degree 1)
        # Degree 1 fit
        degree = 1
        beta_hat = fit_legendre_pinv(x_train, y_train, degree)
        all_y_fit = eval_legendre_poly(beta_hat, all_x, degree)
        y_train_pred = eval_legendre_poly(beta_hat, x_train, degree)
        y_test_pred = eval_legendre_poly(beta_hat, x_test, degree)
        train_error_1 = np.mean((y_train - y_train_pred)**2)
        test_error_1 = np.mean((y_test - y_test_pred)**2)

        # Degree 2 fit
        degree2 = 2
        beta_hat2 = fit_legendre_pinv(x_train, y_train, degree2)
        all_y_fit2 = eval_legendre_poly(beta_hat2, all_x, degree2)
        y_train_pred2 = eval_legendre_poly(beta_hat2, x_train, degree2)
        y_test_pred2 = eval_legendre_poly(beta_hat2, x_test, degree2)
        train_error_2 = np.mean((y_train - y_train_pred2)**2)
        test_error_2 = np.mean((y_test - y_test_pred2)**2)
        
        # Degree 3 fit
        degree3 = 3
        beta_hat3 = fit_legendre_pinv(x_train, y_train, degree3)
        all_y_fit3 = eval_legendre_poly(beta_hat3, all_x, degree3)
        y_train_pred3 = eval_legendre_poly(beta_hat3, x_train, degree3)
        y_test_pred3 = eval_legendre_poly(beta_hat3, x_test, degree3)
        train_error_3 = np.mean((y_train - y_train_pred3)**2)
        test_error_3 = np.mean((y_test - y_test_pred3)**2)
        
        # Degree 4 fit
        degree4 = 4
        beta_hat4 = fit_legendre_pinv(x_train, y_train, degree4)
        all_y_fit4 = eval_legendre_poly(beta_hat4, all_x, degree4)
        y_train_pred4 = eval_legendre_poly(beta_hat4, x_train, degree4)
        y_test_pred4 = eval_legendre_poly(beta_hat4, x_test, degree4)
        train_error_4 = np.mean((y_train - y_train_pred4)**2)
        test_error_4 = np.mean((y_test - y_test_pred4)**2)
        
        fit_points = [axes_1.c2p(all_x[i], all_y_fit[i]) for i in range(len(all_x))]
        fit_line_1 = gx.VMobject(color=GREEN, stroke_width=3)
        fit_line_1.set_points_smoothly(fit_points)
        
        # Create equation for degree 1 - positioned right above the green line
        eq_1 = math_tex("y = ax + b", font_size=40).set_color(GREEN)
        # Position above the green line at x=0
        line_y_at_center = axes_1.c2p(0, eval_legendre_poly(beta_hat, np.array([0]), degree)[0])[1]
        eq_1.move_to([-3, line_y_at_center + 0.6, 0])  # Position above the green line
        
        self.play(
            gx.Create(fit_line_1),
            gx.Write(eq_1),
            run_time=1.5
        )
        self.bring_to_front(fit_line_1)  # Keep fit line in front
        self.wait()
        
        # Create Second Axes (Error Plot)
        axes_2 = diagram_axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.2, 1],
            width=6,
            height=5,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": True,
                "include_numbers": True,
                "include_tip": True,
                "stroke_width": 3,
                "tip_config": {"width": 0.02, "length": 0.02}
            }
        )
        axes_2.move_to([4.0, 0.48, 0])
        
        # Error data for degrees 1-4
        degrees = [1, 2, 3, 4]
        train_errors = [0.221350, 0.031825, 0.000803, 0.000000]
        test_errors = [0.691636, 0.079449, 0.727023, 1.328535]
        
        train_error_dots = gx.VGroup(*[
            gx.Dot(axes_2.c2p(degrees[i], train_errors[i]), radius=0.08)
            for i in range(len(degrees))
        ])
        test_error_dots = gx.VGroup(*[
            gx.Dot(axes_2.c2p(degrees[i], test_errors[i]), radius=0.08)
            for i in range(len(degrees))
        ])
        train_error_dots.set_color(YELLOW)
        test_error_dots.set_color(TEST_BLUE)
        
        error_axis_svg = gx.SVGMobject(svg_dir + '/p8_15_2-05.svg')
        error_axis_svg.scale(2.95)
        error_axis_svg.move_to([4.79, 0.75, 0])
        
        # Create Training Error Bars
        train_error_bars = gx.VGroup()
        for i in range(len(x_train)):
            point_pos = axes_1.c2p(x_train[i], y_train[i])
            fit_pos = axes_1.c2p(x_train[i], y_train_pred[i])
            # Always draw from the lower point to the higher point
            if point_pos[1] > fit_pos[1]:  # point is above fit
                error_bar = gx.Line(fit_pos, point_pos, color=YELLOW, stroke_width=3)
            else:  # point is below fit
                error_bar = gx.Line(point_pos, fit_pos, color=YELLOW, stroke_width=3)
            train_error_bars.add(error_bar)
        
        # Create target bars for error plot (stacked bars)
        target_train_bars = gx.VGroup()
        bar_height = train_errors[0] / len(train_error_bars)
        x_pos = degrees[0]
        
        for i in range(len(train_error_bars)):
            bottom_y = i * bar_height
            top_y = (i + 1) * bar_height
            bottom = axes_2.c2p(x_pos, bottom_y)
            top = axes_2.c2p(x_pos, top_y)
            target_bar = gx.Line(bottom, top, color=YELLOW, stroke_width=3)
            target_train_bars.add(target_bar)
        
        # Create Testing Error Bars
        test_error_bars = gx.VGroup()
        for i in range(len(x_test)):
            point_pos = axes_1.c2p(x_test[i], y_test[i])
            fit_pos = axes_1.c2p(x_test[i], y_test_pred[i])
            # Always draw from the lower point to the higher point
            if point_pos[1] > fit_pos[1]:  # point is above fit
                error_bar = gx.Line(fit_pos, point_pos, color=TEST_BLUE, stroke_width=3)
            else:  # point is below fit
                error_bar = gx.Line(point_pos, fit_pos, color=TEST_BLUE, stroke_width=3)
            test_error_bars.add(error_bar)
        
        # Create target bars for error plot (stacked bars)
        target_test_bars = gx.VGroup()
        bar_height = test_errors[0] / len(test_error_bars)
        x_pos = degrees[0]
        
        for i in range(len(test_error_bars)):
            bottom_y = i * bar_height
            top_y = (i + 1) * bar_height
            bottom = axes_2.c2p(x_pos, bottom_y)
            top = axes_2.c2p(x_pos, top_y)
            target_bar = gx.Line(bottom, top, color=TEST_BLUE, stroke_width=3)
            target_test_bars.add(target_bar)
        
        # Animate Training Error Bars
        self.wait()
        self.play(
            gx.LaggedStart(*[gx.Create(bar) for bar in train_error_bars], lag_ratio=0.1),
            run_time=1.5
        )
        self.bring_to_front(fit_line_1)  # Keep fit line in front of error bars
        
        train_error_bars_copy = train_error_bars.copy()
        self.add(train_error_bars_copy)
        self.bring_to_front(fit_line_1)  # Bring fit line to front after adding copy
        train_error_bars.set_opacity(0.5)
        self.wait()
        
        self.play(
            self.animate_view(0, 0, 0, (0.82, 0.46, 0.0), 8.86),
            # Legend stays at x=-3 aligned with axes_1, no shift needed
            gx.Write(error_axis_svg),
            gx.ReplacementTransform(train_error_bars, target_train_bars),
            run_time=4
        )
        self.play(gx.Create(train_error_dots[0]))
        self.wait()
        
        # Load Error Curves SVG
        error_curves_svg = gx.SVGMobject(svg_dir + '/p8_15_2-06.svg')
        error_curves_svg.scale(3.15)
        error_curves_svg.move_to([4.84, 0.74, 0])
        error_curves_svg.shift(gx.UP*0.51)
        error_curves_svg[0].set_color(TEST_BLUE)
        
        # Animate Testing Error Bars
        self.wait()
        
        self.play(
            gx.LaggedStart(*[gx.Create(bar) for bar in test_error_bars], lag_ratio=0.1),
            run_time=1.5
        )
        self.bring_to_front(fit_line_1)  # Keep fit line in front of error bars
        
        # Copy test error bars before moving
        test_error_bars_copy = test_error_bars.copy()
        self.add(test_error_bars_copy)
        self.bring_to_front(fit_line_1)  # Bring fit line to front after adding copy
        
        # Fade out yellow bars first, then move blue bars
        self.play(target_train_bars.animate.set_opacity(0.0), run_time=1.5)
        self.bring_to_front(train_error_dots[0])
        self.bring_to_front(fit_line_1)  # Keep fit line in front
        self.play(gx.ReplacementTransform(test_error_bars_copy, target_test_bars), run_time=3.0)
        self.bring_to_front(train_error_dots[0])
        self.play(gx.Create(test_error_dots[0]))
        
        self.wait()
        
        # ===== DEGREE 2 FIT ANIMATION =====
        
        # Remove all degree 1 error bars from the left plot
        self.play(
            gx.FadeOut(train_error_bars_copy),
            gx.FadeOut(test_error_bars),
            run_time=1.0
        )
        
        # Create degree 2 fit line
        fit_points_2 = [axes_1.c2p(all_x[i], all_y_fit2[i]) for i in range(len(all_x))]
        fit_line_2 = gx.VMobject(color=YELLOW, stroke_width=3)
        fit_line_2.set_points_smoothly(fit_points_2)
        
        # Create equation for degree 2 - centered above left graph
        eq_2 = math_tex("y = ax^2 + bx + c", font_size=40).set_color(YELLOW)
        eq_2.move_to([-3, 3.2, 0])  # Centered at x=-3 (axes_1 position), above the graph
        
        self.play(
            gx.Create(fit_line_2),
            fit_line_1.animate.set_stroke(opacity=0.3),
            gx.FadeOut(eq_1),
            run_time=2
        )
        self.play(gx.Write(eq_2))
        self.bring_to_front(fit_line_1)  # Keep all fit lines in front
        self.bring_to_front(fit_line_2)
        self.wait()
        
        # Create Training Error Bars for Degree 2
        train_error_bars_2 = gx.VGroup()
        for i in range(len(x_train)):
            point_pos = axes_1.c2p(x_train[i], y_train[i])
            fit_pos_2 = axes_1.c2p(x_train[i], y_train_pred2[i])
            if point_pos[1] > fit_pos_2[1]:
                error_bar = gx.Line(fit_pos_2, point_pos, color=YELLOW, stroke_width=3)
            else:
                error_bar = gx.Line(point_pos, fit_pos_2, color=YELLOW, stroke_width=3)
            train_error_bars_2.add(error_bar)
        
        # Create target bars for degree 2 on error plot
        target_train_bars_2 = gx.VGroup()
        bar_height_2 = train_errors[1] / len(train_error_bars_2)
        x_pos_2 = degrees[1]
        
        for i in range(len(train_error_bars_2)):
            bottom_y = i * bar_height_2
            top_y = (i + 1) * bar_height_2
            bottom = axes_2.c2p(x_pos_2, bottom_y)
            top = axes_2.c2p(x_pos_2, top_y)
            target_bar = gx.Line(bottom, top, color=YELLOW, stroke_width=3)
            target_train_bars_2.add(target_bar)
        
        # Animate training error bars for degree 2
        self.play(
            gx.LaggedStart(*[gx.Create(bar) for bar in train_error_bars_2], lag_ratio=0.1),
            run_time=1.5
        )
        self.bring_to_front(fit_line_1)  # Keep all fit lines in front of error bars
        self.bring_to_front(fit_line_2)
        
        train_error_bars_2_copy = train_error_bars_2.copy()
        self.add(train_error_bars_2_copy)
        self.bring_to_front(fit_line_1)  # Bring fit lines to front after adding copy
        self.bring_to_front(fit_line_2)
        train_error_bars_2.set_opacity(0.5)
        self.wait()
        
        # Fade out degree 1 test bars and move degree 2 training bars
        self.play(target_test_bars.animate.set_opacity(0.0), run_time=1.5)
        self.bring_to_front(fit_line_1)  # Keep fit lines in front
        self.bring_to_front(fit_line_2)
        self.play(gx.ReplacementTransform(train_error_bars_2, target_train_bars_2), run_time=3.0)
        self.bring_to_front(fit_line_1)  # Keep fit lines in front after transform
        self.bring_to_front(fit_line_2)
        self.play(gx.Create(train_error_dots[1]))
        self.wait()
        
        # Create Testing Error Bars for Degree 2
        test_error_bars_2 = gx.VGroup()
        for i in range(len(x_test)):
            point_pos = axes_1.c2p(x_test[i], y_test[i])
            fit_pos_2 = axes_1.c2p(x_test[i], y_test_pred2[i])
            if point_pos[1] > fit_pos_2[1]:
                error_bar = gx.Line(fit_pos_2, point_pos, color=TEST_BLUE, stroke_width=3)
            else:
                error_bar = gx.Line(point_pos, fit_pos_2, color=TEST_BLUE, stroke_width=3)
            test_error_bars_2.add(error_bar)
        
        # Create target test bars for degree 2
        target_test_bars_2 = gx.VGroup()
        bar_height_test_2 = test_errors[1] / len(test_error_bars_2)
        x_pos_test_2 = degrees[1]
        
        for i in range(len(test_error_bars_2)):
            bottom_y = i * bar_height_test_2
            top_y = (i + 1) * bar_height_test_2
            bottom = axes_2.c2p(x_pos_test_2, bottom_y)
            top = axes_2.c2p(x_pos_test_2, top_y)
            target_bar = gx.Line(bottom, top, color=TEST_BLUE, stroke_width=3)
            target_test_bars_2.add(target_bar)
        
        # Animate testing error bars for degree 2
        self.play(
            gx.LaggedStart(*[gx.Create(bar) for bar in test_error_bars_2], lag_ratio=0.1),
            run_time=1.5
        )
        self.bring_to_front(fit_line_1)  # Keep all fit lines in front of error bars
        self.bring_to_front(fit_line_2)
        
        test_error_bars_2_copy = test_error_bars_2.copy()
        self.add(test_error_bars_2_copy)
        self.bring_to_front(fit_line_1)  # Bring fit lines to front after adding copy
        self.bring_to_front(fit_line_2)
        
        # Fade out degree 2 training bars and move degree 2 test bars
        self.play(target_train_bars_2.animate.set_opacity(0.0), run_time=1.5)
        self.bring_to_front(train_error_dots[1])
        self.bring_to_front(fit_line_1)  # Keep fit lines on top
        self.bring_to_front(fit_line_2)
        self.play(gx.ReplacementTransform(test_error_bars_2_copy, target_test_bars_2), run_time=3.0)
        self.bring_to_front(train_error_dots[1])
        self.bring_to_front(fit_line_1)  # Keep fit lines on top after transform
        self.bring_to_front(fit_line_2)
        self.play(gx.Create(test_error_dots[1]))
        
        self.wait()
        
        # ===== DEGREE 3 FIT ANIMATION =====
        
        # Remove all degree 2 error bars from the left plot
        self.play(
            gx.FadeOut(train_error_bars_2_copy),
            gx.FadeOut(test_error_bars_2),
            run_time=1.0
        )
        
        # Create degree 3 fit line
        fit_points_3 = [axes_1.c2p(all_x[i], all_y_fit3[i]) for i in range(len(all_x))]
        fit_line_3 = gx.VMobject(color=gx.ORANGE, stroke_width=3)
        fit_line_3.set_points_smoothly(fit_points_3)
        
        # Create equation for degree 3 - centered above left graph
        eq_3 = math_tex("y = ax^3 + bx^2 + cx + d", font_size=40).set_color(gx.ORANGE)
        eq_3.move_to([-3, 3.2, 0])  # Centered at x=-3 (axes_1 position), above the graph
        
        self.play(
            gx.Create(fit_line_3),
            fit_line_2.animate.set_stroke(opacity=0.3),
            gx.FadeOut(eq_2),
            run_time=2
        )
        self.play(gx.Write(eq_3))
        self.bring_to_front(fit_line_1)  # Keep all fit lines in front
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        self.wait()
        
        # Create Training Error Bars for Degree 3
        train_error_bars_3 = gx.VGroup()
        for i in range(len(x_train)):
            point_pos = axes_1.c2p(x_train[i], y_train[i])
            fit_pos_3 = axes_1.c2p(x_train[i], y_train_pred3[i])
            if point_pos[1] > fit_pos_3[1]:
                error_bar = gx.Line(fit_pos_3, point_pos, color=YELLOW, stroke_width=3)
            else:
                error_bar = gx.Line(point_pos, fit_pos_3, color=YELLOW, stroke_width=3)
            train_error_bars_3.add(error_bar)
        
        # Create target bars for degree 3 on error plot
        target_train_bars_3 = gx.VGroup()
        bar_height_3 = train_errors[2] / len(train_error_bars_3)
        x_pos_3 = degrees[2]
        
        for i in range(len(train_error_bars_3)):
            bottom_y = i * bar_height_3
            top_y = (i + 1) * bar_height_3
            bottom = axes_2.c2p(x_pos_3, bottom_y)
            top = axes_2.c2p(x_pos_3, top_y)
            target_bar = gx.Line(bottom, top, color=YELLOW, stroke_width=3)
            target_train_bars_3.add(target_bar)
        
        # Animate training error bars for degree 3
        self.play(
            gx.LaggedStart(*[gx.Create(bar) for bar in train_error_bars_3], lag_ratio=0.1),
            run_time=1.5
        )
        self.bring_to_front(fit_line_1)  # Keep all fit lines in front of error bars
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        
        train_error_bars_3_copy = train_error_bars_3.copy()
        self.add(train_error_bars_3_copy)
        self.bring_to_front(fit_line_1)  # Bring fit lines to front after adding copy
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        train_error_bars_3.set_opacity(0.5)
        self.wait()
        
        # Fade out degree 2 test bars and move degree 3 training bars
        self.play(target_test_bars_2.animate.set_opacity(0.0), run_time=1.5)
        self.bring_to_front(fit_line_1)  # Keep fit lines in front
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        self.play(gx.ReplacementTransform(train_error_bars_3, target_train_bars_3), run_time=3.0)
        self.bring_to_front(fit_line_1)  # Keep fit lines in front after transform
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        self.play(gx.Create(train_error_dots[2]))
        self.wait()
        
        # Create Testing Error Bars for Degree 3
        test_error_bars_3 = gx.VGroup()
        for i in range(len(x_test)):
            point_pos = axes_1.c2p(x_test[i], y_test[i])
            fit_pos_3 = axes_1.c2p(x_test[i], y_test_pred3[i])
            if point_pos[1] > fit_pos_3[1]:
                error_bar = gx.Line(fit_pos_3, point_pos, color=TEST_BLUE, stroke_width=3)
            else:
                error_bar = gx.Line(point_pos, fit_pos_3, color=TEST_BLUE, stroke_width=3)
            test_error_bars_3.add(error_bar)
        
        # Create target test bars for degree 3
        target_test_bars_3 = gx.VGroup()
        bar_height_test_3 = test_errors[2] / len(test_error_bars_3)
        x_pos_test_3 = degrees[2]
        
        for i in range(len(test_error_bars_3)):
            bottom_y = i * bar_height_test_3
            top_y = (i + 1) * bar_height_test_3
            bottom = axes_2.c2p(x_pos_test_3, bottom_y)
            top = axes_2.c2p(x_pos_test_3, top_y)
            target_bar = gx.Line(bottom, top, color=TEST_BLUE, stroke_width=3)
            target_test_bars_3.add(target_bar)
        
        # Animate testing error bars for degree 3
        self.play(
            gx.LaggedStart(*[gx.Create(bar) for bar in test_error_bars_3], lag_ratio=0.1),
            run_time=1.5
        )
        self.bring_to_front(fit_line_1)  # Keep all fit lines in front of error bars
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        
        test_error_bars_3_copy = test_error_bars_3.copy()
        self.add(test_error_bars_3_copy)
        self.bring_to_front(fit_line_1)  # Bring fit lines to front after adding copy
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        
        # Fade out degree 3 training bars and move degree 3 test bars
        self.play(target_train_bars_3.animate.set_opacity(0.0), run_time=1.5)
        self.bring_to_front(train_error_dots[2])
        self.bring_to_front(fit_line_1)  # Keep fit lines in front
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        self.play(gx.ReplacementTransform(test_error_bars_3_copy, target_test_bars_3), run_time=3.0)
        self.bring_to_front(train_error_dots[2])
        self.play(gx.Create(test_error_dots[2]))
        
        self.wait()
        
        # ===== DEGREE 4 FIT ANIMATION =====
        
        # Remove all degree 3 error bars from the left plot
        self.play(
            gx.FadeOut(train_error_bars_3_copy),
            gx.FadeOut(test_error_bars_3),
            run_time=1.0
        )
        
        # Create degree 4 fit line
        fit_points_4 = [axes_1.c2p(all_x[i], all_y_fit4[i]) for i in range(len(all_x))]
        fit_line_4 = gx.VMobject(color=gx.MAROON_B, stroke_width=3)
        fit_line_4.set_points_smoothly(fit_points_4)
        
        # Create equation for degree 4 - centered above left graph
        eq_4 = math_tex("y = ax^4 + bx^3 + cx^2 + dx + e", font_size=40).set_color(gx.MAROON_B)
        eq_4.move_to([-3, 3.2, 0])  # Centered at x=-3 (axes_1 position), above the graph
        
        self.play(
            gx.Create(fit_line_4),
            fit_line_3.animate.set_stroke(opacity=0.3),
            gx.FadeOut(eq_3),
            run_time=2
        )
        self.play(gx.Write(eq_4))
        self.bring_to_front(fit_line_1)  # Keep all fit lines in front
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        self.bring_to_front(fit_line_4)
        self.wait()
        
        # Skip training error bars for degree 4 since training error is essentially zero
        # Just fade out degree 3 test bars and show training error dot at zero
        self.play(target_test_bars_3.animate.set_opacity(0.0), run_time=1.5)
        self.play(gx.Create(train_error_dots[3]))  # Draw yellow dot at zero before blue bars
        self.wait()
        
        # Create Testing Error Bars for Degree 4
        test_error_bars_4 = gx.VGroup()
        for i in range(len(x_test)):
            point_pos = axes_1.c2p(x_test[i], y_test[i])
            fit_pos_4 = axes_1.c2p(x_test[i], y_test_pred4[i])
            if point_pos[1] > fit_pos_4[1]:
                error_bar = gx.Line(fit_pos_4, point_pos, color=TEST_BLUE, stroke_width=3)
            else:
                error_bar = gx.Line(point_pos, fit_pos_4, color=TEST_BLUE, stroke_width=3)
            test_error_bars_4.add(error_bar)
        
        # Create target test bars for degree 4
        target_test_bars_4 = gx.VGroup()
        bar_height_test_4 = test_errors[3] / len(test_error_bars_4)
        x_pos_test_4 = degrees[3]
        
        for i in range(len(test_error_bars_4)):
            bottom_y = i * bar_height_test_4
            top_y = (i + 1) * bar_height_test_4
            bottom = axes_2.c2p(x_pos_test_4, bottom_y)
            top = axes_2.c2p(x_pos_test_4, top_y)
            target_bar = gx.Line(bottom, top, color=TEST_BLUE, stroke_width=3)
            target_test_bars_4.add(target_bar)
        
        # Animate testing error bars for degree 4
        self.play(
            gx.LaggedStart(*[gx.Create(bar) for bar in test_error_bars_4], lag_ratio=0.1),
            run_time=1.5
        )
        self.bring_to_front(fit_line_1)  # Keep all fit lines in front of error bars
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        self.bring_to_front(fit_line_4)
        
        test_error_bars_4_copy = test_error_bars_4.copy()
        self.add(test_error_bars_4_copy)
        self.bring_to_front(fit_line_1)  # Bring fit lines to front after adding copy
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        self.bring_to_front(fit_line_4)
        
        # Move degree 4 test bars (training dot already shown at zero)
        self.bring_to_front(train_error_dots[3])  # Keep yellow dot visible
        self.bring_to_front(fit_line_1)  # Keep fit lines in front
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        self.bring_to_front(fit_line_4)
        self.play(gx.ReplacementTransform(test_error_bars_4_copy, target_test_bars_4), run_time=3.0)
        self.bring_to_front(train_error_dots[3])  # Keep yellow dot in front
        self.bring_to_front(fit_line_1)  # Keep fit lines in front after transform
        self.bring_to_front(fit_line_2)
        self.bring_to_front(fit_line_3)
        self.bring_to_front(fit_line_4)
        self.play(gx.Create(test_error_dots[3]))
        
        self.wait()
        
        # Fade out all error bars on the left plot at the end
        self.play(
            gx.FadeOut(test_error_bars_4),
            gx.FadeOut(target_test_bars_4),
            run_time=1.0
        )
        
        # Fade all fit lines on the left to match the right side
        self.play(
            fit_line_1.animate.set_stroke(opacity=0.15),
            fit_line_2.animate.set_stroke(opacity=0.15),
            fit_line_3.animate.set_stroke(opacity=0.15),
            fit_line_4.animate.set_stroke(opacity=0.15),
            gx.FadeOut(eq_4),  # Fade out the last equation
            run_time=1.5
        )
        
        self.wait()
        
        self.play(gx.FadeIn(error_curves_svg[0]))
        self.play(gx.FadeIn(error_curves_svg[1]))
        self.wait()

        # self.remove(legend, parabola, fit_line_1, fit_line_2, fit_line_3, fit_line_4)
        # self.remove(curve_fit_axis_svg, train_dots, test_dots, train_error_dots, test_error_dots, error_curves_svg[1])
        
        self.wait()
        self.play(gx.FadeOut(legend),
                  gx.FadeOut(parabola),
                  gx.FadeOut(fit_line_1),
                  gx.FadeOut(fit_line_2),
                  gx.FadeOut(fit_line_3),
                  gx.FadeOut(fit_line_4),
                  gx.FadeOut(curve_fit_axis_svg),
                  gx.FadeOut(train_dots),
                  gx.FadeOut(test_dots),
                  gx.FadeOut(train_error_dots),
                  gx.FadeOut(test_error_dots),
                  gx.FadeOut(error_curves_svg[1]),
                  self.animate_view(0, 0, 0, (4.53, 1.33, 0.0), 9.13),
                  run_time=4)
        self.wait()









        self.wait(20)
        # Scene authoring stops here in the interactive editor.

        # Final Wait























