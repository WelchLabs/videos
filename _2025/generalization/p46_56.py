from welch_paths import welch_path
import manimgx as gx
from curve_fitting import eval_legendre_poly, f, fit_legendre_pinv, get_fit_line, get_noisy_data
import numpy as np
from welch_geometry import diagram_axes
from welch_scene import WelchScene
from welch_text import math_tex

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
TEST_BLUE='#008080' 

graphics_dir=welch_path('double_descent/graphics/')
svg_dir=welch_path('double_descent/graphics/to_manim')

# def f(x): return 0.5*(x**4-3*x**2)
# def f(x): return np.add(2.0 * x, np.cos(x * 25)) #[:, 0]


class p46_56(WelchScene):
    '''
    Ok long scene here, let's start chipping away. 
    '''
    def construct(self):

        curve_fit_axis_svg=gx.SVGMobject(svg_dir+'/p8_15_2a.svg')[1:]
        curve_fit_axis_svg.scale(4.0)
        curve_fit_axis_svg.move_to([-2.86, 0.6, 0])

        random_seed=428
        n_points=10
        noise_level=0.2


        all_x = np.linspace(-2, 2, 128)
        all_y = f(all_x)

        n_train_points=int(np.floor(n_points*0.5))
        n_test_points=n_points-n_train_points
        x,y=get_noisy_data(n_points, noise_level, random_seed)
                           
        x_train, y_train=x[:n_train_points], y[:n_train_points]
        x_test, y_test=x[n_train_points:],y[n_train_points:]

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
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
        )
        axes_1.move_to([-3, 0, 0])

        parabola = axes_1.plot(
            lambda x: f(x),
            x_range=[-2, 2],
            color=CHILL_BROWN
        )
        parabola.set_stroke(width=3)


        train_dots = gx.VGroup(*[gx.Dot(axes_1.c2p(x_train[i], y_train[i]), radius=0.08) for i in range(len(x_train))])
        test_dots = gx.VGroup(*[ gx.Dot(axes_1.c2p(x_test[i], y_test[i]), radius=0.08) for i in range(len(x_test))])
        all_dots=gx.VGroup(test_dots, train_dots)
        all_dots.set_color(YELLOW)

        #Sorted dots so I can bring them in nicely. 
        dots_with_x = []
        for i, dot in enumerate(train_dots):
            dots_with_x.append((x_train[i], dot, 'train'))
        for i, dot in enumerate(test_dots):
            dots_with_x.append((x_test[i], dot, 'test'))
        dots_with_x.sort(key=lambda item: item[0])
        sorted_dots = [item[1] for item in dots_with_x]


        # Create legend items
        legend_train_dot = gx.Dot(radius=0.05).set_color(YELLOW)
        legend_train_text = gx.Text("Training Data", font_size=18, font='myraid-pro').set_color(CHILL_BROWN)
        legend_train = gx.VGroup(legend_train_dot, legend_train_text).arrange(gx.RIGHT, buff=0.15)
        
        legend_test_dot = gx.Dot(radius=0.05).set_opacity(0.5).set_color(BLUE)
        legend_test_text = gx.Text("Testing Data", font_size=18, font='myraid-pro').set_color(CHILL_BROWN)
        legend_test = gx.VGroup(legend_test_dot, legend_test_text).arrange(gx.RIGHT, buff=0.15)

        legend_line = gx.Line(gx.LEFT * 0.2, gx.RIGHT * 0.2, color=CHILL_BROWN, stroke_width=3)
        legend_line_text = gx.Text("Target Function", font_size=18, font='myraid-pro').set_color(CHILL_BROWN)
        legend_line_item = gx.VGroup(legend_line, legend_line_text).arrange(gx.RIGHT, buff=0.15)
        
        # Arrange legend items horizontally
        legend_items = gx.VGroup(legend_train, legend_test, legend_line_item).arrange(gx.RIGHT, buff=0.3)
        
        # Create rounded rectangle background
        legend_box = gx.RoundedRectangle(
            width=legend_items.width + 0.6,
            height=legend_items.height + 0.3,
            corner_radius=0.08,
            stroke_color=CHILL_BROWN,
            stroke_width=2,
            fill_color=None,
            fill_opacity=0.0
        )
        legend_box.set_stroke(opacity=0.7)
        
        # Position legend below the plot
        legend = gx.VGroup(legend_box, legend_items)
        legend.move_to(axes_1.get_bottom() + gx.DOWN * 0.15 + gx.RIGHT * 0.1)


        fit_line_1, test_error_1, train_error_1, y_train_pred_1, y_test_pred_1 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=1, color=GREEN)
        fit_line_2, test_error_2, train_error_2, y_train_pred_2, y_test_pred_2 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=2, color=YELLOW)
        fit_line_3, test_error_3, train_error_3, y_train_pred_3, y_test_pred_3 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=3, color=gx.ORANGE)
        fit_line_4, test_error_4, train_error_4, y_train_pred_4, y_test_pred_4 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=4, color='#FF00FF')
        fit_line_5, test_error_5, train_error_5, y_train_pred_5, y_test_pred_5 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=5, color='#FFFFFF')
        fit_line_10, test_error_10, train_error_10, y_train_pred_10, y_test_pred_10 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=10, color='#be1e2d')

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
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
        )
        axes_2.move_to([4.0, 0.48, 0])


        degrees = [1, 2, 3, 4, 5, 10]
        train_errors = [0.221350, 0.031825, 0.000803, 0.000000, 0.0, 0.0]
        test_errors = [0.691636, 0.079449, 0.727023, 1.328535, 0.694120, 0.481703]

        train_error_dots = gx.VGroup(*[gx.Dot(axes_2.c2p(degrees[i], train_errors[i]), radius=0.08)
                                    for i in range(len(degrees))])
        test_error_dots = gx.VGroup(*[gx.Dot(axes_2.c2p(degrees[i], test_errors[i]), radius=0.08)
                                  for i in range(len(degrees))])
        train_error_dots.set_color(YELLOW)
        test_error_dots.set_color(CYAN)
        test_error_dots.set_opacity(0.7)


        error_axis_svg=gx.SVGMobject(svg_dir+'/p8_15_2-05.svg') #[1:]
        degree_label=error_axis_svg[39:]
        error_axis_svg.scale(2.95)
        error_axis_svg.move_to([4.79, 0.75, 0])

        extended_axis_labels_svg=gx.SVGMobject(svg_dir+'/p46_56_2-04.svg')[1:]
        extended_axis_svg=gx.SVGMobject(svg_dir+'/p46_56_2-05.svg')[1:]
        extended_axis_group=gx.Group(extended_axis_svg, extended_axis_labels_svg)
        extended_axis_group.scale(4.5)
        extended_axis_group.move_to([6.57, -1.98, 0])
        extended_axis_svg.scale([0.67, 1, 1], about_point=extended_axis_svg.get_left())

        # self.remove()
        # self.add(error_axis_svg)

        # train_error_bars = VGroup()
        # for i in range(len(x_train)):
        #     point_pos = axes_1.c2p(x_train[i], y_train[i])
        #     fit_pos = axes_1.c2p(x_train[i], y_train_pred[i])
        #     # Always draw from the lower point to the higher point
        #     if point_pos[1] > fit_pos[1]:  # point is above fit
        #         error_bar = Line(fit_pos, point_pos, color=YELLOW, stroke_width=3)
        #     else:  # point is below fit
        #         error_bar = Line(point_pos, fit_pos, color=YELLOW, stroke_width=3)
        #     train_error_bars.add(error_bar)

        # # self.add(train_error_bars)

        # target_bars = VGroup()
        # bar_height = train_errors[0] / len(train_error_bars)  # Each bar gets equal portion of total height
        # x_pos = degrees[0]  # x position is degree 1

        # for i in range(len(train_error_bars)):
        #     # Stack bars on top of each other
        #     bottom_y = i * bar_height
        #     top_y = (i + 1) * bar_height
        #     bottom = axes_2.c2p(x_pos, bottom_y)
        #     top = axes_2.c2p(x_pos, top_y)
        #     target_bar = Line(bottom, top, color=YELLOW, stroke_width=3)
        #     target_bars.add(target_bar)

        #Setup in p46
        self.wait()
        test_dots.set_color('#008080')
        self.set_view(0, 0, 0, (0.8, 0.54, 0.0), 8.94)
        parabola.set_stroke(opacity=0.5)
        self.add(curve_fit_axis_svg, error_axis_svg[1:])
        self.add(extended_axis_svg)

        self.play(gx.Create(parabola), gx.LaggedStart(*[gx.FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15), run_time=2)
        self.add(legend)


        error_curves_svg=gx.SVGMobject(svg_dir+'/p8_15_2-06.svg') #[1:]
        error_curves_svg.scale(3.1)
        error_curves_svg.move_to([4.82, 1.2, 0])
        error_curves_svg[0].set_color('#00BBBB')

        self.wait()
        self.play(gx.LaggedStart(*[gx.FadeIn(train_error_dots[i]) for i in range(4)], lag_ratio=0.15),
                  gx.LaggedStart(*[gx.FadeIn(test_error_dots[i]) for i in range(4)], lag_ratio=0.15),
                  gx.FadeIn(error_curves_svg), run_time=2)
        self.wait()

        #Ok, now into p47 wehere we talk about existing fits. 
        train_pos = train_error_dots[1].get_center()
        test_pos = test_error_dots[1].get_center()

        # Calculate box dimensions with some padding
        padding = 0.3
        left = min(train_pos[0], test_pos[0]) - padding
        right = max(train_pos[0], test_pos[0]) + padding
        bottom = min(train_pos[1], test_pos[1]) - padding
        top = max(train_pos[1], test_pos[1]) + padding

        # Create the box
        degree_2_box = gx.Rectangle(
            width=right - left,
            height=top - bottom,
            stroke_color='#FF00FF',  # Magenta
            stroke_width=3,
            fill_opacity=0
        )
        degree_2_box.move_to([(left + right) / 2, (bottom + top) / 2, 0])


        box_padding = 0.15
        degree_3_train_box = gx.Rectangle(
            width=0.3, height=0.3,
            stroke_color='#FF00FF', stroke_width=3, fill_opacity=0
        ).move_to(train_error_dots[2].get_center())

        degree_3_test_box = gx.Rectangle(
            width=0.3, height=0.3,
            stroke_color='#FF00FF', stroke_width=3, fill_opacity=0
        ).move_to(test_error_dots[2].get_center())

        degree_4_train_box = gx.Rectangle(
            width=0.3, height=0.3,
            stroke_color='#FF00FF', stroke_width=3, fill_opacity=0
        ).move_to(train_error_dots[3].get_center())

        degree_4_test_box = gx.Rectangle(
            width=0.3, height=0.3,
            stroke_color='#FF00FF', stroke_width=3, fill_opacity=0
        ).move_to(test_error_dots[3].get_center())

        self.play(gx.Create(fit_line_2), run_time=2)
        self.play(gx.Create(degree_2_box), run_time=1.5)
        self.wait()

        self.play(gx.FadeOut(degree_2_box), fit_line_2.animate.set_stroke(opacity=0.2), run_time=2)
        self.play(gx.Create(fit_line_3), run_time=3)
        self.wait()
        self.play(gx.Create(degree_3_train_box))
        self.wait()
        self.play(gx.Create(degree_3_test_box))
        self.wait()

        self.play(gx.FadeOut(degree_3_train_box), gx.FadeOut(degree_3_test_box), fit_line_3.animate.set_stroke(opacity=0.2), run_time=2)
        self.wait()


        polynomial_equation_4=math_tex('f(x)=ax^4+bx^3+cx^2+dx+e', font_size=28).set_color('#FF00FF')
        polynomial_equation_4.move_to([-2.5, 3, 0])
        # self.add(polynomial_equation_4)

        self.play(gx.Create(fit_line_4), run_time=3)
        self.wait()
        self.play(gx.Create(degree_4_test_box))
        self.wait()
        self.play(gx.Create(degree_4_train_box))
        self.wait()

        self.play(gx.Write(polynomial_equation_4))
        #Ok think we just add quick arrows in illustrator pointing to the 5 parameters. 
        self.wait()


        # P48
        # Vertical dotted line at degree 4
        interp_threshold_line = gx.DashedLine(
            start=axes_2.c2p(4, -0.2),
            end=axes_2.c2p(4, 1.5),
            color=gx.WHITE,
            stroke_width=3,
            dash_length=0.1
        )

        interp_threshold_label = gx.Text(
            "Interpolation Threshold",
            font_size=24,
            font='myraid-pro',
            color=gx.WHITE
        )
        interp_threshold_label.next_to(axes_2.c2p(4, 0), gx.DOWN, buff=0.9)

        self.wait()
        self.play(gx.FadeOut(degree_4_test_box), gx.FadeOut(degree_4_train_box), gx.Create(interp_threshold_line), gx.Write(interp_threshold_label), run_time=1.5)
        

        # Ok now let's fade out a bunch of stuff, zoom out a bit, need to expand my error plot axes a bit
        # That's going to be a little annoying, but we can figure it out
        # Would love to do it once and have room for 5th order and 10th order, but we'll see. 

        # extended_axis_group.move_to([6.57, -2.05, 0])
        # self.set_view(0, 0, 0, (7.3, -0.18, 0.0), 7.39)
        # self.add(extended_axis_group)

        
        self.wait()

        self.remove(interp_threshold_line, interp_threshold_label)
        self.play(self.animate_view(0, 0, 0, (1.57, 0.64, 0.0), 9.98),
                  gx.FadeOut(error_curves_svg),
                  gx.FadeOut(fit_line_2),
                  gx.FadeOut(fit_line_3),
                  gx.FadeOut(fit_line_4),
                  gx.FadeOut(polynomial_equation_4),
                  # FadeOut(interp_threshold_line),
                  # FadeOut(interp_threshold_label),
                  #interp_threshold_line.animate.set_opacity(0.5),
                  #interp_threshold_label.animate.set_opacity(0.5),
                  degree_label.animate.move_to([9.5, -2.1, 0]),
                  extended_axis_svg.animate.scale([1.3, 1, 1], about_point=extended_axis_svg.get_left()),
                  #Replace axes with longer one here!
                  run_time=2)
        self.add(extended_axis_group[1][6], extended_axis_group[1][8])
        

        lil_arrow_1=gx.SVGMobject(svg_dir+'/p46_56_2-06.svg') #[1:]
        lil_arrow_1.scale(0.3)
        lil_arrow_1.set_color('#FF00FF')
        lil_arrow_1.move_to([8.25, -2.55, 0])

        # self.add(lil_arrow_1)

        polynomial_equation_5=math_tex('f(x)=ax^5+bx^4+cx^3+dx^2+ex+f', font_size=32).set_color('#FF00FF')
        # polynomial_equation_5.move_to([-2.5, 3, 0]) 
        polynomial_equation_5.move_to([5.0, -2.8, 0])
        
        self.wait()
        self.play(gx.Write(polynomial_equation_5))
        self.play(gx.Write(lil_arrow_1))
        self.wait()

        # Now add arrows pointing to each arrow in illustrator - already made em
        # Ok now time to tackle N different curve fits, probalby back to jupyter for a minute
        # to figure out who exactly i want to do this 
        # I need to pick out two examples too and figurout thier vondermonte coefficicent
        # Cool wiil do that and then come back.  

        # Load up 100 different perfect 5th order fits and coefficients from jupyter notebook
        # I want to highlight 48 (chill) and 72 (nuts)
        all_fits=np.load(welch_path('double_descent/graphics/all_fits_oct_13_1.npy'))
        all_coeffs=np.load(welch_path('double_descent/graphics/all_coefs_oct_13_1.npy'))


        all_fifth_order_fits=gx.VGroup()
        for af in all_fits:
            fit_points = [axes_1.c2p(all_x[i], af[i]) for i in range(len(all_x))]
            fit_line = gx.VMobject(stroke_width=3)
            fit_line.set_points_smoothly(fit_points)
            fit_line.set_color('#FF00FF')
            all_fifth_order_fits.add(fit_line)

        all_fifth_order_fits.set_stroke(width=1.0, opacity=0.4)

        # fit_points_72 = [axes_1.c2p(all_x[i], all_fits[72][i]) for i in range(len(all_x))]
        # dashed_fit_72 = DashedVMobject(all_fifth_order_fits[72], num_dashes=200, color='#FF00FF')
        # dashed_fit_72.set_stroke(width=3)
        # self.add(dashed_fit_72)

        self.wait() #This is DOPE
        self.play(*[gx.Create(all_fifth_order_fits[i]) for i in range(len(all_fifth_order_fits))], run_time=7)
        

        # Add footnote about about minium norm calculation in editing. 
        # Ok this little bit is fairly complex, but I think one of the 2-3 most complex parts of the vid. 
        # Ok I think i really want to focus on on the fits here, and temporarily fade some stuff out. 
        polynomial_equation_5a=math_tex('f(x)=-0.66x^5-2.85x^4-0.01x^3+3.12x^2+0.36x-0.83', font_size=36).set_color('#FF00FF')
        polynomial_equation_5a.move_to([4.8, -2.8, 0])


        self.wait()
        self.play(gx.FadeOut(polynomial_equation_5),
                  gx.FadeOut(lil_arrow_1),
                  gx.FadeOut(error_axis_svg[1:]),
                  gx.FadeOut(degree_label),
                  gx.FadeOut(extended_axis_svg),
                  gx.FadeOut(extended_axis_group[1][6]),
                  gx.FadeOut(extended_axis_group[1][8]),
                  gx.FadeOut(train_error_dots[:4]),
                  gx.FadeOut(test_error_dots[:4]),
                  gx.FadeOut(legend),
                  self.animate_view(0, 0, 0, (1.43, 0.35, 0.0), 11.25),
                  all_fifth_order_fits[:72].animate.set_stroke(width=0.5, opacity=0.1).set_color(CHILL_BROWN),
                  all_fifth_order_fits[73:].animate.set_stroke(width=0.5, opacity=0.1).set_color(CHILL_BROWN),
                  all_fifth_order_fits[72].animate.set_stroke(width=4, opacity=0.9), #.set_color(YELLOW),
                  run_time=4
            )
        self.play(gx.Write(polynomial_equation_5a), run_time=2)

        self.wait()
        self.play(all_fifth_order_fits[48].animate.set_stroke(width=4, opacity=0.9).set_color(YELLOW), 
                  run_time=2)

        polynomial_equation_5b=math_tex('f(x)=-0.54x^5-1.99x^4+0.95x^3+1.36x^2-0.11x-0.01', font_size=36).set_color(YELLOW)
        polynomial_equation_5b.move_to([4.8, 2.3, 0])
        self.wait()
        self.play(gx.Write(polynomial_equation_5b), run_time=2)
        self.wait()

       #  array([-0.5370232 , -1.9890692 ,  0.9468571 ,  1.3580599 , -0.10506456,
       # -0.2067696 ], dtype=float32)


        coeffs_squared_1 = math_tex(
            r'(-0.66)^2 + (-2.85)^2 + (-0.01)^2 + (3.12)^2 + (0.36)^2 + (-0.83)^2',
            font_size=32
        ).set_color('#FF00FF')
        coeffs_squared_1.next_to(polynomial_equation_5a, gx.DOWN, buff=0.4).shift([0.4, 0, 0])

        #To do ->  finish matching indices here so all 6 terms move down nicely. 
        self.wait()
        self.play(gx.ReplacementTransform(polynomial_equation_5a[5:10].copy(), coeffs_squared_1[1:6]),
                  gx.ReplacementTransform(polynomial_equation_5a[12:17].copy(), coeffs_squared_1[10:15]),
                  gx.ReplacementTransform(polynomial_equation_5a[19:24].copy(), coeffs_squared_1[19:24]),
                  gx.ReplacementTransform(polynomial_equation_5a[27:31].copy(), coeffs_squared_1[28:32]),
                  gx.ReplacementTransform(polynomial_equation_5a[34:38].copy(), coeffs_squared_1[36:40]),
                  gx.ReplacementTransform(polynomial_equation_5a[39:44].copy(), coeffs_squared_1[44:49]),
                  run_time=2)

        self.play(gx.FadeIn(coeffs_squared_1[0]),
                  gx.FadeIn(coeffs_squared_1[6:10]),
                  gx.FadeIn(coeffs_squared_1[15:19]),
                  gx.FadeIn(coeffs_squared_1[24:28]),
                  gx.FadeIn(coeffs_squared_1[32:36]),
                  gx.FadeIn(coeffs_squared_1[40:44]),
                  gx.FadeIn(coeffs_squared_1[49:]),
                  run_time=1)
        self.wait()


        # self.add(coeffs_squared_1) #Remove this after we finish the nice animated version above. 

        result_eq_1 = math_tex('=19.13', font_size=32).set_color('#FF00FF')
        result_eq_1.next_to(coeffs_squared_1, gx.DOWN, buff=0.4, aligned_edge=gx.LEFT)
        self.play(gx.Write(result_eq_1))


        #0.54^2+1.99^2+0.95^2+1.36^2+0.11^2+0.01^2

        # Create the squared coefficients expression
        coeffs_squared_2 = math_tex(
            r'(-0.54)^2 + (-1.99)^2 + (0.95)^2 + (1.36)^2 + (-0.11)^2 + (-0.01)^2',
            font_size=32
        ).set_color(YELLOW)
        coeffs_squared_2.next_to(polynomial_equation_5b, gx.DOWN, buff=0.3)

        self.play(gx.ReplacementTransform(polynomial_equation_5b[5:10].copy(), coeffs_squared_2[1:6]),
                  gx.ReplacementTransform(polynomial_equation_5b[12:17].copy(), coeffs_squared_2[10:15]),
                  gx.ReplacementTransform(polynomial_equation_5b[20:24].copy(), coeffs_squared_2[19:23]),
                  gx.ReplacementTransform(polynomial_equation_5b[27:31].copy(), coeffs_squared_2[27:31]),
                  gx.ReplacementTransform(polynomial_equation_5b[33:38].copy(), coeffs_squared_2[35:40]),
                  gx.ReplacementTransform(polynomial_equation_5b[39:44].copy(), coeffs_squared_2[44:49]),
                  run_time=2)

        self.play(gx.FadeIn(coeffs_squared_2[0]),
                  gx.FadeIn(coeffs_squared_2[6:10]),
                  gx.FadeIn(coeffs_squared_2[15:19]),
                  gx.FadeIn(coeffs_squared_2[23:27]),  # Changed from 24:28 to match indices better
                  gx.FadeIn(coeffs_squared_2[31:35]),  # Changed from 32:36 to match indices better
                  gx.FadeIn(coeffs_squared_2[40:44]),
                  gx.FadeIn(coeffs_squared_2[49:]),
                  run_time=1)
        self.wait()

        # To-do -> nice animation bringing coefficeints from polynomial_equation_5b into coeffs_squared_2
        # and adding parenthesis and squares. 
        # self.add(coeffs_squared_2)


        result_eq_2 = math_tex('=7.04', font_size=32).set_color(YELLOW)
        result_eq_2.next_to(coeffs_squared_2, gx.DOWN, buff=0.4, aligned_edge=gx.LEFT)
        self.play(gx.Write(result_eq_2))

        # Right at the end of p50 here, fade out magenta curve. 
        self.wait()
        self.play(gx.FadeOut(polynomial_equation_5a),
                  gx.FadeOut(coeffs_squared_1),
                  gx.FadeOut(result_eq_1),
                  all_fifth_order_fits[72].animate.set_stroke(width=0.5, opacity=0.1).set_color(CHILL_BROWN),
                  run_time=2)
        self.wait()

        #P51 lets go
        self.play(all_fifth_order_fits[:48].animate.set_stroke(width=1.0, opacity=0.4),
                  all_fifth_order_fits[49:].animate.set_stroke(width=1.0, opacity=0.4),
                  run_time=2.5 )
        self.remove(train_dots, test_dots, all_fifth_order_fits[48])
        self.add(train_dots, test_dots, all_fifth_order_fits[48])
        self.wait()

        # Ok time to bring back in error plot, lose all the other curves, lose thet equations, 
        # zoom back in a bit, and then do a "measure error with lines again" 

        # Ok I think i want more out of the way legend now -> might also be nice to 
        # have a more out of the way one earlier, we'll see!
        legend_train_2=legend_train.copy()
        legend_test_2=legend_test.copy()
        legend_line_item_2=legend_line_item.copy()
        legend_items_2 = gx.VGroup(legend_train_2, legend_test_2, legend_line_item_2).arrange(gx.DOWN, buff=0.1, aligned_edge=gx.LEFT)
        
        legend_box_2 = gx.RoundedRectangle(
            width=legend_items_2.width + 0.6,
            height=legend_items_2.height + 0.3,
            corner_radius=0.08,
            stroke_color=CHILL_BROWN,
            stroke_width=2,
            fill_color=None,
            fill_opacity=0.0
        )
        legend_box_2.set_stroke(opacity=0.7)
        
        # Position legend below the plot
        legend_2 = gx.VGroup(legend_box_2, legend_items_2)
        legend_2.move_to(axes_1.get_bottom() + gx.DOWN * 0.5 + gx.RIGHT * -2.0)
        # self.add(legend_2)

        self.wait()
        self.remove(polynomial_equation_5b, coeffs_squared_2, result_eq_2)
        self.play(self.animate_view(0, 0, 0, (1.6, 0.58, 0.0), 9.91),
                  #FadeOut(polynomial_equation_5b),
                  #FadeOut(coeffs_squared_2),
                  #FadeOut(result_eq_2),
                  all_fifth_order_fits[:48].animate.set_stroke(opacity=0.0),
                  all_fifth_order_fits[49:].animate.set_stroke(opacity=0.0),
                  gx.FadeIn(error_axis_svg[1:]),
                  degree_label.animate.set_opacity(1.0),
                  gx.FadeIn(extended_axis_svg),
                  gx.FadeIn(extended_axis_group[1][6]),
                  gx.FadeIn(extended_axis_group[1][8]),
                  gx.FadeIn(train_error_dots[:4]),
                  gx.FadeIn(test_error_dots[:4]),
                  gx.FadeIn(legend_2),
                  run_time=4)
        self.wait()

        self.play(gx.Create(train_error_dots[4]))
        self.wait(0)



        #Hmm I think we want to bring back the 4th order fit in low opacity here, comparison is important. 

        fit_line_4.set_color(gx.MAROON_B).set_stroke(opacity=0.8)
        self.play(gx.Create(fit_line_4), run_time=2)
        self.wait()
        

        # Create Testing Error Bars
        test_error_bars = gx.VGroup()
        for i in range(len(x_test)):
            point_pos = axes_1.c2p(x_test[i], y_test[i])
            fit_pos = axes_1.c2p(x_test[i], y_test_pred_5[i])
            # Always draw from the lower point to the higher point
            if point_pos[1] > fit_pos[1]:  # point is above fit
                error_bar = gx.Line(fit_pos, point_pos, color=TEST_BLUE, stroke_width=3)
            else:  # point is below fit
                error_bar = gx.Line(point_pos, fit_pos, color=TEST_BLUE, stroke_width=3)
            test_error_bars.add(error_bar)
        

        # Create target bars for error plot (stacked bars)
        target_test_bars = gx.VGroup()
        bar_height = test_errors[4] / len(test_error_bars)
        x_pos = degrees[4]
        
        for i in range(len(test_error_bars)):
            bottom_y = i * bar_height
            top_y = (i + 1) * bar_height
            bottom = axes_2.c2p(x_pos, bottom_y)
            top = axes_2.c2p(x_pos, top_y)
            target_bar = gx.Line(bottom, top, color=TEST_BLUE, stroke_width=3)
            target_test_bars.add(target_bar)


        # Animate Test Error Bars
        self.wait()
        self.play(
            gx.LaggedStart(*[gx.Create(bar) for bar in test_error_bars], lag_ratio=0.1),
            run_time=1.5
        )
        self.bring_to_front(all_fifth_order_fits[48])  # Keep fit line in front of error bars
        

        test_error_bars_copy = test_error_bars.copy()
        self.play(gx.ReplacementTransform(test_error_bars_copy, target_test_bars), run_time=3.0)
        self.play(gx.Create(test_error_dots[4]))


        self.play(gx.FadeOut(target_test_bars), gx.FadeOut(test_error_bars))
        self.wait()
        # Ok so now I need a little double descent curve action! 
        # I think just do illustrator/premiere here I think. 

        # Ok I think 52 might be all in illustrator, going to hack on that
        # Ok yeah that's going to be al illustrator
        # Now let me jump on p54 here. 

        #hmm yeah I think totally take away the fourth order fit? Shit's about to get complicated
        self.play(gx.FadeOut(fit_line_4), all_fifth_order_fits[48].animate.set_stroke(opacity=0.0), run_time=2.0)
        self.wait()



        all_fits=np.load(welch_path('double_descent/graphics/all_fits_degree_10_oct_13_1.npy'))
        all_tenth_order_fits=gx.VGroup()
        for af in all_fits:
            fit_points = [axes_1.c2p(all_x[i], af[i]) for i in range(len(all_x))]
            fit_line = gx.VMobject(stroke_width=3)
            fit_line.set_points_smoothly(fit_points)
            fit_line.set_color('#FFFFFF')  #FF00FF
            all_tenth_order_fits.add(fit_line)

        all_tenth_order_fits.set_stroke(width=1.0, opacity=0.3)

        self.wait()
        self.play(*[gx.Create(all_tenth_order_fits[i]) for i in range(len(all_tenth_order_fits))], run_time=7)
        self.remove(train_dots); self.add(train_dots)
        self.wait()

        self.remove(all_tenth_order_fits[66]); 
        all_tenth_order_fits[66].set_stroke(width=5, opacity=0.9).set_color('#FF00FF')

        self.play(gx.Create(all_tenth_order_fits[66]), run_time=3.0)
        # self.play(all_tenth_order_fits[66].animate.set_stroke(width=3, opacity=0.9).set_color('#FF00FF'))
        
        #66self.add(all_tenth_order_fits[66])

        fit_line_10.set_stroke(width=4).set_color(YELLOW)
        # self.add(fit_line_10)

        self.wait()
        self.play(
                  # all_tenth_order_fits[66].animate.set_stroke(width=1.0, opacity=0.3).set_color('#FFFFFF'),
                  all_tenth_order_fits.animate.set_stroke(width=0.5, opacity=0.01).set_color('#FFFFFF'),
                  # FadeIn(fit_line_10),
                  run_time=3.0)
        self.add(fit_line_10)
        self.wait()

        #Ok I think probably do the error bars again and tweak the VO to give a little more space for it?
        test_error_bars = gx.VGroup()
        for i in range(len(x_test)):
            point_pos = axes_1.c2p(x_test[i], y_test[i])
            fit_pos = axes_1.c2p(x_test[i], y_test_pred_10[i])
            # Always draw from the lower point to the higher point
            if point_pos[1] > fit_pos[1]:  # point is above fit
                error_bar = gx.Line(fit_pos, point_pos, color=TEST_BLUE, stroke_width=3)
            else:  # point is below fit
                error_bar = gx.Line(point_pos, fit_pos, color=TEST_BLUE, stroke_width=3)
            test_error_bars.add(error_bar)
        

        # Create target bars for error plot (stacked bars)
        target_test_bars = gx.VGroup()
        bar_height = test_errors[5] / len(test_error_bars)
        x_pos = degrees[5]
        
        for i in range(len(test_error_bars)):
            bottom_y = i * bar_height
            top_y = (i + 1) * bar_height
            bottom = axes_2.c2p(x_pos, bottom_y)
            top = axes_2.c2p(x_pos, top_y)
            target_bar = gx.Line(bottom, top, color=TEST_BLUE, stroke_width=3)
            target_test_bars.add(target_bar)


        # Animate Test Error Bars
        self.wait()
        self.play(
            gx.LaggedStart(*[gx.Create(bar) for bar in test_error_bars], lag_ratio=0.1),
            run_time=1.5
        )
        # self.bring_to_front(all_fifth_order_fits[48])  # Keep fit line in front of error bars
        

        test_error_bars_copy = test_error_bars.copy()
        final_ticks=gx.SVGMobject(svg_dir+'/p46_56_2-14.svg')[1:]
        final_ticks.scale(4.5)
        final_ticks.move_to([12.98, -2, 0])

        self.wait()

        self.play(self.animate_view(0, 0, 0, (5.17, 0.23, 0.0), 13.80),
                  extended_axis_svg.animate.scale([1.94, 1, 1], about_point=extended_axis_svg.get_left()),
                  degree_label.animate.move_to([16.7, -2.1, 0]), 
                  gx.ReplacementTransform(test_error_bars_copy, target_test_bars),
                  run_time=4.0)
        self.add(final_ticks, train_error_dots[5])
        self.play(gx.Create(test_error_dots[5]))
        


        # self.play(ReplacementTransform(test_error_bars_copy, target_test_bars), run_time=3.0)
        # self.play(ShowCreation(test_error_dots[5]))
        # self.add(train_error_dots[5])


        #Make these sucker into a a nice animation.
        # self.set_view(0, 0, 0, (5.17, 0.23, 0.0), 13.80)
        # self.play(extended_axis_svg.animate.scale([1.94, 1, 1], about_point=extended_axis_svg.get_left()))
        # degree_label.move_to([16.7, -2.1, 0])


        #Ok cool so for 56 I think we replay drawing the 4th order curve, 
        #And and probably bring back in interpolation treshold dotted line and label
        self.wait()
        self.play(all_tenth_order_fits.animate.set_stroke(opacity=0.0), 
                  gx.FadeOut(test_error_bars),
                  gx.FadeOut(fit_line_10),
                  run_time=2)
        self.wait()

        self.play(gx.FadeIn(interp_threshold_label), gx.FadeIn(interp_threshold_line))
        fit_line_4.set_color(YELLOW)

        self.play(gx.Create(fit_line_4), run_time=3)
        self.wait()

        self.play(gx.FadeOut(fit_line_4))
        self.wait()

        tenth_order_group=gx.VGroup(fit_line_10, all_tenth_order_fits)
        self.remove(tenth_order_group)
        tenth_order_group.set_stroke(width=1.0, opacity=0.3)
        # fit_line_10.set_stroke(width=1.0, opacity=0.3)
        self.play(*[gx.Create(all_tenth_order_fits[i]) for i in range(len(all_tenth_order_fits))], run_time=7)
        self.remove(train_dots); self.add(train_dots)

        self.play(gx.FadeOut(all_tenth_order_fits),
                 fit_line_10.animate.set_color(YELLOW).set_stroke(width=3, opacity=0.9), #a little jumpy buy maybe ok?
                 run_time=3.0)
        self.wait()




        self.wait(20)
        # Scene authoring stops here in the interactive editor.

        # self.bring_to_front(train_error_dots[0])


        # train_error_bars_copy = train_error_bars.copy()
        # self.add(train_error_bars_copy)
        # self.bring_to_front(all_fifth_order_fits[48])  # Bring fit line to front after adding copy
        # train_error_bars.set_opacity(0.5)
        # self.wait()


        # # Copy test error bars before moving
        # test_error_bars_copy = test_error_bars.copy()
        # self.add(test_error_bars_copy)
        # self.bring_to_front(fit_line_1)  # Bring fit line to front after adding copy
        
        # # Fade out yellow bars first, then move blue bars
        # self.play(target_train_bars.animate.set_opacity(0.0), run_time=1.5)
        # self.bring_to_front(train_error_dots[0])
        # self.bring_to_front(fit_line_1)  # Keep fit line in front
        # self.play(ReplacementTransform(test_error_bars_copy, target_test_bars), run_time=3.0)
        # self.bring_to_front(train_error_dots[0])
        # self.play(ShowCreation(test_error_dots[0]))










