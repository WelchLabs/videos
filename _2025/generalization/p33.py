from welch_paths import welch_path
import manimgx as gx
from curve_fitting import eval_legendre_poly, f, fit_legendre_pinv, get_noisy_data
import numpy as np
from welch_geometry import diagram_axes
from welch_scene import WelchScene

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

graphics_dir=welch_path('double_descent/graphics/')
svg_dir=welch_path('double_descent/graphics/to_manim')

# def f(x): return 0.5*(x**4-3*x**2)
# def f(x): return np.add(2.0 * x, np.cos(x * 25)) #[:, 0]


class p33_1(WelchScene):
    '''
    
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
        test_dots.set_color(CYAN).set_opacity(0.5)


        self.set_view(0, 0, 0, (-2.94, 0.08, 0.0), 7.45)

        self.wait()
        self.play(gx.Write(curve_fit_axis_svg), run_time=3)
        self.play(gx.Create(parabola), gx.LaggedStart(*[gx.FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15), run_time=2)
        self.wait()

        self.add(legend)

        # interpolating_fit_svg=SVGMobject(svg_dir+'/p33.svg')[1:] 
        # interpolating_fit_svg.scale(3.72)
        # interpolating_fit_svg.move_to([-2.92, 0.6, 0])

        # self.wait()
        # self.play(Write(interpolating_fit_svg))

        # self.add(interpolating_fit_svg)


        self.wait(20)
        # Scene authoring stops here in the interactive editor.
