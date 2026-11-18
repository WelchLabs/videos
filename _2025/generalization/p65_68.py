from welch_paths import welch_path
import manimgx as gx
from curve_fitting import eval_legendre_poly, f, fit_legendre_pinv, get_fit_line, get_noisy_data
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
TEST_BLUE='#008080' 

graphics_dir=welch_path('double_descent/graphics/')
svg_dir=welch_path('double_descent/graphics/to_manim')

class p65_68_v2(WelchScene):
    '''
    Ok, so the idea here is to potentially do a bit of a composite thing
    in the dark region next to the book - 
    '''
    def construct(self):



        # Ok, so i think first let's bring in a version of both plots, maybe with some nice animation. 
        # I think we keep the same side-by-side orientation we used earlier, and
        # if I want to arange these plots vertically, I can do that in premiere. 




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


        fit_line_1, test_error_1, train_error_1, y_train_pred_1, y_test_pred_1 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=1, color=GREEN)
        fit_line_2, test_error_2, train_error_2, y_train_pred_2, y_test_pred_2 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=2, color=YELLOW)
        fit_line_3, test_error_3, train_error_3, y_train_pred_3, y_test_pred_3 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=3, color=gx.ORANGE)
        fit_line_4, test_error_4, train_error_4, y_train_pred_4, y_test_pred_4 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=4, color='#FF00FF')
        fit_line_5, test_error_5, train_error_5, y_train_pred_5, y_test_pred_5 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=5, color='#FFFFFF')
        fit_line_10, test_error_10, train_error_10, y_train_pred_10, y_test_pred_10 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=10, color='#be1e2d')

        ##


        ##

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

        #Probably need to noodle with scale here!
        extended_axis_svg.scale([0.85, 1, 1], about_point=extended_axis_svg.get_left()) 
        
        test_dots.set_color('#008080')
        self.set_view(0, 0, 0, (1.32, 0.56, 0.0), 9.80)

        parabola.set_stroke(opacity=0.5)
        self.wait()
        self.play(gx.Write(curve_fit_axis_svg))
        self.play(gx.Create(parabola), gx.LaggedStart(*[gx.FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15), run_time=2)
        

        degree_label.scale(1.6)
        degree_label.move_to([5.5, -2.5, 0])
        self.wait()

        self.play(gx.Write(error_axis_svg[1:]),
            gx.Write(extended_axis_svg),
            gx.Write(extended_axis_group[1][6]),
            gx.Write(extended_axis_group[1][8]))

        double_descent_curve_svg=gx.SVGMobject(svg_dir+'/p46_56_2-12.svg')

        double_descent_curve_svg.scale(2.6)
        double_descent_curve_svg.move_to([5.62, 1.12, 0])

        

        self.wait()
        self.play(gx.Create(train_error_dots), gx.Create(test_error_dots), run_time=2)
        self.play(gx.Write(double_descent_curve_svg))
        self.wait()
        #Hmm kinda feel like i want the double descent curve in manim?


        # self.play(ShowCreation(double_descent_curve_svg))

        interp_threshold_line = gx.DashedLine(
            start=axes_2.c2p(4, 0.0),
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
        interp_threshold_label.next_to(axes_2.c2p(4, 1.5), gx.UP, buff=0.2)

        self.wait()
        self.play(gx.FadeIn(interp_threshold_line), gx.Write(interp_threshold_label))

        strikethrough_line = gx.Line(
            start=degree_label.get_left() + gx.LEFT * 0.1,
            end=degree_label.get_right() + gx.RIGHT * 0.1,
            color=YELLOW,
            stroke_width=3
        )
        flexibility_label = gx.Text(
            "Flexibility",
            font_size=32,
            font='myriad-pro',
        ).set_color(YELLOW)

        flexibility_label.next_to(degree_label, gx.DOWN, buff=0.1).shift([0.02, 0, 0])


        self.wait()
        self.play(gx.Create(strikethrough_line), run_time=1.0)
        self.play(gx.Write(flexibility_label), run_time=1)
        self.wait()

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
        self.remove(train_dots); self.add(train_dots)

        fit_line_5.set_color('#FF00FF').set_stroke(width=4)
        self.wait()
        self.play(gx.FadeIn(fit_line_5), gx.FadeOut(all_fifth_order_fits), run_time=3)
        self.wait()

        fit_line_4.set_color(gx.MAROON_B)
        self.play(gx.Create(fit_line_4), run_time=3)
        

        # Will need to add some illustrator labels for the degrees of these fits. 
        # Ok on to P66 -> let's go!
        # Ok not sure if/how I'm going to try to composite here -> i probably don't want to 
        # composite for that long, ya know? Anyway, 
        # Ok so here I think I want to lose everything except the initial parabaola and 
        # training and testing points. Then I can explort different fits
        # I do want to massage my language here for sure - let's see what we can do!
        self.wait()
        self.play(gx.FadeOut(interp_threshold_line),
                  gx.FadeOut(interp_threshold_label),
                  gx.FadeOut(flexibility_label),
                  gx.FadeOut(strikethrough_line),
                  gx.FadeOut(test_error_dots),
                  gx.FadeOut(train_error_dots),
                  gx.FadeOut(double_descent_curve_svg),
                  gx.FadeOut(degree_label),
                  gx.FadeOut(error_axis_svg[1:]),
                  gx.FadeOut(extended_axis_svg),
                  gx.FadeOut(extended_axis_group[1][6]),
                  gx.FadeOut(extended_axis_group[1][8]),
                  gx.FadeOut(fit_line_5),
                  gx.FadeOut(fit_line_4),
                  self.animate_view(0, 0, 0, (-3.05, 0.65, 0.0), 7.66),
                  run_time=3.0)

        self.wait()
        self.play(gx.Create(fit_line_2), run_time=2)

        random_seed=25
        x,y=get_noisy_data(n_points, noise_level, random_seed)     
        x_train, y_train=x[:n_train_points], y[:n_train_points]
        x_test, y_test=x[n_train_points:],y[n_train_points:]
        fit_line_2b, test_error_2b, train_error_2b, y_train_pred_2b, y_test_pred_2b = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=2, color=YELLOW)
        
        train_dots_2 = gx.VGroup(*[gx.Dot(axes_1.c2p(x_train[i], y_train[i]), radius=0.08) for i in range(len(x_train))])
        test_dots_2 = gx.VGroup(*[ gx.Dot(axes_1.c2p(x_test[i], y_test[i]), radius=0.08) for i in range(len(x_test))])
        test_dots_2.set_color('#008080')
        train_dots_2.set_color(YELLOW)

        dots_with_x = []
        for i, dot in enumerate(train_dots_2):
            dots_with_x.append((x_train[i], dot, 'train'))
        for i, dot in enumerate(test_dots_2):
            dots_with_x.append((x_test[i], dot, 'test'))
        dots_with_x.sort(key=lambda item: item[0])
        sorted_dots = [item[1] for item in dots_with_x]


        self.wait()
        self.play(fit_line_2.animate.set_stroke(opacity=0.5), 
                  gx.FadeOut(train_dots),
                  gx.FadeOut(test_dots),
                  run_time=2)

        self.wait()
        self.play(gx.LaggedStart(*[gx.FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15), run_time=2)
        self.play(gx.Create(fit_line_2b), run_time=2)
        
        self.wait()
        self.play(gx.FadeOut(train_dots_2), gx.FadeOut(test_dots_2), fit_line_2b.animate.set_stroke(opacity=0.5))


        random_seed=52
        x,y=get_noisy_data(n_points, noise_level, random_seed)     
        x_train, y_train=x[:n_train_points], y[:n_train_points]
        x_test, y_test=x[n_train_points:],y[n_train_points:]
        fit_line_2c, test_error_2b, train_error_2b, y_train_pred_2b, y_test_pred_2b = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=2, color=YELLOW)
        
        train_dots_2 = gx.VGroup(*[gx.Dot(axes_1.c2p(x_train[i], y_train[i]), radius=0.08) for i in range(len(x_train))])
        test_dots_2 = gx.VGroup(*[ gx.Dot(axes_1.c2p(x_test[i], y_test[i]), radius=0.08) for i in range(len(x_test))])
        test_dots_2.set_color('#008080')
        train_dots_2.set_color(YELLOW)

        dots_with_x = []
        for i, dot in enumerate(train_dots_2):
            dots_with_x.append((x_train[i], dot, 'train'))
        for i, dot in enumerate(test_dots_2):
            dots_with_x.append((x_test[i], dot, 'test'))
        dots_with_x.sort(key=lambda item: item[0])
        sorted_dots = [item[1] for item in dots_with_x]

        self.wait()
        self.play(gx.LaggedStart(*[gx.FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15), run_time=2)
        self.play(gx.Create(fit_line_2c), run_time=2)


        #ok I'm going to add 50 more fits - Hmm how much compute do i want to do here vs 
        # in jupyter and import
        # Eh kinda feel like i want to import?


        all_variance_fits_np=np.load(welch_path('double_descent/graphics/variance_fits_oct_14_1.npy'))

        all_variance_fits=gx.VGroup()
        for af in all_variance_fits_np:
            fit_points = [axes_1.c2p(all_x[i], af[i]) for i in range(len(all_x))]
            fit_line = gx.VMobject(stroke_width=3)
            fit_line.set_points_smoothly(fit_points)
            fit_line.set_color(YELLOW)
            all_variance_fits.add(fit_line)
        all_variance_fits.set_stroke(width=1.0, opacity=0.4)

        self.wait()
        self.play(fit_line_2.animate.set_stroke(width=1.0, opacity=0.5),
                  fit_line_2b.animate.set_stroke(width=1.0, opacity=0.5),
                  fit_line_2c.animate.set_stroke(width=1.0, opacity=0.5),
                  gx.FadeOut(train_dots_2),
                  gx.FadeOut(test_dots_2),
                  gx.FadeIn(all_variance_fits),
                  run_time=3.0)


        # Hmm do we need to switch to ~30 datapoints at some point here?
        # Not sure yet, let me push a little further on animations and language tweaks, and we'll see. 
        mean_fit=np.mean(all_variance_fits_np, 0)
        std_fit=np.std(all_variance_fits_np, 0)
        # bias=np.mean((all_variance_fits_np-mean_fit)**2)
        # variance=np.mean((all_variance_fits_np-mean_fit)**2) 

        fit_points = [axes_1.c2p(all_x[i], mean_fit[i]) for i in range(len(all_x))]
        mean_fit_line = gx.VMobject(stroke_width=3)
        mean_fit_line.set_points_smoothly(fit_points)
        mean_fit_line.set_color(YELLOW)
        mean_fit_line.set_stroke(width=3.0, opacity=0.9)

        upper_bound = mean_fit + std_fit
        lower_bound = mean_fit - std_fit

        # Create points for the shaded region (need to create a closed polygon)
        upper_points = [axes_1.c2p(all_x[i], upper_bound[i]) for i in range(len(all_x))]
        lower_points = [axes_1.c2p(all_x[i], lower_bound[i]) for i in range(len(all_x)-1, -1, -1)]

        # Combine upper and lower bounds to create a closed shape
        std_region_points = upper_points + lower_points

        # Create the shaded region
        std_region = gx.VMobject()
        std_region.set_points_as_corners(std_region_points + [upper_points[0]])  # Close the shape
        std_region.set_fill(YELLOW, opacity=0.2)
        std_region.set_stroke(width=0)


        # self.add(std_region)
        all_variance_fits_copy=all_variance_fits.copy()
        # self.add(all_variance_fits_copy)
        # all_variance_fits.set_stroke(opacity=0.1)

        self.wait()
        self.remove(fit_line_2, fit_line_2b, fit_line_2c)
        self.play(*[gx.ReplacementTransform(all_variance_fits[i], mean_fit_line) for i in range(len(all_variance_fits))],
                    gx.FadeIn(std_region),
                    run_time=5)
        self.bring_to_back(std_region)
        

        #Ok yeah that's not bad!
        #Ok now i'll add labels in illustartor, then I want to highlight the undelying fit. 
        mean_std_eqs=gx.SVGMobject(svg_dir+'/p65_68-02.svg')[1:]
        mean_std_eqs.scale(3.85)
        mean_std_eqs.move_to([0.62, 2.75, 0])

        bias_var_labels=gx.SVGMobject(svg_dir+'/p65_68-04.svg')[1:]
        bias_var_labels.scale(3.85)
        bias_var_labels.move_to([-1.8, 2.42, 0])

        self.wait()
        # self.add(mean_std_eqs)
        self.play(gx.Write(mean_std_eqs))


        parabola_copy=parabola.copy()
        parabola_copy.set_stroke(width=3.0, opacity=0.8).set_color(gx.WHITE)
        self.wait()
        self.bring_to_front(parabola_copy)
        self.play(gx.Create(parabola_copy), run_time=2.5)
        self.play(gx.Write(bias_var_labels[:16]), run_time=2)
        self.remove(parabola)

        #Ok if I'm doing all my labels in manim, shold bring in a "Target function" label. 
        # self.wait()
        parabola_y = f(all_x)

        # Create points for both curves
        parabola_points = [axes_1.c2p(all_x[i], parabola_y[i]) for i in range(len(all_x))]
        mean_fit_points = [axes_1.c2p(all_x[i], mean_fit[i]) for i in range(len(all_x))]

        # Create the region by going along one curve and back along the other
        bias_region_points = parabola_points + mean_fit_points[::-1]

        # Create the shaded region
        bias_region = gx.VMobject()
        bias_region.set_points_as_corners(bias_region_points + [parabola_points[0]])  # Close the shape
        bias_region.set_fill('#FF00FF', opacity=0.4)
        bias_region.set_stroke(width=0)


        #I think a zoom here, but does that mean I should put the equations/labels into manim?
        # self.play(self.animate_view(0, 0, 0, (-5.85, 2.35, 0.0), 3.46), run_time=3)
        self.wait()
        self.play(self.animate_view(0, 0, 0, (-2.36, 1.67, 0.0), 5.53), run_time=3)
        self.wait()
        self.play(gx.FadeIn(bias_region), gx.Write(bias_var_labels[18:24]), run_time=3)
        # self.play, run_time=2)
        self.wait()
        self.play(self.animate_view(0, 0, 0, (-3.05, 0.65, 0.0), 7.66), run_time=3)
        # self.bring_to_front(parabola_copy, mean_fit_line)s
        self.wait()
        self.play(gx.Write(bias_var_labels[24:]), gx.Write(bias_var_labels[16:18]), run_time=2)
        self.wait()



        #Bring back in various fit curves for a second, then take back out. 

        self.play(gx.FadeIn(all_variance_fits_copy), run_time=2)
        self.wait()
        self.play(gx.FadeOut(all_variance_fits_copy), run_time=2)
        self.wait()


        #Ok, now we want to bring back the testing error curve and bring in bias-variance. 
        test_error_dots[1].shift([0, 0.3, 0]) #Fudge position a little to make the breakdown more clear. 

        self.wait()
        self.remove(mean_std_eqs, bias_var_labels[:16])
        self.play(
          gx.FadeIn(degree_label),
          gx.FadeIn(error_axis_svg[1:]),
          gx.FadeIn(extended_axis_svg),
          gx.FadeIn(extended_axis_group[1][6]),
          gx.FadeIn(extended_axis_group[1][8]),
          # FadeIn(train_error_dots),
          gx.FadeIn(test_error_dots),
          self.animate_view(0, 0, 0, (1.34, 0.67, 0.0), 9.41),
          run_time=3.0)
        

        bias_var_legend=gx.SVGMobject(svg_dir+'/p65_68-06.svg')[1:]
        bias_var_legend.scale(4)
        bias_var_legend.move_to([5.5, -2.7, 0])
        self.add(bias_var_legend)
        self.wait()

        # i want to break apart the distance between zero and test_error_dots[1] on the 
        # error plot into 3 lines, one magenta line for bias, one yellow line for variance, and one 
        # green line for irreducible error. Let's make bias take up 10%, variance take up 60%, and irreducible 
        # error take up the final upper 30%. 
        # I want to move a copy of the magenta shaded bias region into becoming the first part of the error line
        # then a copy of the shaded variance region, then draw in the final irreducible error region. 
        # Don't worry about adding labels, I'll do those myself. 


        # Get the position of test_error_dots[1] (degree 2)
        dot_pos = test_error_dots[1].get_center()
        zero_pos = axes_2.c2p(2, 0)

        # Calculate the total height and the three segments
        total_height = dot_pos[1] - zero_pos[1]
        bias_height = total_height * 0.10
        variance_height = total_height * 0.60
        irreducible_height = total_height * 0.30

        # Create the three line segments
        bias_line_start = zero_pos
        bias_line_end = zero_pos + gx.UP * bias_height

        variance_line_start = bias_line_end
        variance_line_end = variance_line_start + gx.UP * variance_height

        irreducible_line_start = variance_line_end
        irreducible_line_end = irreducible_line_start + gx.UP * irreducible_height

        # Create the line objects
        bias_error_line = gx.Line(bias_line_start, bias_line_end, color='#FF00FF', stroke_width=8)
        variance_error_line = gx.Line(variance_line_start, variance_line_end, color=YELLOW, stroke_width=8)
        irreducible_error_line = gx.Line(irreducible_line_start, irreducible_line_end, color=GREEN, stroke_width=8)

        # Create copies of the shaded regions for transformation
        bias_region_copy = bias_region.copy()
        std_region_copy = std_region.copy()

        # Animate the transformation
        self.wait()
        self.play(
            gx.ReplacementTransform(bias_region_copy, bias_error_line),
            run_time=3
        )
        self.wait()
        self.play(
            gx.ReplacementTransform(std_region_copy, variance_error_line),
            run_time=3
        )
        self.wait()
        self.play(
            gx.Create(irreducible_error_line),
            run_time=3
        )


        all_first_order_fits_np=np.load(welch_path('double_descent/graphics/fits_first_order_oct_14_1.npy'))

        all_first_order_fits=gx.VGroup()
        for af in all_first_order_fits_np:
            fit_points = [axes_1.c2p(all_x[i], af[i]) for i in range(len(all_x))]
            fit_line = gx.VMobject(stroke_width=3)
            fit_line.set_points_smoothly(fit_points)
            fit_line.set_color(YELLOW)
            all_first_order_fits.add(fit_line)
        all_first_order_fits.set_stroke(width=1.0, opacity=0.4)

        self.wait()
        self.play(gx.FadeOut(bias_var_labels[18:24]),
                  gx.FadeOut(bias_var_labels[24:]),
                  gx.FadeOut(bias_var_labels[16:18]),
                  gx.FadeOut(bias_region),
                  gx.FadeOut(std_region),
                  gx.FadeOut(mean_fit_line),
                  run_time=2.0)
        self.wait()

        self.play(gx.Create(all_first_order_fits), run_time=2.5)
        self.wait()


        # Calculate mean and standard deviation for first order fits
        mean_fit_1 = np.mean(all_first_order_fits_np, 0)
        std_fit_1 = np.std(all_first_order_fits_np, 0)

        # Create the mean fit line
        fit_points_1 = [axes_1.c2p(all_x[i], mean_fit_1[i]) for i in range(len(all_x))]
        mean_fit_line_1 = gx.VMobject(stroke_width=3)
        mean_fit_line_1.set_points_smoothly(fit_points_1)
        mean_fit_line_1.set_color(YELLOW)
        mean_fit_line_1.set_stroke(width=3.0, opacity=0.9)

        # Create the standard deviation region
        upper_bound_1 = mean_fit_1 + std_fit_1
        lower_bound_1 = mean_fit_1 - std_fit_1

        upper_points_1 = [axes_1.c2p(all_x[i], upper_bound_1[i]) for i in range(len(all_x))]
        lower_points_1 = [axes_1.c2p(all_x[i], lower_bound_1[i]) for i in range(len(all_x)-1, -1, -1)]

        std_region_points_1 = upper_points_1 + lower_points_1

        std_region_1 = gx.VMobject()
        std_region_1.set_points_as_corners(std_region_points_1 + [upper_points_1[0]])
        std_region_1.set_fill(YELLOW, opacity=0.2)
        std_region_1.set_stroke(width=0)

        # Animate the transformation
        self.play(
            *[gx.ReplacementTransform(all_first_order_fits[i], mean_fit_line_1) for i in range(len(all_first_order_fits))],
            gx.FadeIn(std_region_1),
            run_time=5
        )
        self.bring_to_back(std_region_1)
        self.wait()

        #Ok claude, now, as we did above with the second order fit, I want to shade the are between the average fit and 
        # target parabola magenta. 
        # Create the bias region between first-order mean fit and parabola
        parabola_y = f(all_x)

        # Create points for both curves
        parabola_points = [axes_1.c2p(all_x[i], parabola_y[i]) for i in range(len(all_x))]
        mean_fit_1_points = [axes_1.c2p(all_x[i], mean_fit_1[i]) for i in range(len(all_x))]

        # Create the region by going along one curve and back along the other
        bias_region_1_points = parabola_points + mean_fit_1_points[::-1]

        # Create the shaded region
        bias_region_1 = gx.VMobject()
        bias_region_1.set_points_as_corners(bias_region_1_points + [parabola_points[0]])  # Close the shape
        bias_region_1.set_fill('#FF00FF', opacity=0.5)
        bias_region_1.set_stroke(width=0)


        # Animate it appearing
        self.wait()
        self.play(gx.FadeIn(bias_region_1), run_time=2)
        # self.bring_to_back(bias_region_1)
        self.wait()

        first_order_bias_variance_labels=gx.SVGMobject(svg_dir+'/p65_68-08.svg')[1:]
        first_order_bias_variance_labels.scale(5)
        first_order_bias_variance_labels.move_to([-2.6, -0.4, 0])
        self.add(first_order_bias_variance_labels)
        self.wait()

        #Ok Claude, now I want to animate my bias and variance regions coming over again to the error plot
        # this time for the first order fit. Now I want bias to be 47% of the height, variance to be 47% of 
        # the height, and irreducible error to be 6%. 

        # Get the position of test_error_dots[0] (degree 1)
        dot_pos_1 = test_error_dots[0].get_center()
        zero_pos_1 = axes_2.c2p(1, 0)

        # Calculate the total height and the three segments for first order
        total_height_1 = dot_pos_1[1] - zero_pos_1[1]
        bias_height_1 = total_height_1 * 0.47
        variance_height_1 = total_height_1 * 0.47
        irreducible_height_1 = total_height_1 * 0.06

        # Create the three line segments for first order
        bias_line_start_1 = zero_pos_1
        bias_line_end_1 = zero_pos_1 + gx.UP * bias_height_1

        variance_line_start_1 = bias_line_end_1
        variance_line_end_1 = variance_line_start_1 + gx.UP * variance_height_1

        irreducible_line_start_1 = variance_line_end_1
        irreducible_line_end_1 = irreducible_line_start_1 + gx.UP * irreducible_height_1

        # Create the line objects for first order
        bias_error_line_1 = gx.Line(bias_line_start_1, bias_line_end_1, color='#FF00FF', stroke_width=8)
        variance_error_line_1 = gx.Line(variance_line_start_1, variance_line_end_1, color=YELLOW, stroke_width=8)
        irreducible_error_line_1 = gx.Line(irreducible_line_start_1, irreducible_line_end_1, color=GREEN, stroke_width=8)

        # Create copies of the shaded regions for transformation
        bias_region_1_copy = bias_region_1.copy()
        std_region_1_copy = std_region_1.copy()

        # Animate the transformation
        self.wait()
        self.play(
            gx.ReplacementTransform(bias_region_1_copy, bias_error_line_1),
            run_time=3
        )
        self.wait()
        self.play(
            gx.ReplacementTransform(std_region_1_copy, variance_error_line_1),
            run_time=3
        )
        self.wait()
        self.play(
            gx.Create(irreducible_error_line_1),
            run_time=3
        )
        self.wait()

    
        #now I want to repeat the same process above but for thrid order fits
        # So first we remove exisitng shaded regions and labels
        # Then draw all fits
        # Then collapse them down into fit and standard deviation range
        # Dont' worry about adding labels, i can do this
        # Finally, we move the various sections over to the third order part of the test error plot
        # This time pleass make irreducible error 5%, bias 1% and variance 94%         

        all_first_third_fits_np=np.load(welch_path('double_descent/graphics/fits_third_order_oct_14_1.npy'))


        # Remove existing shaded regions and labels
        self.wait()
        self.play(
            gx.FadeOut(first_order_bias_variance_labels),
            gx.FadeOut(bias_region_1),
            gx.FadeOut(std_region_1),
            gx.FadeOut(mean_fit_line_1),
            run_time=2.0
        )
        self.wait()

        # Create and show all third order fits
        all_third_order_fits = gx.VGroup()
        for af in all_first_third_fits_np:
            fit_points = [axes_1.c2p(all_x[i], af[i]) for i in range(len(all_x))]
            fit_line = gx.VMobject(stroke_width=3)
            fit_line.set_points_smoothly(fit_points)
            fit_line.set_color(YELLOW)
            all_third_order_fits.add(fit_line)
        all_third_order_fits.set_stroke(width=1.0, opacity=0.4)

        self.play(gx.Create(all_third_order_fits), run_time=2.5)
        self.wait()

        # Calculate mean and standard deviation for third order fits
        mean_fit_3 = np.mean(all_first_third_fits_np, 0)
        std_fit_3 = np.std(all_first_third_fits_np, 0)

        # Create the mean fit line
        fit_points_3 = [axes_1.c2p(all_x[i], mean_fit_3[i]) for i in range(len(all_x))]
        mean_fit_line_3 = gx.VMobject(stroke_width=3)
        mean_fit_line_3.set_points_smoothly(fit_points_3)
        mean_fit_line_3.set_color(YELLOW)
        mean_fit_line_3.set_stroke(width=3.0, opacity=0.9)

        # Create the standard deviation region
        upper_bound_3 = mean_fit_3 + std_fit_3
        lower_bound_3 = mean_fit_3 - std_fit_3

        upper_points_3 = [axes_1.c2p(all_x[i], upper_bound_3[i]) for i in range(len(all_x))]
        lower_points_3 = [axes_1.c2p(all_x[i], lower_bound_3[i]) for i in range(len(all_x)-1, -1, -1)]

        std_region_points_3 = upper_points_3 + lower_points_3

        std_region_3 = gx.VMobject()
        std_region_3.set_points_as_corners(std_region_points_3 + [upper_points_3[0]])
        std_region_3.set_fill(YELLOW, opacity=0.2)
        std_region_3.set_stroke(width=0)

        # Collapse fits into mean and std region
        self.play(
            *[gx.ReplacementTransform(all_third_order_fits[i], mean_fit_line_3) for i in range(len(all_third_order_fits))],
            gx.FadeIn(std_region_3),
            run_time=5
        )
        self.bring_to_back(std_region_3)
        self.wait()

        # Create bias region between third-order mean fit and parabola
        parabola_y = f(all_x)
        parabola_points = [axes_1.c2p(all_x[i], parabola_y[i]) for i in range(len(all_x))]
        mean_fit_3_points = [axes_1.c2p(all_x[i], mean_fit_3[i]) for i in range(len(all_x))]
        bias_region_3_points = parabola_points + mean_fit_3_points[::-1]

        bias_region_3 = gx.VMobject()
        bias_region_3.set_points_as_corners(bias_region_3_points + [parabola_points[0]])
        bias_region_3.set_fill('#FF00FF', opacity=0.5)
        bias_region_3.set_stroke(width=0)

        self.wait()
        self.play(gx.FadeIn(bias_region_3), run_time=2)
        self.wait()

        # Get the position of test_error_dots[2] (degree 3)
        dot_pos_3 = test_error_dots[2].get_center()
        zero_pos_3 = axes_2.c2p(3, 0)

        # Calculate the total height and the three segments (1% bias, 94% variance, 5% irreducible)
        total_height_3 = dot_pos_3[1] - zero_pos_3[1]
        bias_height_3 = total_height_3 * 0.01
        variance_height_3 = total_height_3 * 0.94
        irreducible_height_3 = total_height_3 * 0.05

        # Create the three line segments for third order
        bias_line_start_3 = zero_pos_3
        bias_line_end_3 = zero_pos_3 + gx.UP * bias_height_3

        variance_line_start_3 = bias_line_end_3
        variance_line_end_3 = variance_line_start_3 + gx.UP * variance_height_3

        irreducible_line_start_3 = variance_line_end_3
        irreducible_line_end_3 = irreducible_line_start_3 + gx.UP * irreducible_height_3

        # Create the line objects for third order
        bias_error_line_3 = gx.Line(bias_line_start_3, bias_line_end_3, color='#FF00FF', stroke_width=8)
        variance_error_line_3 = gx.Line(variance_line_start_3, variance_line_end_3, color=YELLOW, stroke_width=8)
        irreducible_error_line_3 = gx.Line(irreducible_line_start_3, irreducible_line_end_3, color=GREEN, stroke_width=8)

        # Create copies of the shaded regions for transformation
        bias_region_3_copy = bias_region_3.copy()
        std_region_3_copy = std_region_3.copy()

        # Animate the transformation
        self.wait()
        self.play(
            gx.ReplacementTransform(bias_region_3_copy, bias_error_line_3),
            run_time=3
        )
        self.wait()
        self.play(
            gx.ReplacementTransform(std_region_3_copy, variance_error_line_3),
            run_time=3
        )
        self.wait()
        self.play(
            gx.Create(irreducible_error_line_3),
            run_time=3
        )
        self.wait()

        #Little zoom in action on error plot, then back out.
        self.play(gx.FadeOut(bias_region_3), gx.FadeOut(std_region_3), gx.FadeOut(mean_fit_line_3),
                 self.animate_view(0, 0, 0, (5.27, 0.56, 0.0), 7.35),
                 run_time=4)


        self.wait()
        self.play(self.animate_view(0, 0, 0, (1.34, 0.67, 0.0), 9.41), run_time=4)

        # now I want to briefly deomnstrate what would happen if we increase the number of data points
        # in our samples. I want to draw 30 points on our parabola, sorting them first as we did earlier
        # while these are drawn in, i want to shift the spread of bias and variance in our first order 
        # example on our error plot. I want to change the proportions to 94% bias, 1% variance, and 5% irreducible error. 
        # then I want to fade out the additional training points, and return the first order proportions back to 
        # their original values. 


        # Create 30 training dots and sort them
        random_seed = 25
        n_points_large = 30
        noise_level = 0.2

        x_large, y_large = get_noisy_data(n_points_large, noise_level, random_seed)
        x_train, y_train=x_large[:15], y_large[:15] #Eh
        x_test, y_test=x_large[15:],y_large[15:]

        train_dots_large = gx.VGroup(*[gx.Dot(axes_1.c2p(x_train[i], y_train[i]), radius=0.08) for i in range(len(x_train))])
        test_dots_large = gx.VGroup(*[ gx.Dot(axes_1.c2p(x_test[i], y_test[i]), radius=0.08) for i in range(len(x_test))])
        test_dots_large.set_color('#008080')
        train_dots_large.set_color(YELLOW)

        #Sorted dots so I can bring them in nicely. 
        dots_with_x = []
        for i, dot in enumerate(train_dots_large):
            dots_with_x.append((x_train[i], dot, 'train'))
        for i, dot in enumerate(test_dots_large):
            dots_with_x.append((x_test[i], dot, 'test'))
        dots_with_x.sort(key=lambda item: item[0])
        sorted_dots_large = [item[1] for item in dots_with_x]


        # Calculate new line positions for first order (94% bias, 1% variance, 5% irreducible)
        bias_height_1_new = total_height_1 * 0.94
        variance_height_1_new = total_height_1 * 0.01
        irreducible_height_1_new = total_height_1 * 0.05

        bias_line_end_1_new = zero_pos_1 + gx.UP * bias_height_1_new
        variance_line_start_1_new = bias_line_end_1_new
        variance_line_end_1_new = variance_line_start_1_new + gx.UP * variance_height_1_new
        irreducible_line_start_1_new = variance_line_end_1_new
        irreducible_line_end_1_new = irreducible_line_start_1_new + gx.UP * irreducible_height_1_new

        # Create new line objects
        bias_error_line_1_new = gx.Line(bias_line_start_1, bias_line_end_1_new, color='#FF00FF', stroke_width=8)
        variance_error_line_1_new = gx.Line(variance_line_start_1_new, variance_line_end_1_new, color=YELLOW, stroke_width=8)
        irreducible_error_line_1_new = gx.Line(irreducible_line_start_1_new, irreducible_line_end_1_new, color=GREEN, stroke_width=8)

        # Animate the dots appearing AND the bias/variance proportions changing simultaneously
        self.wait()
        self.play(
            gx.LaggedStart(*[gx.FadeIn(dot) for dot in sorted_dots_large], lag_ratio=0.05),
            # ReplacementTransform(bias_error_line_1, bias_error_line_1_new),
            # ReplacementTransform(variance_error_line_1, variance_error_line_1_new),
            # ReplacementTransform(irreducible_error_line_1, irreducible_error_line_1_new),
            run_time=4
        )
        self.play(
            # LaggedStart(*[FadeIn(dot) for dot in sorted_dots_large], lag_ratio=0.05),
            gx.ReplacementTransform(bias_error_line_1, bias_error_line_1_new),
            gx.ReplacementTransform(variance_error_line_1, variance_error_line_1_new),
            gx.ReplacementTransform(irreducible_error_line_1, irreducible_error_line_1_new),
            run_time=4
        )
        self.wait()

        # Fade out the additional training points and return proportions to original
        # Create new line objects with ORIGINAL proportions (47% bias, 47% variance, 6% irreducible)
        bias_error_line_1_original = gx.Line(bias_line_start_1, bias_line_end_1, color='#FF00FF', stroke_width=8)
        variance_error_line_1_original = gx.Line(variance_line_start_1, variance_line_end_1, color=YELLOW, stroke_width=8)
        irreducible_error_line_1_original = gx.Line(irreducible_line_start_1, irreducible_line_end_1, color=GREEN, stroke_width=8)

        # Fade out the additional training points and return proportions to original
        self.play(
            gx.FadeOut(train_dots_large),
            gx.FadeOut(test_dots_large),
            gx.ReplacementTransform(bias_error_line_1_new, bias_error_line_1_original),
            gx.ReplacementTransform(variance_error_line_1_new, variance_error_line_1_original),
            gx.ReplacementTransform(irreducible_error_line_1_new, irreducible_error_line_1_original),
            run_time=3
        )
        self.wait()

        # Eh tryign to scriggle my way through here
        # I don't think that showing the same exact process here is going to quite cut it
        # 

        self.play(gx.FadeIn(interp_threshold_line), gx.Write(interp_threshold_label))
        self.wait()


        #Back to of training set
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

        train_dots = gx.VGroup(*[gx.Dot(axes_1.c2p(x_train[i], y_train[i]), radius=0.08) for i in range(len(x_train))])
        test_dots = gx.VGroup(*[ gx.Dot(axes_1.c2p(x_test[i], y_test[i]), radius=0.08) for i in range(len(x_test))])
        all_dots=gx.VGroup(test_dots, train_dots)
        all_dots.set_color(YELLOW)
        test_dots.set_color('#008080')

        #Sorted dots so I can bring them in nicely. 
        dots_with_x = []
        for i, dot in enumerate(train_dots):
            dots_with_x.append((x_train[i], dot, 'train'))
        for i, dot in enumerate(test_dots):
            dots_with_x.append((x_test[i], dot, 'test'))
        dots_with_x.sort(key=lambda item: item[0])
        sorted_dots = [item[1] for item in dots_with_x]

        fit_line_4, test_error_4, train_error_4, y_train_pred_4, y_test_pred_4 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=4, color=YELLOW)

        self.wait()
        self.play(gx.Create(fit_line_4), gx.LaggedStart(*[gx.FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15), run_time=2)
        self.wait()

        self.play(gx.FadeOut(train_dots), gx.FadeOut(test_dots),
                  fit_line_4.animate.set_stroke(opacity=0.2))

        #Ok same thing, but new draw...
        random_seed=328
        n_points=10
        noise_level=0.2

        all_x = np.linspace(-2, 2, 128)
        all_y = f(all_x)

        n_train_points=int(np.floor(n_points*0.5))
        n_test_points=n_points-n_train_points
        x,y=get_noisy_data(n_points, noise_level, random_seed)
                           
        x_train, y_train=x[:n_train_points], y[:n_train_points]
        x_test, y_test=x[n_train_points:],y[n_train_points:]

        train_dots = gx.VGroup(*[gx.Dot(axes_1.c2p(x_train[i], y_train[i]), radius=0.08) for i in range(len(x_train))])
        test_dots = gx.VGroup(*[ gx.Dot(axes_1.c2p(x_test[i], y_test[i]), radius=0.08) for i in range(len(x_test))])
        all_dots=gx.VGroup(test_dots, train_dots)
        all_dots.set_color(YELLOW)
        test_dots.set_color('#008080')

        #Sorted dots so I can bring them in nicely. 
        dots_with_x = []
        for i, dot in enumerate(train_dots):
            dots_with_x.append((x_train[i], dot, 'train'))
        for i, dot in enumerate(test_dots):
            dots_with_x.append((x_test[i], dot, 'test'))
        dots_with_x.sort(key=lambda item: item[0])
        sorted_dots = [item[1] for item in dots_with_x]

        fit_line_4b, test_error_4, train_error_4, y_train_pred_4, y_test_pred_4 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=4, color=YELLOW)

        self.wait()
        self.play(gx.Create(fit_line_4b), gx.LaggedStart(*[gx.FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15), run_time=2)
        self.wait()

        self.play(gx.FadeOut(train_dots), gx.FadeOut(test_dots),
                  fit_line_4b.animate.set_stroke(opacity=0.2))
        random_seed=228
        n_points=10
        noise_level=0.2

        all_x = np.linspace(-2, 2, 128)
        all_y = f(all_x)

        n_train_points=int(np.floor(n_points*0.5))
        n_test_points=n_points-n_train_points
        x,y=get_noisy_data(n_points, noise_level, random_seed)
                           
        x_train, y_train=x[:n_train_points], y[:n_train_points]
        x_test, y_test=x[n_train_points:],y[n_train_points:]

        train_dots = gx.VGroup(*[gx.Dot(axes_1.c2p(x_train[i], y_train[i]), radius=0.08) for i in range(len(x_train))])
        test_dots = gx.VGroup(*[ gx.Dot(axes_1.c2p(x_test[i], y_test[i]), radius=0.08) for i in range(len(x_test))])
        all_dots=gx.VGroup(test_dots, train_dots)
        all_dots.set_color(YELLOW)
        test_dots.set_color('#008080')

        #Sorted dots so I can bring them in nicely. 
        dots_with_x = []
        for i, dot in enumerate(train_dots):
            dots_with_x.append((x_train[i], dot, 'train'))
        for i, dot in enumerate(test_dots):
            dots_with_x.append((x_test[i], dot, 'test'))
        dots_with_x.sort(key=lambda item: item[0])
        sorted_dots = [item[1] for item in dots_with_x]

        fit_line_4c, test_error_4, train_error_4, y_train_pred_4, y_test_pred_4 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=4, color=YELLOW)

        self.wait()
        self.play(gx.Create(fit_line_4c), gx.LaggedStart(*[gx.FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15), run_time=2)
        self.wait()


        # Ok now I want to repeat the same process above but for fourth order fits
        # First we ned to draw all fits
        # Then collapse them down into fit and standard deviation range
        # Dont' worry about adding labels, i can do this
        # Finally, we move the various sections over to the third order part of the test error plot
        # This time pleass make irreducible error 3%, bias 1% and variance 96%         

        all_fourth_fits_np=np.load(welch_path('double_descent/graphics/fits_fourth_order_oct_14_1.npy'))


        # Create and show all fourth order fits
        all_fourth_order_fits = gx.VGroup()
        for af in all_fourth_fits_np:  # Note: variable name suggests this is actually fourth order data
            fit_points = [axes_1.c2p(all_x[i], af[i]) for i in range(len(all_x))]
            fit_line = gx.VMobject(stroke_width=3)
            fit_line.set_points_smoothly(fit_points)
            fit_line.set_color(YELLOW)
            all_fourth_order_fits.add(fit_line)
        all_fourth_order_fits.set_stroke(width=1.0, opacity=0.4)

        self.wait()
        self.play(gx.Create(all_fourth_order_fits), run_time=2.5)
        self.wait()

        # Calculate mean and standard deviation for fourth order fits
        mean_fit_4 = np.mean(all_fourth_fits_np, 0)
        std_fit_4 = np.std(all_fourth_fits_np, 0)

        # Create the mean fit line
        fit_points_4 = [axes_1.c2p(all_x[i], mean_fit_4[i]) for i in range(len(all_x))]
        mean_fit_line_4 = gx.VMobject(stroke_width=3)
        mean_fit_line_4.set_points_smoothly(fit_points_4)
        mean_fit_line_4.set_color(YELLOW)
        mean_fit_line_4.set_stroke(width=3.0, opacity=0.9)

        # Create the standard deviation region
        upper_bound_4 = mean_fit_4 + std_fit_4
        lower_bound_4 = mean_fit_4 - std_fit_4

        upper_points_4 = [axes_1.c2p(all_x[i], upper_bound_4[i]) for i in range(len(all_x))]
        lower_points_4 = [axes_1.c2p(all_x[i], lower_bound_4[i]) for i in range(len(all_x)-1, -1, -1)]

        std_region_points_4 = upper_points_4 + lower_points_4

        std_region_4 = gx.VMobject()
        std_region_4.set_points_as_corners(std_region_points_4 + [upper_points_4[0]])
        std_region_4.set_fill(YELLOW, opacity=0.2)
        std_region_4.set_stroke(width=0)

        # Collapse fits into mean and std region
        self.play(
            *[gx.ReplacementTransform(all_fourth_order_fits[i], mean_fit_line_4) for i in range(len(all_fourth_order_fits))],
            gx.FadeIn(std_region_4),
            gx.FadeOut(train_dots),
            gx.FadeOut(test_dots),
            gx.FadeOut(fit_line_4c),
            gx.FadeOut(fit_line_4b),
            gx.FadeOut(fit_line_4),
            run_time=5
        )
        self.bring_to_back(std_region_4)
        self.wait()

        # # Get the position of test_error_dots[3] (degree 4)
        dot_pos_4 = test_error_dots[3].get_center()
        zero_pos_4 = axes_2.c2p(4, 0)

        # Calculate the total height and the three segments (1% bias, 96% variance, 3% irreducible)
        total_height_4 = dot_pos_4[1] - zero_pos_4[1]
        bias_height_4 = total_height_4 * 0.01
        variance_height_4 = total_height_4 * 0.96
        irreducible_height_4 = total_height_4 * 0.03

        # Create the three line segments for fourth order
        bias_line_start_4 = zero_pos_4
        bias_line_end_4 = zero_pos_4 + gx.UP * bias_height_4

        variance_line_start_4 = bias_line_end_4
        variance_line_end_4 = variance_line_start_4 + gx.UP * variance_height_4

        irreducible_line_start_4 = variance_line_end_4
        irreducible_line_end_4 = irreducible_line_start_4 + gx.UP * irreducible_height_4

        bias_error_line_4 = gx.Line(bias_line_start_4, bias_line_end_4, color='#FF00FF', stroke_width=8)
        variance_error_line_4 = gx.Line(variance_line_start_4, variance_line_end_4, color=YELLOW, stroke_width=8)
        irreducible_error_line_4 = gx.Line(irreducible_line_start_4, irreducible_line_end_4, color=GREEN, stroke_width=8)

        self.wait()
        self.remove(interp_threshold_line, interp_threshold_label)
        self.play(gx.FadeOut(std_region_4),
                  gx.FadeOut(mean_fit_line_4),
                  gx.Create(bias_error_line_4),
                  gx.Create(variance_error_line_4),
                  gx.Create(irreducible_error_line_4))


        # Ok claude, last section! Now I want to like 5 different 5th order fits
        # I want to be able to pick the 5 ones i show manually so i can adjust if I need to
        # I also want to show one final bias variance irreducible error breakdown for the 5th order fit. 
        # This time pleass make irreducible error 5%, bias 1% and variance 94%
        # Let's try animating in the error bars and the fits at the same time.    
        all_fifth_order_fits_np=np.load(welch_path('double_descent/graphics/fits_fifth_order_oct_14_1.npy'))

        # Select 5 specific fifth order fits (you can adjust these indices)
        selected_indices = [10, 25, 40, 55, 5] #, 2] #,Pick some fits

        # Create the 5 selected fifth order fits
        selected_fifth_order_fits = gx.VGroup()
        for idx in selected_indices:
            fit_points = [axes_1.c2p(all_x[i], all_fifth_order_fits_np[idx][i]) for i in range(len(all_x))]
            fit_line = gx.VMobject(stroke_width=3)
            fit_line.set_points_smoothly(fit_points)
            fit_line.set_color(YELLOW)
            selected_fifth_order_fits.add(fit_line)
        selected_fifth_order_fits.set_stroke(width=2.0, opacity=0.6)

        # Get the position of test_error_dots[4] (degree 5)
        dot_pos_5 = test_error_dots[4].get_center()
        zero_pos_5 = axes_2.c2p(5, 0)

        # Calculate the total height and the three segments (1% bias, 94% variance, 5% irreducible)
        total_height_5 = dot_pos_5[1] - zero_pos_5[1]
        bias_height_5 = total_height_5 * 0.01
        variance_height_5 = total_height_5 * 0.94
        irreducible_height_5 = total_height_5 * 0.05

        # Create the three line segments for fifth order
        bias_line_start_5 = zero_pos_5
        bias_line_end_5 = zero_pos_5 + gx.UP * bias_height_5

        variance_line_start_5 = bias_line_end_5
        variance_line_end_5 = variance_line_start_5 + gx.UP * variance_height_5

        irreducible_line_start_5 = variance_line_end_5
        irreducible_line_end_5 = irreducible_line_start_5 + gx.UP * irreducible_height_5

        # Create the line objects for fifth order
        bias_error_line_5 = gx.Line(bias_line_start_5, bias_line_end_5, color='#FF00FF', stroke_width=8)
        variance_error_line_5 = gx.Line(variance_line_start_5, variance_line_end_5, color=YELLOW, stroke_width=8)
        irreducible_error_line_5 = gx.Line(irreducible_line_start_5, irreducible_line_end_5, color=GREEN, stroke_width=8)

        # Animate fits and error bars simultaneously
        self.wait()
        self.play(
            gx.LaggedStart(*[gx.Create(selected_fifth_order_fits[i]) for i in range(len(selected_fifth_order_fits))], lag_ratio=0.2),
            gx.Create(bias_error_line_5),
            gx.Create(variance_error_line_5),
            gx.Create(irreducible_error_line_5),
            run_time=4
        )
        self.remove(test_error_dots); self.add(test_error_dots)
        self.wait()

        self.play(gx.FadeOut(selected_fifth_order_fits))
        self.wait()

        self.play(self.animate_view(0, 0, 0, (5.42, 0.77, 0.0), 8.36),
                 gx.FadeOut(parabola_copy),
                 gx.FadeOut(curve_fit_axis_svg),
                 run_time=2)
        self.wait()


        # self.remove(parabola_copy, curve_fit_axis_svg)

        self.wait(20)
        # Scene authoring stops here in the interactive editor.
