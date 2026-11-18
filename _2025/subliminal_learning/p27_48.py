import manimgx as gx
import numpy as np
from subliminal_assets import scene_asset
from welch_geometry import diagram_arrow
from welch_scene import WelchScene, image_at_height
from welch_text import markup_text, math_tex


CHILL_BROWN = "#958a7a"
YELLOW = "#ffd35a"
YELLOW_FADE = "#7f6a2d"
BLUE = "#65c8d0"
GREEN = "#6e9671"
CHILL_GREEN = "#6c946f"
CHILL_BLUE = "#3d5c6f"
FRESH_TAN = "#dfd0b9"
COOL_GREEN = "#00a14b"
CYAN = "#00FFFF"

def create_right_angle_symbol(vertex, side1_end, side2_end, size=0.15):
    """
    Create a right angle symbol at vertex between two sides
    vertex: the corner point where the right angle is
    side1_end: end point of first side
    side2_end: end point of second side
    size: size of the right angle symbol
    """
    # Get unit vectors for both sides
    vec1 = side1_end - vertex
    vec2 = side2_end - vertex
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    
    # Create the right angle square
    corner1 = vertex + size * unit_vec1
    corner2 = vertex + size * unit_vec1 + size * unit_vec2
    corner3 = vertex + size * unit_vec2
    
    # Create lines for the right angle symbol
    line1 = gx.Line(corner1, corner2, stroke_width=2, color=CYAN)
    line2 = gx.Line(corner2, corner3, stroke_width=2, color=CYAN)
    
    return gx.VGroup(line1, line2)

class P27_47c(WelchScene):
    def construct(self):
        # p27_to_manim_1 = SVGMobject("p27_to_manim-01.svg")[1:].scale(4)
        p27_to_manim_2 = gx.SVGMobject(scene_asset('p27_to_manim-02.svg'))[1:].scale(4)
        p27_to_manim_3 = gx.SVGMobject(scene_asset('p27_to_manim-03.svg'))[1:].scale(4)
        # p27_to_manim_4 = SVGMobject("p27_to_manim-04.svg")[1:].scale(4)
        # p27_to_manim_5 = SVGMobject("p27_to_manim-05.svg")[1:].scale(4)
        p27_to_manim_6 = gx.SVGMobject(scene_asset('p27_to_manim-06.svg'))[1:].scale(4)
        # p27_to_manim_7 = SVGMobject("p27_to_manim-07.svg")[1:].scale(4)
        p28_to_manim_4 = gx.SVGMobject(scene_asset('p28_to_manim-04.svg'))[1:].scale(4)
        p29_to_32_to_manim_1 = gx.SVGMobject(scene_asset('p29_to_32_to_manim-01.svg'))[1:].scale(4)

        mnist_network = image_at_height(scene_asset('mnist_network.png'))

        n2_1 = p27_to_manim_2.submobjects[10]
        n3_2 = p27_to_manim_2.submobjects[9]
        n1_1 = p27_to_manim_2.submobjects[11]
        n1_2 = p27_to_manim_2.submobjects[12]
        n2_2 = p27_to_manim_2.submobjects[13]
        n3_1 = p27_to_manim_2.submobjects[8]

        n3_1_arrow = p27_to_manim_2.submobjects[14]
        n3_2_arrow = p27_to_manim_2.submobjects[15]

        n2_1_graph = p27_to_manim_2.submobjects[16]
        n2_2_graph = p27_to_manim_2.submobjects[21]

        n2_1_text = gx.VGroup(
            p27_to_manim_2.submobjects[17],
            p27_to_manim_2.submobjects[18],
            p27_to_manim_2.submobjects[19],
            p27_to_manim_2.submobjects[20],
        )
        n2_2_text = gx.VGroup(
            p27_to_manim_2.submobjects[22],
            p27_to_manim_2.submobjects[23],
            p27_to_manim_2.submobjects[24],
            p27_to_manim_2.submobjects[25],
        )

        left_dimension = gx.VGroup(
            p27_to_manim_3[0], p27_to_manim_3[1], p27_to_manim_3[2]
        ).next_to(gx.VGroup(n1_1, n1_2), gx.LEFT)
        top_dimension = gx.VGroup(
            p27_to_manim_3[3], p27_to_manim_3[4], p27_to_manim_3[5]
        ).next_to(gx.VGroup(n2_1, n3_1), gx.UP)

        x1 = math_tex(r"x_1").set_color(FRESH_TAN).move_to(n1_1.get_center())
        x2 = math_tex(r"x_2").set_color(FRESH_TAN).move_to(n1_2.get_center())

        model_parameters = (
            math_tex(
                r"\theta = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
            )
            .set_color(FRESH_TAN)
            .scale(0.75)
            .next_to(n2_2, gx.DOWN, buff=0.4)
        )

        model_parameters_text = (
            gx.Text("Model Parameters*", font="Myriad Pro", font_size=24, weight=gx.BOLD)
            .next_to(model_parameters, gx.DOWN)
            .set_color(CHILL_BROWN)
        )

        def edge_point(c1, c2):
            c1_center = c1.get_center()
            c2_center = c2.get_center()
            direction = gx.normalize(c2_center - c1_center)

            # ManimGL compatible approach - use get_corner instead of get_critical_point
            ul = c1.get_corner(gx.UP + gx.LEFT)
            ur = c1.get_corner(gx.UP + gx.RIGHT)
            dl = c1.get_corner(gx.DOWN + gx.LEFT)
            dr = c1.get_corner(gx.DOWN + gx.RIGHT)

            width = np.linalg.norm(ur - ul)
            height = np.linalg.norm(ul - dl)
            radius = max(width, height) / 2

            return c1_center + radius * direction

        def split_line(line, break_width, split_ratio):
            start = line.get_start()
            end = line.get_end()
            direction = (end - start) / line.get_length()

            split_point = start + direction * line.get_length() * split_ratio

            half_gap = direction * (break_width / 2)

            line1 = gx.Line(start, split_point - half_gap)
            line2 = gx.Line(split_point + half_gap, end)

            return line1, line2, split_point

        def animate_split_line(line, split_line_1, split_line_2, split_ratio):
            line_part_1, line_part_2, _ = split_line(line, 0, split_ratio)
            line_part_1.set_color(split_line_1.get_color())
            line_part_2.set_color(split_line_2.get_color())
            self.remove(line)
            return gx.AnimationGroup(
                gx.ReplacementTransform(line_part_1, split_line_1),
                gx.ReplacementTransform(line_part_2, split_line_2),
            )

        connections = gx.VGroup(
            gx.Line(edge_point(n1_1, n2_1), edge_point(n2_1, n1_1), color=FRESH_TAN),
            gx.Line(edge_point(n1_1, n2_2), edge_point(n2_2, n1_1), color=FRESH_TAN),
            gx.Line(edge_point(n1_2, n2_1), edge_point(n2_1, n1_2), color=FRESH_TAN),
            gx.Line(edge_point(n1_2, n2_2), edge_point(n2_2, n1_2), color=FRESH_TAN),
            gx.Line(edge_point(n2_1, n3_1), edge_point(n3_1, n2_1), color=FRESH_TAN),
            gx.Line(edge_point(n2_1, n3_2), edge_point(n3_2, n2_1), color=CHILL_BROWN),
            gx.Line(edge_point(n2_2, n3_1), edge_point(n3_1, n2_2), color=FRESH_TAN),
            gx.Line(edge_point(n2_2, n3_2), edge_point(n3_2, n2_2), color=CHILL_BROWN),
        )

        ln1_1_n2_1 = connections[0]
        ln1_1_n2_2 = connections[1]
        ln1_2_n2_1 = connections[2]
        ln1_2_n2_2 = connections[3]
        ln2_1_n3_1 = connections[4]
        ln2_1_n3_2 = connections[5]
        ln2_2_n3_1 = connections[6]
        ln2_2_n3_2 = connections[7]

        theta_1 = math_tex(r"\theta_1").scale(0.6).set_color(FRESH_TAN)
        theta_2 = math_tex(r"\theta_2").scale(0.6).set_color(FRESH_TAN)
        theta_3 = math_tex(r"\theta_3").scale(0.6).set_color(FRESH_TAN)
        theta_4 = math_tex(r"\theta_4").scale(0.6).set_color(FRESH_TAN)
        theta_5 = math_tex(r"\theta_5").scale(0.6).set_color(FRESH_TAN)
        theta_6 = math_tex(r"\theta_6").scale(0.6).set_color(FRESH_TAN)
        theta_7 = math_tex(r"\theta_7").scale(0.6).set_color(CHILL_BROWN)
        theta_8 = math_tex(r"\theta_8").scale(0.6).set_color(CHILL_BROWN)

        ln1_1_n2_1_left, ln1_1_n2_1_right, theta_1_position = split_line(
            ln1_1_n2_1, theta_1.width + 0.2, 0.5
        )
        theta_1.move_to(theta_1_position)
        ln1_1_n2_1_left.set_color(FRESH_TAN)
        ln1_1_n2_1_right.set_color(FRESH_TAN)

        ln1_2_n2_1_left, ln1_2_n2_1_right, theta_2_position = split_line(
            ln1_2_n2_1, theta_2.width + 0.2, 0.75
        )
        theta_2.move_to(theta_2_position)
        ln1_2_n2_1_left.set_color(FRESH_TAN)
        ln1_2_n2_1_right.set_color(FRESH_TAN)

        ln1_1_n2_2_left, ln1_1_n2_2_right, theta_3_position = split_line(
            ln1_1_n2_2, theta_3.width + 0.2, 0.75
        )
        theta_3.move_to(theta_3_position)
        ln1_1_n2_2_left.set_color(FRESH_TAN)
        ln1_1_n2_2_right.set_color(FRESH_TAN)

        ln1_2_n2_2_left, ln1_2_n2_2_right, theta_4_position = split_line(
            ln1_2_n2_2, theta_4.width + 0.2, 0.5
        )
        theta_4.move_to(theta_4_position)
        ln1_2_n2_2_left.set_color(FRESH_TAN)
        ln1_2_n2_2_right.set_color(FRESH_TAN)

        ln2_1_n3_1_left, ln2_1_n3_1_right, theta_5_position = split_line(
            ln2_1_n3_1, theta_5.width + 0.2, 0.5
        )
        theta_5.move_to(theta_5_position)
        ln2_1_n3_1_left.set_color(FRESH_TAN)
        ln2_1_n3_1_right.set_color(FRESH_TAN)

        ln2_2_n3_1_left, ln2_2_n3_1_right, theta_6_position = split_line(
            ln2_2_n3_1, theta_6.width + 0.2, 0.75
        )
        theta_6.move_to(theta_6_position)
        ln2_2_n3_1_left.set_color(FRESH_TAN)
        ln2_2_n3_1_right.set_color(FRESH_TAN)

        ln2_1_n3_2_left, ln2_1_n3_2_right, theta_7_position = split_line(
            ln2_1_n3_2, theta_7.width + 0.2, 0.75
        )
        theta_7.move_to(theta_7_position)
        ln2_1_n3_2_left.set_color(CHILL_BROWN)
        ln2_1_n3_2_right.set_color(CHILL_BROWN)

        ln2_2_n3_2_left, ln2_2_n3_2_right, theta_8_position = split_line(
            ln2_2_n3_2, theta_8.width + 0.2, 0.5
        )
        theta_8.move_to(theta_8_position)
        ln2_2_n3_2_left.set_color(CHILL_BROWN)
        ln2_2_n3_2_right.set_color(CHILL_BROWN)

        group1 = gx.AnimationGroup(gx.FadeIn(gx.VGroup(n1_1, n1_2, x1, x2)))

        group2 = gx.AnimationGroup(
            gx.GrowArrow(ln1_1_n2_1),
            gx.GrowArrow(ln1_1_n2_2),
            gx.GrowArrow(ln1_2_n2_1),
            gx.GrowArrow(ln1_2_n2_2),
        )

        group3 = gx.AnimationGroup(gx.FadeIn(gx.VGroup(n2_1, n2_2)))

        group4 = gx.AnimationGroup(
            gx.FadeIn(gx.VGroup(n2_1_graph, n2_2_graph)),
            gx.Write(n2_1_text),
            gx.Write(n2_2_text),
        )

        group5 = gx.AnimationGroup(
            gx.GrowArrow(ln2_1_n3_1),
            gx.GrowArrow(ln2_1_n3_2),
            gx.GrowArrow(ln2_2_n3_1),
            gx.GrowArrow(ln2_2_n3_2),
        )

        group6 = gx.AnimationGroup(
            gx.FadeIn(gx.VGroup(n3_1, n3_2)),
            gx.GrowFromEdge(n3_1_arrow, gx.LEFT),
            gx.GrowFromEdge(n3_2_arrow, gx.LEFT),
        )

        self.wait()
        self.play(
            gx.LaggedStart(
                group1,
                group2,
                group3,
                group4,
                group5,
                group6,
                lag_ratio=0.2,
                run_time=2.5,
            )
        )

        self.wait(1)

        self.add(left_dimension, top_dimension)

        self.wait(1)

        theta1_group = gx.AnimationGroup(
            animate_split_line(ln1_1_n2_1, ln1_1_n2_1_left, ln1_1_n2_1_right, 0.5),
            gx.Write(theta_1),
            lag_ratio=0.5,
        )

        theta2_group = gx.AnimationGroup(
            animate_split_line(ln1_2_n2_1, ln1_2_n2_1_left, ln1_2_n2_1_right, 0.75),
            gx.Write(theta_2),
            lag_ratio=0.5,
        )

        theta3_group = gx.AnimationGroup(
            animate_split_line(ln1_1_n2_2, ln1_1_n2_2_left, ln1_1_n2_2_right, 0.75),
            gx.Write(theta_3),
            lag_ratio=0.5,
        )

        theta4_group = gx.AnimationGroup(
            animate_split_line(ln1_2_n2_2, ln1_2_n2_2_left, ln1_2_n2_2_right, 0.5),
            gx.Write(theta_4),
            lag_ratio=0.5,
        )

        theta5_group = gx.AnimationGroup(
            animate_split_line(ln2_1_n3_1, ln2_1_n3_1_left, ln2_1_n3_1_right, 0.5),
            gx.Write(theta_5),
            lag_ratio=0.5,
        )

        theta6_group = gx.AnimationGroup(
            animate_split_line(ln2_2_n3_1, ln2_2_n3_1_left, ln2_2_n3_1_right, 0.75),
            gx.Write(theta_6),
            lag_ratio=0.5,
        )

        theta7_group = gx.AnimationGroup(
            animate_split_line(ln2_1_n3_2, ln2_1_n3_2_left, ln2_1_n3_2_right, 0.75),
            gx.Write(theta_7),
            lag_ratio=0.5,
        )

        theta8_group = gx.AnimationGroup(
            animate_split_line(ln2_2_n3_2, ln2_2_n3_2_left, ln2_2_n3_2_right, 0.5),
            gx.Write(theta_8),
            lag_ratio=0.5,
        )

        self.play(
            gx.LaggedStart(
                theta1_group,
                theta2_group,
                theta3_group,
                theta4_group,
                theta5_group,
                theta6_group,
                theta7_group,
                theta8_group,
                lag_ratio=0.3,
                run_time=5,
            )
        )

        self.wait(1)

        model_parameters_1 = gx.AnimationGroup(gx.FadeIn(model_parameters[0:3]))

        model_parameters_2 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_1.copy(), model_parameters[3:5]),
            gx.FadeIn(model_parameters[5]),
        )

        model_parameters_3 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_2.copy(), model_parameters[6:8]),
            gx.FadeIn(model_parameters[8]),
        )

        model_parameters_4 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_3.copy(), model_parameters[9:11]),
            gx.FadeIn(model_parameters[11]),
        )

        model_parameters_5 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_4.copy(), model_parameters[12:14]),
            gx.FadeIn(model_parameters[14]),
        )

        model_parameters_6 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_5.copy(), model_parameters[15:17]),
            gx.FadeIn(model_parameters[17]),
        )

        model_parameters_7 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_6.copy(), model_parameters[18:20]),
            gx.FadeIn(model_parameters[20]),
        )

        model_parameters_8 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_7.copy(), model_parameters[21:23]),
            gx.FadeIn(model_parameters[23]),
        )

        model_parameters_9 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_8.copy(), model_parameters[24:26]),
            gx.FadeIn(model_parameters[26]),
        )

        self.play(
            gx.LaggedStart(
                model_parameters_1,
                model_parameters_2,
                model_parameters_3,
                model_parameters_4,
                model_parameters_5,
                model_parameters_6,
                model_parameters_7,
                model_parameters_8,
                model_parameters_9,
                lag_ratio=0.3,
                run_time=5,
            )
        )

        self.wait(1)

        self.camera.frame.save_state()

        self.play(gx.FadeIn(model_parameters_text))

        self.wait(1)

        self.play(
            self.camera.frame.animate.move_to(
                [3, 0, 0]
            ),  # TODO: Update camera position
            gx.FadeOut(top_dimension),
            gx.FadeOut(left_dimension),
            gx.FadeOut(model_parameters_text),
            run_time=2,
        )

        self.wait(1)

        p27_to_manim_6[29].next_to(n3_1_arrow, gx.RIGHT, buff=0.1)
        p27_to_manim_6[28].next_to(n3_2_arrow, gx.RIGHT, buff=0.1)

        p27_to_manim_6[0:13].next_to(p27_to_manim_6[29], gx.DOWN, buff=0.1)
        p27_to_manim_6[13:28].next_to(p27_to_manim_6[28], gx.DOWN, buff=0.1)

        primary_output = gx.VGroup(p27_to_manim_6[29], p27_to_manim_6[0:13])
        auxiliary_output = gx.VGroup(p27_to_manim_6[28], p27_to_manim_6[13:28])

        outputs = gx.VGroup(primary_output, auxiliary_output)

        """self.play(FadeIn(primary_output), FadeIn(auxiliary_output), model_parameters.animate.move_to([ 2.81398236e-01, -1.57497560e+00, -2.99721055e-17]))

        self.wait(1)"""

        # p28_to_manim_4.move_to([(primary_output.get_x() + p28_to_manim_4.width/2) - (p28_to_manim_4.submobjects[1].width/2), 0, 0])

        p28_to_manim_4.move_to(
            [
                (primary_output.get_x() + p28_to_manim_4.width / 2)
                - (p28_to_manim_4.submobjects[1].width / 2),
                0,
                0,
            ]
        )

        top_arrow = gx.VGroup(
            p28_to_manim_4.submobjects[0],
            p28_to_manim_4.submobjects[1],
            p28_to_manim_4.submobjects[2],
        )
        bottom_arrow = gx.VGroup(
            p28_to_manim_4.submobjects[3],
            p28_to_manim_4.submobjects[4],
            p28_to_manim_4.submobjects[5],
        )

        top_outputs = outputs.get_top()[1]
        bottom_outputs = outputs.get_bottom()[1]
        bottom_submob1 = p28_to_manim_4.submobjects[1].get_bottom()[1]
        top_submob5 = p28_to_manim_4.submobjects[5].get_top()[1]
        vertical_shift = (
            top_outputs - bottom_submob1 - top_submob5 + bottom_outputs
        ) / 2

        # Apply the shift
        p28_to_manim_4.shift([0, vertical_shift, 0])

        # MATH TIME ;(
        """
        p28_to_manim_4.width = 5.860444245515045
        was about to pull out the ti84 for this lol
        """

        mnist_network.move_to([7.325, -0.175, 0])  # TODO: Adjust position

        self.play(gx.FadeIn(mnist_network))

        self.wait(1)

        self.play(
            gx.LaggedStart(gx.FadeIn(primary_output), gx.FadeIn(top_arrow), lag_ratio=0.75)
        )

        self.wait(1)

        self.play(
            gx.LaggedStart(gx.FadeIn(auxiliary_output), gx.FadeIn(bottom_arrow), lag_ratio=0.75)
        )

        # when mnist gets updated
        # self.play(FadeIn(mnist_network))

        self.wait(1)

        # P29

        self.play(
            self.camera.frame.animate.move_to([0, 0, 0]),
            gx.FadeOut(mnist_network),
            gx.FadeOut(p28_to_manim_4),
            run_time=2,
        )

        self.wait(3)

        t_network_copy = gx.VGroup(
            # Input neurons with labels
            n1_1.copy(),
            n1_2.copy(),
            x1.copy(),
            x2.copy(),
            # Hidden layer neurons and their activation functions
            n2_1.copy(),
            n2_2.copy(),
            n2_1_graph.copy(),
            n2_2_graph.copy(),
            n2_1_text.copy(),
            n2_2_text.copy(),
            # Output layer neurons and arrows
            n3_1.copy(),
            n3_2.copy(),
            n3_1_arrow.copy(),
            n3_2_arrow.copy(),
            # Split lines between input and hidden layer
            ln1_1_n2_1_left.copy(),
            ln1_1_n2_1_right.copy(),
            ln1_2_n2_1_left.copy(),
            ln1_2_n2_1_right.copy(),
            ln1_1_n2_2_left.copy(),
            ln1_1_n2_2_right.copy(),
            ln1_2_n2_2_left.copy(),
            ln1_2_n2_2_right.copy(),
            # Split lines between hidden and output layer
            ln2_1_n3_1_left.copy(),
            ln2_1_n3_1_right.copy(),
            ln2_2_n3_1_left.copy(),
            ln2_2_n3_1_right.copy(),
            ln2_1_n3_2_left.copy(),
            ln2_1_n3_2_right.copy(),
            ln2_2_n3_2_left.copy(),
            ln2_2_n3_2_right.copy(),
            # Thetas (weights) on the connections
            theta_1.copy(),
            theta_2.copy(),
            theta_3.copy(),
            theta_4.copy(),
            theta_5.copy(),
            theta_6.copy(),
            theta_7.copy(),
            theta_8.copy(),
            model_parameters.copy(),
            auxiliary_output.copy(),
            primary_output.copy(),
        )


        # Assign individual variables for each element in the VGroup
        s_n1_1 = t_network_copy[0]
        s_n1_2 = t_network_copy[1]
        s_x1 = t_network_copy[2]
        s_x2 = t_network_copy[3]
        s_n2_1 = t_network_copy[4]
        s_n2_2 = t_network_copy[5]
        s_n2_1_graph = t_network_copy[6]
        s_n2_2_graph = t_network_copy[7]
        s_n2_1_text = t_network_copy[8]
        s_n2_2_text = t_network_copy[9]
        s_n3_1 = t_network_copy[10]
        s_n3_2 = t_network_copy[11]
        s_n3_1_arrow = t_network_copy[12]
        s_n3_2_arrow = t_network_copy[13]
        s_ln1_1_n2_1_left = t_network_copy[14]
        s_ln1_1_n2_1_right = t_network_copy[15]
        s_ln1_2_n2_1_left = t_network_copy[16]
        s_ln1_2_n2_1_right = t_network_copy[17]
        s_ln1_1_n2_2_left = t_network_copy[18]
        s_ln1_1_n2_2_right = t_network_copy[19]
        s_ln1_2_n2_2_left = t_network_copy[20]
        s_ln1_2_n2_2_right = t_network_copy[21]
        s_ln2_1_n3_1_left = t_network_copy[22]
        s_ln2_1_n3_1_right = t_network_copy[23]
        s_ln2_2_n3_1_left = t_network_copy[24]
        s_ln2_2_n3_1_right = t_network_copy[25]
        s_ln2_1_n3_2_left = t_network_copy[26]
        s_ln2_1_n3_2_right = t_network_copy[27]
        s_ln2_2_n3_2_left = t_network_copy[28]
        s_ln2_2_n3_2_right = t_network_copy[29]
        s_theta_1 = t_network_copy[30]
        s_theta_2 = t_network_copy[31]
        s_theta_3 = t_network_copy[32]
        s_theta_4 = t_network_copy[33]
        s_theta_5 = t_network_copy[34]
        s_theta_6 = t_network_copy[35]
        s_theta_7 = t_network_copy[36]
        s_theta_8 = t_network_copy[37]
        s_model_parameters = t_network_copy[38]
        s_auxiliary_output = t_network_copy[39]
        s_primary_output = t_network_copy[40]

        self.play(
            # self.camera.frame.animate.scale_to_fit_width(16).move_to([3.9, -1.4, 0]),
            self.animate_view(0.0, 0.0, 0.0, (3.73, -1.84, 0.0), 8.74), #SW little different framing
            run_time=3,  # TODO: Find best camera position
        )

        self.wait(1)

        self.play(t_network_copy.animate.shift(gx.RIGHT * 7), run_time=2.5)

        self.wait(1)

        t_model_parameters = (
            math_tex(
                r"\theta_T = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
            )
            .set_color(FRESH_TAN)
            .scale(0.75)
            .next_to(gx.VGroup(n1_1, auxiliary_output), gx.DOWN, buff=0.4)
        )

        t_f_t = (
            gx.VGroup(
                p29_to_32_to_manim_1.submobjects[76],
                p29_to_32_to_manim_1.submobjects[183],
            )
            .set_color(FRESH_TAN)
            .move_to(primary_output[0].get_center())
        )

        t_g_t = (
            gx.VGroup(
                p29_to_32_to_manim_1.submobjects[75],
                p29_to_32_to_manim_1.submobjects[184],
            )
            .set_color(CHILL_BROWN)
            .move_to(auxiliary_output[0].get_center())
        )

        def create_spaced_text(text_string, font, font_size, spacing=0.3, **kwargs):
            letters = []
            for i, letter in enumerate(text_string):
                letter_obj = gx.Text(letter, font=font, font_size=font_size, **kwargs)
                if i > 0:
                    letter_obj.next_to(letters[-1], gx.RIGHT, buff=spacing)
                letters.append(letter_obj)
            return gx.VGroup(*letters)


        teacher_text = (
            # Text("TEACHER", font="Myriad Pro", font_size=32, weight=BOLD)
            create_spaced_text("TEACHER", font="Myriad Pro", font_size=32, spacing=0.1, weight=gx.BOLD)
            .next_to(
                gx.VGroup(n1_1, primary_output, auxiliary_output, t_model_parameters),
                gx.UP,
                buff=0.5,
            )
            .set_color(CHILL_BROWN)
        )

        self.play(
            gx.LaggedStart(
                gx.AnimationGroup(
                    gx.ReplacementTransform(
                        model_parameters[1:27], t_model_parameters[2:28]
                    ),
                    gx.ReplacementTransform(model_parameters[0], t_model_parameters[0]),
                    gx.FadeIn(t_model_parameters[1]),
                    gx.ReplacementTransform(primary_output[0], t_f_t[0]),
                    gx.FadeIn(t_f_t[1]),
                    gx.ReplacementTransform(auxiliary_output[0], t_g_t[0]),
                    gx.FadeIn(t_g_t[1]),
                ),
                gx.Write(teacher_text),
                lag_ratio=0.8,
                run_time=2,
            )
        )

        self.wait(1)

        s_auxiliary_output_text = auxiliary_output.copy()[1]
        s_primary_output_text = primary_output.copy()[1]

        s_f_s = (
            gx.VGroup(
                p29_to_32_to_manim_1.submobjects[153],
                p29_to_32_to_manim_1.submobjects[185],
            )
            .next_to(s_primary_output_text, gx.UP, buff=0.1)
            .shift(gx.RIGHT * 7)
        )
        s_g_s = (
            gx.VGroup(
                p29_to_32_to_manim_1.submobjects[152],
                p29_to_32_to_manim_1.submobjects[186],
            )
            .next_to(s_auxiliary_output_text, gx.UP, buff=0.1)
            .shift(gx.RIGHT * 7)
        )

        s_model_parameters = (
            math_tex(
                r"\theta_S = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
            )
            .set_color(FRESH_TAN)
            .scale(0.75)
            .next_to(gx.VGroup(t_network_copy[1], t_network_copy[39]), gx.DOWN, buff=0.4)
        )

        # s_text = (
        #     Text("STUDENT", font="Myriad Pro", font_size=32, weight=BOLD, t2s={"letter_spacing": 10})
        #     .next_to(VGroup(t_network_copy[0], t_network_copy[40]), UP, buff=0.5)
        #     .set_color(CHILL_BROWN)
        # )


        s_text = (
            create_spaced_text("STUDENT", font="Myriad Pro", font_size=32, spacing=0.1, weight=gx.BOLD)
            .next_to(gx.VGroup(t_network_copy[0], t_network_copy[40]), gx.UP, buff=0.5)
            .set_color(CHILL_BROWN)
                )

        self.play(
            gx.LaggedStart(
                gx.AnimationGroup(
                    gx.ReplacementTransform(
                        t_network_copy[38][1:27], s_model_parameters[2:28]
                    ),
                    gx.ReplacementTransform(t_network_copy[38][0], s_model_parameters[0]),
                    gx.FadeIn(s_model_parameters[1]),
                    gx.ReplacementTransform(t_network_copy[40][0], s_f_s[0]),
                    gx.FadeIn(s_f_s[1]),
                    gx.ReplacementTransform(t_network_copy[39][0], s_g_s[0]),
                    gx.FadeIn(s_g_s[1]),
                ),
                gx.Write(s_text),
                lag_ratio=0.8,
                run_time=2,
            )
        )

        self.wait(1)

        t_weight_eq = (
            math_tex(r"\theta_T = \theta_T^0 + \Delta \theta_T")
            .move_to(t_model_parameters.get_center())
            .shift(gx.DOWN * 2.75)
        )
        t_weight_text = (  # TODO: Maybe implement
            gx.Text("Teacher weight\nupdate step", font="Myriad Pro", font_size=24)
            .next_to(t_weight_eq, gx.RIGHT, buff=0.1)
            .set_color(CHILL_BROWN)
        )

        s_weight_eq = (
            math_tex(r"\theta_S = \theta_S^0 + \Delta \theta_S")
            .move_to(s_model_parameters.get_center())
            .shift(gx.DOWN * 2.75)
        )
        s_weight_text = (  # TODO: Maybe implement
            gx.Text("Student weight\nupdate step", font="Myriad Pro", font_size=24)
            .next_to(s_weight_eq, gx.RIGHT, buff=0.1)
            .set_color(CHILL_BROWN)
        )

        equal_weights = math_tex(r"\theta_T^0 = \theta_S^0").move_to([3.5, -3, 0])
        equal_weights_text = (  # TODO: Maybe implement
            gx.Text(
                "Teacher and student\nstart with same weights",
                font="Myriad Pro",
                font_size=24,
            )
            .next_to(equal_weights, gx.RIGHT, buff=0.1)
            .set_color(CHILL_BROWN)
        )


        self.play(
            gx.ReplacementTransform(
                t_model_parameters[0:2].copy(),
                gx.VGroup(equal_weights[0], equal_weights[2]),
            ),
            gx.ReplacementTransform(
                s_model_parameters[0:2].copy(),
                gx.VGroup(equal_weights[4], equal_weights[6]),
            ),
            run_time=2,
        )

        self.wait(1)

        self.play(
            gx.LaggedStart(
                gx.AnimationGroup(
                    gx.FadeIn(equal_weights[1]),
                    gx.FadeIn(equal_weights[5]),
                ),
                gx.FadeIn(equal_weights[3]),
                lag_ratio=0.6,
                run_time=2,
            )
        )

        self.wait(1)

        self.play(
            gx.FadeIn(t_weight_eq[0:3]),
            gx.FadeIn(t_weight_eq[6:10]),
            gx.ReplacementTransform(equal_weights[0:3].copy(), t_weight_eq[3:6]),
            run_time=1.5,
        )

        self.wait(1)

        self.play(
            gx.FadeIn(s_weight_eq[0:3]),
            gx.FadeIn(s_weight_eq[6:10]),
            gx.ReplacementTransform(equal_weights[4:7].copy(), s_weight_eq[3:6]),
            run_time=1.5,
        )

        self.wait(2)

        # P33

        green_arrow = image_at_height(
            scene_asset('green_arrow.png')
        ).scale(  # TODO: If possible replace this with a SVG and make the position mathematically centered
            0.16517317500000003  # this is just a guess and check number
        )  # the svg version of this was bugged so i got sam to export a png version

        green_arrow.move_to((t_g_t.get_center() + s_g_s.get_center()) / 2)
        green_arrow.shift(gx.DOWN * 0.6)

        small_t_weight_eq = t_weight_eq.copy().scale_to_fit_height(0.4).set_color(CHILL_BROWN)
        small_s_weight_eq = s_weight_eq.copy().scale_to_fit_height(0.4).set_color(CHILL_BROWN)
        small_equal_weights = (
            equal_weights.copy().scale_to_fit_height(0.4).set_color(CHILL_BROWN)
        )

        

        small_equations = gx.VGroup(
            small_equal_weights, small_t_weight_eq, small_s_weight_eq
        ).arrange(gx.DOWN)
        small_equations.move_to(
            [n1_2.get_left()[0] + small_equations.width / 2, -4.5, 0]
        )
        small_equal_weights.move_to([-2.25, -3.8, 0]) #SW Manual bump here to make these guys left aligned. 

        self.play(
            gx.ReplacementTransform(equal_weights, small_equal_weights),
            gx.ReplacementTransform(t_weight_eq, small_t_weight_eq),
            gx.ReplacementTransform(s_weight_eq, small_s_weight_eq),
            run_time=2,
        )

        self.wait(1)

        self.play(
            gx.FadeOut(t_model_parameters),
            gx.FadeOut(s_model_parameters),
            gx.FadeOut(auxiliary_output[1]),
            gx.FadeOut(t_network_copy[-2][1]),
        )

        # self.remove(small_s_weight_eq[-3:])

        self.wait(1)

        self.play(
            gx.LaggedStart(
                gx.VGroup(
                    n3_2,
                    n3_2_arrow,
                    theta_7,
                    theta_8,
                    ln2_2_n3_2_left,
                    ln2_2_n3_2_right,
                    ln2_1_n3_2_left,
                    ln2_1_n3_2_right,
                    t_g_t,
                ).animate.set_color(COOL_GREEN),
                gx.FadeIn(green_arrow),
                gx.VGroup(
                    t_network_copy[11],  # n3_2.copy()
                    t_network_copy[13],  # n3_2_arrow.copy()
                    t_network_copy[36],  # theta_7.copy()
                    t_network_copy[37],  # theta_8.copy()
                    t_network_copy[28],  # ln2_2_n3_2_left.copy()
                    t_network_copy[29],  # ln2_2_n3_2_right.copy()
                    t_network_copy[26],  # ln2_1_n3_2_left.copy()
                    t_network_copy[27],  # ln2_1_n3_2_right.copy()
                    s_g_s,
                ).animate.set_color(COOL_GREEN),
                lag_ratio=0.8,
                run_time=3.5,
            )
        )

        self.wait(1)

        squared_error = (
            math_tex(r"L_S = \frac{1}{2}(g_T - g_S)^2").scale(0.9).next_to(green_arrow, gx.DOWN)
        )

        squared_error[7:9].set_color(COOL_GREEN)
        squared_error[10:12].set_color(COOL_GREEN)

        self.play(
            gx.FadeIn(squared_error[0:7]),
            gx.FadeIn(squared_error[12:14]),
            gx.FadeIn(squared_error[9]),
            gx.ReplacementTransform(t_g_t.copy(), squared_error[7:9]),
            gx.ReplacementTransform(s_g_s.copy(), squared_error[10:12]),
            run_time=2,
        )

        self.wait(1)

        # P34
        # SW jumping in here, ah man almost missed that this was a delta but is supposed to be a nabla
        # easy mistake!

        gradient_descent_eq = (
            math_tex(r"\Delta \theta_S = - \alpha \nabla_\theta L_S")
            # .scale(0.9)
            .next_to(squared_error, gx.DOWN, buff=0.6)
        )

        gradient_descent_eq.shift([-0.2, 0, 0]) #Nudge for better horizontal aligmeent. 

        gradient_descent_eq[6:8].set_color(YELLOW)

        #SW -> Ok can i move over the Delta Theta S from the subscript equation - that would really help!
        delta_theta_s_copy=small_s_weight_eq[-3:].copy()


        self.wait()
        self.play(gx.ReplacementTransform(delta_theta_s_copy, gradient_descent_eq[:3]))
        self.play(gx.Write(gradient_descent_eq[3:]))

        self.wait(1)

        # SW Lets take a crack at adding in text & arrows and see how it feels
        p33_to_manim_2 = gx.SVGMobject(scene_asset('p33_40_to_manim-02.svg'))[1:].scale(5)

        p33_to_manim_2.move_to([6.8, -3.55, 0])

        self.play(gx.Write(p33_to_manim_2[:21])) #Label 1
        self.add(p33_to_manim_2[71]) #Arrer
        self.wait()

        self.play(gx.Write(p33_to_manim_2[58:70])) #Label 2
        self.add(p33_to_manim_2[70]) #Arrer
        self.wait()

        self.play(gx.Write(p33_to_manim_2[42:58])) #Label 3
        self.add(p33_to_manim_2[72]) #Arrer
        self.wait()

        self.play(gx.Write(p33_to_manim_2[21:42])) #Label 4
        self.add(p33_to_manim_2[73]) #Arrer
        self.wait()
        # self.play(FadeIn(p33_to_manim_2[:21]))

        # self.add(p33_to_manim_2[72])

        # self.remove(p33_to_manim_2)
        self.wait()


        # P35
        funky_arrow_1 = gx.SVGMobject(scene_asset('p33_40_to_manim-03.svg'))[1:].scale(5)
        funky_arrow_1.move_to([7.4, -4, 0])
        gradient_vector = (
            math_tex(
                (
                    r"\left[ \frac{\partial L_S}{\partial \theta_1},"
                    r"\frac{\partial L_S}{\partial \theta_2},"
                    r"\cdots,"
                    r"\frac{\partial L_S}{\partial \theta_8} \right]"
                )
            )
            .set_color(YELLOW)
            .scale(0.8)
            .next_to(gradient_descent_eq, gx.DOWN, buff=0.7)
        )
        gradient_vector.shift(([0.64, 0.0, 0])) #lil nudge

        self.remove(p33_to_manim_2[:21], p33_to_manim_2[42:58], p33_to_manim_2[71], p33_to_manim_2[72])
        self.play(
            gx.LaggedStart(
                gx.AnimationGroup(
                    gx.FadeIn(gradient_vector[0]),
                    gx.FadeIn(gradient_vector[1]),
                    gx.FadeIn(gradient_vector[4]),
                    gx.FadeIn(gradient_vector[5]),
                    gx.FadeIn(gradient_vector[8]),
                    gx.ReplacementTransform(
                        squared_error[0:2].copy().copy().copy().copy(),
                        gradient_vector[2:4],
                    ),
                    gx.ReplacementTransform(s_theta_1.copy(), gradient_vector[6:8]),
                ),
                gx.AnimationGroup(
                    gx.FadeIn(gradient_vector[9]),
                    gx.FadeIn(gradient_vector[12]),
                    gx.FadeIn(gradient_vector[13]),
                    gx.FadeIn(gradient_vector[16]),
                    gx.ReplacementTransform(
                        squared_error[0:2].copy().copy().copy(), gradient_vector[10:12]
                    ),
                    gx.ReplacementTransform(s_theta_2.copy(), gradient_vector[14:16]),
                    gx.FadeIn(gradient_vector[17:21]),
                ),
                gx.AnimationGroup(
                    gx.FadeIn(gradient_vector[28]),
                    gx.FadeIn(gradient_vector[21]),
                    gx.FadeIn(gradient_vector[24]),
                    gx.FadeIn(gradient_vector[25]),
                    gx.ReplacementTransform(
                        squared_error[0:2].copy().copy(), gradient_vector[22:24]
                    ),
                    gx.ReplacementTransform(s_theta_8.copy(), gradient_vector[26:28]),
                ),
                lag_ratio=0.5,
                run_time=5,
            )
        )
        
        self.add(funky_arrow_1)
        self.wait()

        #P36 Let's Go
        self.play(gx.FadeOut(gradient_vector), gx.FadeOut(funky_arrow_1), gx.FadeOut(p33_to_manim_2[58:70]),
                  gx.FadeOut(p33_to_manim_2[21:42]), gx.FadeOut(p33_to_manim_2[70]), gx.FadeOut(p33_to_manim_2[73]))
        self.wait()

        #Ok a few things at once, move grad equation to the left, add a second part, brining over needed compoenents
        #Leg me kinda work backwards here...

        # gradient_descent_eq.move_to([6.303-2, -3.425,  0]) #Animate this

        gradient_descent_eq_2 = math_tex(r" = - \alpha \nabla_\theta \bigg[ \frac{1}{2}(g_T - g_S)^2  \bigg]")
        gradient_descent_eq_2.move_to([6.303+2.3, -3.425,  0])
        # self.add(gradient_descent_eq_2[:5])
        gradient_descent_eq_2[5:].scale(0.9) #Gets kinda big otherwise?
        gradient_descent_eq_2[5:].shift([-0.2, 0, 0])
        # self.add(gradient_descent_eq_2[5:])

        gradient_descent_eq_2[3:5].set_color(YELLOW)
        gradient_descent_eq_2[10:12].set_color(COOL_GREEN)
        gradient_descent_eq_2[13:15].set_color(COOL_GREEN)

        # gradient_descent_eq_2.scale(0.9) #Too inconsistent or ok?
        
        self.wait()
        self.play(gradient_descent_eq.animate.move_to([6.303-2, -3.425,  0]))
        self.play(gx.FadeIn(gradient_descent_eq_2[0]),
                  gx.ReplacementTransform(gradient_descent_eq[4:8].copy(),gradient_descent_eq_2[1:5])) #Might want to make an explicity copy here for cleanup!
        self.play(
                # FadeIn(gradient_descent_eq_2[5]), FadeIn(gradient_descent_eq_2[-1]), 
                  gx.ReplacementTransform(squared_error[3:].copy(), gradient_descent_eq_2[6:-1]))
        self.add(gradient_descent_eq_2[5], gradient_descent_eq_2[-1]) #Add dem berkets
        self.wait()


        # Script goes kinda fast here, so I'm wondering if want to move down a copy while moving down the 2?
        # Could also obviously adjust the script. Hmm. 
        # Hmm how do I want to do this...
        # Ok yeah I think I I want to move a copy down first, let me see here...

        gradient_descent_eq_2_copy_a=gradient_descent_eq_2[:3].copy()
        gradient_descent_eq_2_copy_b=gradient_descent_eq_2[6:-1].copy()

        self.wait()
        self.add(gradient_descent_eq_2_copy_a, gradient_descent_eq_2_copy_b)
        self.play(gradient_descent_eq_2_copy_a.animate.shift([0, -1, 0]),
                    gradient_descent_eq_2_copy_b.animate.shift([0, -1, 0]))

        # gradient_descent_eq_2_copy_b[-1].next_to(gradient_descent_eq_2_copy_b[1], LEFT, buff=0.2)
        # gradient_descent_eq_2_copy_b[-1].scale(1.5)
        self.wait()
        self.play(gradient_descent_eq_2_copy_b[-1].animate.scale(1.36).next_to(gradient_descent_eq_2_copy_b[1], gx.LEFT, buff=0.2))
        self.wait()

        #Ah I don't want to draw a diagonal line but it will look nice...here we go...
        cross_out_line_1=gx.Line([ 7.8, -4.25,  0], [ 8.5, -4.85,  0])
        cross_out_line_1.set_stroke(color=YELLOW, width=4)

        self.play(gx.Create(cross_out_line_1))
        self.play(gx.FadeOut(cross_out_line_1), gx.FadeOut(gradient_descent_eq_2_copy_b[-1]),
            gx.FadeOut(gradient_descent_eq_2_copy_b[:3]))
            # gradient_descent_eq_2_copy_b[3:].animate.shift([-1.1, 0, 0]))
        self.play(gradient_descent_eq_2_copy_b[3:-1].animate.shift([-1.05, 0, 0]))
        self.wait()


        # Ok, somewhere in here I need to lost the outer negative, again following the chain rule
        # And add a little line to the VO I think!
        # Ok I think that I bring a negative in front of the gradient(annoying!) I think I'll need backets...







        gs_copy=gradient_descent_eq_2_copy_b[-4:-2].copy()
        nabla_copy=gradient_descent_eq_2[3:5].copy()
        self.add(nabla_copy, gs_copy)

        p36_fix_eq_1=math_tex(r"(-" ).scale(0.9)
        p36_fix_eq_1.next_to(gradient_descent_eq_2_copy_b[3:-1], gx.RIGHT, buff=0.1)
        p36_fix_eq_2=math_tex(r")" ).scale(0.9)
        p36_fix_eq_2.next_to(gradient_descent_eq_2_copy_b[3:-1], gx.RIGHT, buff=1.7)

        # self.add(p36_fix_eq_1, p36_fix_eq_2)

        
        # nabla_copy.next_to(gradient_descent_eq_2_copy_b[3:-1], RIGHT, buff=0.1).shift([0, -0.04, 0])
        # gs_copy.next_to(gradient_descent_eq_2_copy_b[3:-1], RIGHT, buff=0.7).shift([0.03, -0.1, 0])
        self.wait()
        self.play(nabla_copy.animate.next_to(gradient_descent_eq_2_copy_b[3:-1], gx.RIGHT, buff=0.6).shift([0, -0.04, 0]),
                  gs_copy.animate.next_to(gradient_descent_eq_2_copy_b[3:-1], gx.RIGHT, buff=1.2).shift([0.03, -0.1, 0]),
                 gx.ReplacementTransform(gradient_descent_eq_2_copy_b[-5].copy(), p36_fix_eq_1[1])
                 )
        self.add(p36_fix_eq_1, p36_fix_eq_2)
        self.wait()


        gradient_vector_2 = (
            math_tex(
                (
                    r"\left[ \frac{\partial g_S}{\partial \theta_1},"
                    r"\frac{\partial g_S}{\partial \theta_2},"
                    r"\cdots,"
                    r"\frac{\partial g_S}{\partial \theta_8} \right]"
                )
            )
            .set_color(YELLOW)
            .scale(0.8)
            .next_to(gradient_descent_eq_2_copy_b, gx.DOWN, buff=0.3)
            .shift([0.6, 0, 0])
        )

        gradient_vector_2[2:4].set_color(COOL_GREEN)
        gradient_vector_2[10:12].set_color(COOL_GREEN)
        gradient_vector_2[22:24].set_color(COOL_GREEN)


        # Ok, now how did pranav to the cool animation last time?
        # Also, let me go ahead and add the random arrow
        funky_arrow_2 = gx.SVGMobject(scene_asset('p33_40_to_manim-04.svg'))[1:].scale(5)
        funky_arrow_2.move_to([10.3, -4.95, 0])       

        self.wait()
        self.play(
            gx.LaggedStart(
                gx.AnimationGroup(
                    gx.FadeIn(gradient_vector_2[0]),
                    gx.FadeIn(gradient_vector_2[1]),
                    gx.FadeIn(gradient_vector_2[4]),
                    gx.FadeIn(gradient_vector_2[5]),
                    gx.FadeIn(gradient_vector_2[8]),
                    gx.ReplacementTransform(
                        gs_copy.copy().copy().copy().copy(),
                        gradient_vector_2[2:4],
                    ),
                    gx.ReplacementTransform(s_theta_1.copy(), gradient_vector_2[6:8]),
                ),
                gx.AnimationGroup(
                    gx.FadeIn(gradient_vector_2[9]),
                    gx.FadeIn(gradient_vector_2[12]),
                    gx.FadeIn(gradient_vector_2[13]),
                    gx.FadeIn(gradient_vector_2[16]),
                    gx.ReplacementTransform(
                        gs_copy.copy().copy().copy(), gradient_vector_2[10:12]
                    ),
                    gx.ReplacementTransform(s_theta_2.copy(), gradient_vector_2[14:16]),
                    gx.FadeIn(gradient_vector_2[17:21]),
                ),
                gx.AnimationGroup(
                    gx.FadeIn(gradient_vector_2[28]),
                    gx.FadeIn(gradient_vector_2[21]),
                    gx.FadeIn(gradient_vector_2[24]),
                    gx.FadeIn(gradient_vector_2[25]),
                    gx.ReplacementTransform(
                        gs_copy.copy().copy(), gradient_vector_2[22:24]
                    ),
                    gx.ReplacementTransform(s_theta_8.copy(), gradient_vector_2[26:28]),
                ),
                lag_ratio=0.5,
                run_time=5,
            )
        )
        self.add(funky_arrow_2)
        self.wait()

        #Ok now a smidge of cleanup that I had to add for the negative from the chain rule correction. 
        #Fade out minus signs, then adjust and fade out parenthesis I think, move over funky arrow. 
        self.play(gx.FadeOut(p36_fix_eq_1[1]),
                  gx.FadeOut(gradient_descent_eq_2_copy_a[1]))

        # self.remove(gradient_descent_eq_2_copy_a[0])
        self.play(gx.FadeOut(p36_fix_eq_1[0]),
                  gx.FadeOut(p36_fix_eq_2),
                  nabla_copy.animate.shift([-0.5, 0, 0]),
                  gs_copy.animate.shift([-0.5, 0, 0]),
                  funky_arrow_2.animate.shift([-0.5, 0, 0]), 
                  gradient_vector_2.animate.shift([-0.5, 0, 0]))
        self.wait()


        # P37 - phew p36 was no joke lol
        # Alright first a little clean up
        # I don't want to end up with a frankenstien equation going foward, 
        # So let me see if I can do some decent clean up here...

        gradient_descent_eq_3 = math_tex(r" = \alpha  (g_T - g_S) \nabla_\theta g_S")
        gradient_descent_eq_3.move_to([6.303+1.9, -3.425,  0])
        gradient_descent_eq_3[-4:-2].set_color(YELLOW)
        gradient_descent_eq_3[-2:].set_color(COOL_GREEN)
        gradient_descent_eq_3[3:5].set_color(COOL_GREEN)
        gradient_descent_eq_3[6:8].set_color(COOL_GREEN)



        self.wait()
        self.play(gx.FadeOut(funky_arrow_2), gx.FadeOut(gradient_vector_2), gx.FadeOut(gradient_descent_eq_2))
        self.play(gx.ReplacementTransform(gradient_descent_eq_2_copy_a[0], gradient_descent_eq_3[0]),
                  gx.ReplacementTransform(gradient_descent_eq_2_copy_a[2], gradient_descent_eq_3[1]),
                  gx.ReplacementTransform(gradient_descent_eq_2_copy_b[3:-1], gradient_descent_eq_3[2:9]),
                  gx.ReplacementTransform(nabla_copy, gradient_descent_eq_3[9:11]),
                  gx.ReplacementTransform(gs_copy, gradient_descent_eq_3[11:]))
        self.wait()


        # self.remove(gradient_descent_eq_2_copy_a[0])
        # self.add(gradient_descent_eq_2_copy_a)

        # Hmm maybe I can kinda do a nice lag ratio for the S to zero transformation?
        # Hmm replaceing entire equation seems a bit problematic?

        squared_error_b = (
            math_tex(r"L_S = \frac{1}{2}(g_T - g_0)^2").scale(0.9).move_to(squared_error, aligned_edge=gx.LEFT)
        )

        squared_error_b[7:9].set_color(COOL_GREEN)
        squared_error_b[10:12].set_color(COOL_GREEN)

        gradient_descent_eq_3b = math_tex(r" = \alpha  (g_T - g_0) \nabla_\theta g_0")
        gradient_descent_eq_3b.move_to(gradient_descent_eq_3, aligned_edge=gx.LEFT)
        gradient_descent_eq_3b[-4:-2].set_color(YELLOW)
        gradient_descent_eq_3b[-2:].set_color(COOL_GREEN)
        gradient_descent_eq_3b[3:5].set_color(COOL_GREEN)
        gradient_descent_eq_3b[6:8].set_color(COOL_GREEN)

        self.wait()
        self.play(gx.ReplacementTransform(squared_error, squared_error_b),
                  gx.ReplacementTransform(gradient_descent_eq_3, gradient_descent_eq_3b), lag_ratio=0.5, run_time=3)
        self.wait()

        #Ok I think that will work for changing the g0s to gS. 


        ## --- P38 --- ##
        # Okie dokie artichhokie - now I need to create teh "bottom legend" in manim -> this will take a little time
        # and that's ok -> it will persist for the rest of the video. Tempted to try both equations and text - 
        # let's see how it goes...

        #Eitheer at the same time or before I adjust the legend, I need to reduce the size of the student equations and 
        # the green arrow -> let's try same time
        # Basically all making space for math on th teacher.


        # self.remove(green_arrow)
        # self.remove(small_equal_weights)
        # self.add(small_equal_weights)

        # small_equal_weights, 
        # small_t_weight_eq, 
        # small_s_weight_eq


        #Ok, make new equations or push them around? Maybe new ones?
        t_weight_eq_2 = math_tex(r"\theta_T = \theta_T^0 + \Delta \theta_T", font_size=30).set_color(CHILL_BROWN)
        s_weight_eq_2 = math_tex(r"\theta_S = \theta_S^0 + \Delta \theta_S", font_size=30).set_color(CHILL_BROWN)
        equal_weights_eq_2 = math_tex(r"\theta_T^0 = \theta_S^0 = \theta^0", font_size=30).set_color(CHILL_BROWN)
        g_0_eq = math_tex(r"g_0=g(X, \theta^0)", font_size=30).set_color(CHILL_BROWN)

        spacing=3.7; start=-2.5
        equal_weights_eq_2.move_to([start, -5.85, 0])
        t_weight_eq_2.move_to([start+spacing*1, -5.85, 0])
        s_weight_eq_2.move_to([start+spacing*2, -5.85, 0])
        g_0_eq.move_to([start+spacing*3, -5.85, 0])

        equal_weights_text_2 = (  # TODO: Maybe implement
            markup_text(
                "Teacher and student\nstart with same weights",
                alignment="left",
                font="Myriad Pro",
                font_size=13,
            )
            .next_to(equal_weights_eq_2, gx.RIGHT, buff=0.15)
            .set_color(CHILL_BROWN)
        )

        t_weight_text_2 = (  # TODO: Maybe implement
            markup_text("Teacher weight\nupdate step", alignment="left", font="Myriad Pro", font_size=13)
            .next_to(t_weight_eq_2, gx.RIGHT, buff=0.15)
            .set_color(CHILL_BROWN)
        )

        s_weight_text_2 = (  # TODO: Maybe implement
            markup_text("Student weight\nupdate step", alignment="left", font="Myriad Pro", font_size=13)
            .next_to(s_weight_eq_2, gx.RIGHT, buff=0.15)
            .set_color(CHILL_BROWN)
        )

        g_0_eq_text = (  # TODO: Maybe implement
            markup_text("Initial auxiliary output \nbefore training", alignment="left", font="Myriad Pro", font_size=13)
            .next_to(g_0_eq, gx.RIGHT, buff=0.15)
            .set_color(CHILL_BROWN)
        )


        # self.add(equal_weights_eq_2, t_weight_eq_2, s_weight_eq_2, g_0_eq)
        # self.add(equal_weights_text_2, t_weight_text_2, s_weight_text_2, g_0_eq_text)

        #Ok pretty good, now I need a horizontal line and the big animation!
        legend_line=gx.Line([-3.7, -5.5, 0], [11.2, -5.5, 0], buff=0)
        legend_line.set_stroke(color=CHILL_BROWN, width=1)
        # self.add(legend_line)

        legend_footer_group=gx.Group(t_weight_eq_2, s_weight_eq_2, equal_weights_eq_2, g_0_eq,
                                  equal_weights_text_2, t_weight_text_2, s_weight_text_2, g_0_eq_text, legend_line)
        # self.add(legend_footer_group)

        # self.remove(equal_weights_eq_2, t_weight_eq_2, s_weight_eq_2)


        # squared_error_b.scale(0.8) 
        # squared_error_b.move_to([6.9, -1.8, 0])

        student_line_2_eq_p38=gx.VGroup(gradient_descent_eq, gradient_descent_eq_3b)
        # student_line_2_eq_p38.scale(0.8)
        # student_line_2_eq_p38.move_to([7.2, -2.6, 0])

        #Bring back in these labels
        auxiliary_output[1].set_color(COOL_GREEN)
        t_network_copy[-2][1].set_color(COOL_GREEN)

        #Now do all moves together
        self.wait()
        # self.remove(small_t_weight_eq, small_s_weight_eq, small_equal_weights)
        self.play(gx.FadeOut(green_arrow),
            squared_error_b.animate.scale(0.8).move_to([6.9, -1.8, 0]), 
            student_line_2_eq_p38.animate.scale(0.8).move_to([7.2, -2.6, 0]),
            gx.ReplacementTransform(small_equal_weights, equal_weights_eq_2[:-3]),
            gx.ReplacementTransform(small_t_weight_eq, t_weight_eq_2),
            gx.ReplacementTransform(small_s_weight_eq, s_weight_eq_2),
            )
        self.play(gx.Create(legend_line), gx.FadeIn(equal_weights_eq_2[-3:]), gx.FadeIn(g_0_eq),
                 gx.FadeIn(equal_weights_text_2), gx.FadeIn(t_weight_text_2), gx.FadeIn(s_weight_text_2), gx.FadeIn(g_0_eq_text),
                 gx.FadeIn(auxiliary_output[1]), gx.FadeIn(t_network_copy[-2][1]))
        self.wait()

        # self.add(equal_weights_eq_2, t_weight_eq_2, s_weight_eq_2, g_0_eq)
        # self.add(equal_weights_text_2, t_weight_text_2, s_weight_text_2, g_0_eq_text)

        gt_question_eq=math_tex("g_T= \: ?", font_size=56).set_color(COOL_GREEN)
        gt_question_eq.move_to([0, -2.3, 0])

        self.play(gx.ReplacementTransform(t_g_t.copy(), gt_question_eq[:2]))
        self.play(gx.Write(gt_question_eq[2:]))
        self.wait()


        p33_40_to_manim_05 = gx.SVGMobject(scene_asset('p33_40_to_manim-05.svg'))[1:].scale(4)
        
        p33_40_to_manim_05.move_to([0.2, -2.25, 0])
        g_0_eq_2=math_tex("g_0", font_size=56).set_color(COOL_GREEN)
        g_0_eq_2.move_to([-0.8, -2.4, 0])


        self.remove(gt_question_eq[2:])
        self.play(gt_question_eq[:2].animate.shift([1.7, 0, 0]))
        self.play(gx.Create(p33_40_to_manim_05[:-3]), gx.Write(g_0_eq_2))
        self.wait()
        self.play(gx.ReplacementTransform(t_weight_eq_2[-3:].copy(), p33_40_to_manim_05[-3:]))
        self.wait()


        #Ok let me kinda start with the end in midn here and get finalish equation setup first. 
        taylor_expansion_1=math_tex(r"g_T \approx g_0 + \frac{\partial g_0}{\partial \theta_1} \Delta \theta_1 + ... ", font_size=50)
        taylor_expansion_1.move_to([0.1, -2.3, 0])
        taylor_expansion_1[:2].set_color(COOL_GREEN)
        taylor_expansion_1[3:5].set_color(COOL_GREEN)
        taylor_expansion_1[6:13].set_color(YELLOW)
        taylor_expansion_1[13:16].set_color(CYAN)

        # self.add(taylor_expansion_1)
        # self.remove(taylor_expansion_1)

        self.wait()
        self.play(gx.FadeOut(p33_40_to_manim_05[:-3]))
        self.play(gx.ReplacementTransform(g_0_eq_2, taylor_expansion_1[3:5]),
                  gx.ReplacementTransform(gt_question_eq[:2], taylor_expansion_1[:2]),
                  gx.ReplacementTransform(p33_40_to_manim_05[-3:], taylor_expansion_1[-7:-4]))
        self.play(gx.FadeIn(taylor_expansion_1[2]), gx.FadeIn(taylor_expansion_1[5]),
                  gx.FadeIn(taylor_expansion_1[6:13]), gx.FadeIn(taylor_expansion_1[-4:]))

        self.wait()

        # I think especially when we get to the drawin here, it will be cool to 
        # possibly bring things down from the teacher model!

        p33_40_to_manim_06 = gx.SVGMobject(scene_asset('p33_40_to_manim-06.svg'))[1:].scale(4)
        p33_40_to_manim_07 = gx.SVGMobject(scene_asset('p33_40_to_manim-07.svg'))[1:].scale(4)
        p33_40_to_manim_08 = gx.SVGMobject(scene_asset('p33_40_to_manim-08.svg'))[1:].scale(4)
        p33_40_to_manim_09 = gx.SVGMobject(scene_asset('p33_40_to_manim-09.svg'))[1:].scale(4)
        p33_40_to_manim_06.move_to([0.9, -4.2, 0])
        p33_40_to_manim_07.move_to([1.15, -4.1, 0])
        p33_40_to_manim_08.move_to([0.88, -4.9, 0])
        p33_40_to_manim_09.move_to([0.65, -3.6, 0])
        taylor_expansion_plot_1 = gx.Group(p33_40_to_manim_06, p33_40_to_manim_07, p33_40_to_manim_08, p33_40_to_manim_09)


        self.wait()
        self.play(gx.FadeIn(p33_40_to_manim_06))
        self.play(gx.Write(p33_40_to_manim_07[-1]),
            gx.ReplacementTransform(taylor_expansion_1[6:13].copy(), p33_40_to_manim_07[:-1]))
        self.wait()

        self.play(gx.FadeIn(p33_40_to_manim_08[-2:]), gx.ReplacementTransform(taylor_expansion_1[13:16].copy(), p33_40_to_manim_08[:-2]))
        self.wait()
        self.play(gx.Write(p33_40_to_manim_09))
        self.wait()


        # Ok final bit of this big ole paragraph. 
        # Alright lets see here, let me start with the end again for this last move of p38. 
        p33_40_to_manim_10 = gx.SVGMobject(scene_asset('p33_40_to_manim-10.svg'))[1:].scale(4)
        taylor_expansion_2=math_tex(r"g_T \approx g_0 + \frac{\partial g_0}{\partial \theta_1} \Delta \theta_1 + \frac{\partial g_0}{\partial \theta_2} \Delta \theta_2 + \frac{\partial g_0}{\partial \theta_3} \Delta \theta_3 +... ", font_size=36)
        taylor_expansion_2.move_to([0.1, -2.3, 0])
        taylor_expansion_2[:2].set_color(COOL_GREEN)
        taylor_expansion_2[3:5].set_color(COOL_GREEN)
        taylor_expansion_2[6:13].set_color(YELLOW)
        taylor_expansion_2[13:16].set_color(CYAN)
        taylor_expansion_2[17:24].set_color(YELLOW)
        taylor_expansion_2[24:27].set_color(CYAN)
        taylor_expansion_2[28:35].set_color(YELLOW)
        taylor_expansion_2[35:38].set_color(CYAN)

        # taylor_expansion_plot_1.scale(0.7)
        # taylor_expansion_plot_1.move_to([-1.2, -3.5, 0])
        p33_40_to_manim_10.move_to([1.55, -3.55, 0])


        #Animate then tweak positions and colorz. 
        self.wait()
        self.remove(taylor_expansion_1[-3:]) #Drop elipses
        self.play(gx.ReplacementTransform(taylor_expansion_1[:-3], taylor_expansion_2[:len(taylor_expansion_1)-3]),
                  taylor_expansion_plot_1.animate.scale(0.7).move_to([-1.2, -3.5, 0]))
        self.play(gx.FadeIn(taylor_expansion_2[len(taylor_expansion_1)-3:]), gx.FadeIn(p33_40_to_manim_10))

        self.wait()


        ## --- P39 let's go --- ## 
        taylor_expansion_3=math_tex(r"g_T \approx g_0 + \nabla_{\theta} g_0 \cdot \Delta \theta_T", font_size=42)
        taylor_expansion_3.move_to([0.1, -2.3, 0])
        taylor_expansion_3[:2].set_color(COOL_GREEN)
        taylor_expansion_3[3:5].set_color(COOL_GREEN)
        taylor_expansion_3[6:8].set_color(YELLOW)
        taylor_expansion_3[8:10].set_color(COOL_GREEN)
        taylor_expansion_3[11:].set_color(CYAN)


        # self.add(taylor_expansion_3)


        self.wait()
        self.play(gx.FadeOut(p33_40_to_manim_10), gx.FadeOut(taylor_expansion_plot_1))
        self.remove(taylor_expansion_2[16], taylor_expansion_2[27], taylor_expansion_2[38:])
        self.play(gx.ReplacementTransform(taylor_expansion_2[:6], taylor_expansion_3[:6]),
                  gx.ReplacementTransform(taylor_expansion_2[6:13], taylor_expansion_3[6:8]),
                  gx.ReplacementTransform(taylor_expansion_2[17:24], taylor_expansion_3[6:8]),
                  gx.ReplacementTransform(taylor_expansion_2[28:35], taylor_expansion_3[6:8]),
                  gx.ReplacementTransform(taylor_expansion_2[13:16], taylor_expansion_3[11:]),
                  gx.ReplacementTransform(taylor_expansion_2[24:27], taylor_expansion_3[11:]),
                  gx.ReplacementTransform(taylor_expansion_2[35:38], taylor_expansion_3[11:]),
                  gx.ReplacementTransform(taylor_expansion_2[3:5].copy(), taylor_expansion_3[8:10]),
                  run_time=2.5)
        self.add(taylor_expansion_3[10])
        self.wait()

        # self.remove(taylor_expansion_2)
        # self.add(taylor_expansion_3)
        # self.add(taylor_expansion_3[8:10])
        p33_40_to_manim_11 = gx.SVGMobject(scene_asset('p33_40_to_manim-11.svg'))[1:].scale(4.2)
        p33_40_to_manim_11.move_to([1.1, -2.7, 0])

        self.play(gx.FadeIn(p33_40_to_manim_11))
        self.wait()


        # self.remove(p33_40_to_manim_11)

        ### -- P40 -- ###
        #Make where we want to go here and then replacement transforms
        delta_theta_s_eq_3=math_tex(r'= \alpha (g_0 + \nabla_{\theta} g_0 \cdot \Delta \theta_T - g_0) \nabla_{\theta} g_0', font_size=37)
        delta_theta_s_eq_3.move_to([7.65, -3.4, 0])
        delta_theta_s_eq_3[3:5].set_color(COOL_GREEN)
        delta_theta_s_eq_3[6:8].set_color(YELLOW)
        delta_theta_s_eq_3[8:10].set_color(COOL_GREEN)
        delta_theta_s_eq_3[11:14].set_color(CYAN)
        delta_theta_s_eq_3[15:17].set_color(COOL_GREEN)
        delta_theta_s_eq_3[18:20].set_color(YELLOW)
        delta_theta_s_eq_3[20:].set_color(COOL_GREEN)


        self.wait()
        self.play(gx.ReplacementTransform(gradient_descent_eq_3b[:3].copy(), delta_theta_s_eq_3[:3]),
                  gx.ReplacementTransform(gradient_descent_eq_3b[-8:].copy(), delta_theta_s_eq_3[-8:]), run_time=2)
        self.remove(p33_40_to_manim_11)
        self.play(gx.ReplacementTransform(taylor_expansion_3[3:].copy(), delta_theta_s_eq_3[3:-8]), run_time=2)
        self.wait()

        #Ok let's cancel some shit.
        line_len=0.35
        line_center=delta_theta_s_eq_3[3:5].get_center()
        cross_out_line_2=gx.Line(line_center+np.array([-line_len/2, line_len/2, 0]), line_center+np.array([line_len/2, -line_len/2, 0]))
        cross_out_line_2.set_stroke(color=YELLOW, width=4)

        line_len=0.35
        line_center=delta_theta_s_eq_3[15:17].get_center()
        cross_out_line_3=gx.Line(line_center+np.array([-line_len/2, line_len/2, 0]), line_center+np.array([line_len/2, -line_len/2, 0]))
        cross_out_line_3.set_stroke(color=YELLOW, width=4)


        self.play(gx.Create(cross_out_line_2), gx.Create(cross_out_line_3))
        self.wait()


        delta_theta_s_eq_4=math_tex(r'= \alpha (\nabla_{\theta} g_0 \cdot \Delta \theta_T) \nabla_{\theta} g_0', font_size=37)
        delta_theta_s_eq_4.move_to([6.85, -4.22, 0])        
        delta_theta_s_eq_4[3:5].set_color(YELLOW)
        delta_theta_s_eq_4[5:7].set_color(COOL_GREEN)
        delta_theta_s_eq_4[8:11].set_color(CYAN)
        delta_theta_s_eq_4[12:14].set_color(YELLOW)
        delta_theta_s_eq_4[14:].set_color(COOL_GREEN)        


        self.wait()
        self.play(gx.ReplacementTransform(delta_theta_s_eq_3[:3].copy(), delta_theta_s_eq_4[:3]),
                  gx.ReplacementTransform(delta_theta_s_eq_3[6:14].copy(), delta_theta_s_eq_4[3:11]),
                  gx.ReplacementTransform(delta_theta_s_eq_3[17:].copy(), delta_theta_s_eq_4[11:]), run_time=2.5)
        self.wait()

        # To support the final line, i think we bring down the Delta Theta S
        # Then I think it does away showrlty. 
        delta_theta_s_term_copy=gradient_descent_eq[:3].copy()
        self.play(delta_theta_s_term_copy.animate.next_to(delta_theta_s_eq_4, gx.LEFT, buff=0.15), run_time=2)
        self.wait()
        

        # Ok P41 is a hold, with maybe some minor overlays in Premiere/AE
        # This takes use to...

        ## --- P42 --- ##
        ## ok ok ok ok so now to bring in the paramter space stuff, I need to remove a bunch of stuff and then bring it back
        ## There's a pretty prominent Delta Theta T and delta Theta S I can bring together. 
        ## Then fade a bunch of stuff the fuck out, and bring most of it back in p43. 


        central_dot_product_eq=math_tex(r"\Delta \theta_T \cdot \Delta \theta_S", font_size=42)
        central_dot_product_eq.move_to([3.8, -1.8, 0])
        central_dot_product_eq[:3].set_color(CYAN)


        self.wait()
        self.play(gx.ReplacementTransform(gradient_descent_eq[:3].copy(), central_dot_product_eq[4:]),
                  gx.ReplacementTransform(taylor_expansion_3[-3:].copy(), central_dot_product_eq[:3]),
                  gx.FadeOut(taylor_expansion_3),
                  gx.FadeOut(legend_footer_group),
                  gx.FadeOut(delta_theta_s_term_copy), #Dont' bring this one back
                  gx.FadeOut(delta_theta_s_eq_4),
                  gx.FadeOut(delta_theta_s_eq_3),
                  gx.FadeOut(cross_out_line_3),
                  gx.FadeOut(squared_error_b),
                  gx.FadeOut(cross_out_line_2),
                  gx.FadeOut(gradient_descent_eq_3b),
                  gx.FadeOut(gradient_descent_eq), run_time=2.5)
        self.add(central_dot_product_eq[3]) #Add the dot lol. 
        self.wait()


        #Ok now we need some paramter space!
        p30_48_to_manim_1 = gx.SVGMobject(scene_asset('p40_48_to_manim-01.svg'))[1:].scale(4.2)
        p30_48_to_manim_2 = gx.SVGMobject(scene_asset('p40_48_to_manim-02.svg'))[1:].scale(4.2)

        p30_48_to_manim_1.move_to([4, -4, 0])
        p30_48_to_manim_2.move_to([4.3, -3.75, 0])


        paramter_space_text = markup_text("Parameter space", alignment="left", font="Myriad Pro", font_size=18).set_color(CHILL_BROWN)
        paramter_space_text.move_to([4, -5.8, 0])


        theta_0_dot=gx.Dot(radius=0.07).set_color(CHILL_BROWN)
        theta_0_dot.move_to([3.9, -4, 0])


        teacher_arrow_angle=60 #Degrees
        teacher_arrow_len=1.5
        teacher_arrow_start=theta_0_dot.get_center()
        teacher_arrow_end=theta_0_dot.get_center()+np.array([teacher_arrow_len*np.cos(np.pi*teacher_arrow_angle/180), 
                                                            teacher_arrow_len*np.sin(np.pi*teacher_arrow_angle/180),0])
        teacher_arrow_1 = diagram_arrow(teacher_arrow_start, teacher_arrow_end, thickness=2.5, buff=0)
        teacher_arrow_1.set_color(CYAN)

        student_arrow_angle=10 #Degrees
        student_arrow_len=1.2
        student_arrow_start=theta_0_dot.get_center()
        student_arrow_end=theta_0_dot.get_center()+np.array([student_arrow_len*np.cos(np.pi*student_arrow_angle/180), 
                                                            student_arrow_len*np.sin(np.pi*student_arrow_angle/180),0])
        student_arrow_1 = diagram_arrow(student_arrow_start, student_arrow_end, thickness=2.5, buff=0)
        student_arrow_1.set_color(gx.WHITE)

        #Lerbels move down after we add those suckers
        teacher_label_1=math_tex(r"\Delta \theta_T", font_size=38)
        teacher_label_1.set_color(CYAN)
        teacher_label_1.move_to(teacher_arrow_1).shift([-0.5, 0.1, 0]) #This might get tricky as we rotate!

        student_label_1=math_tex(r"\Delta \theta_S", font_size=38)
        student_label_1.set_color(gx.WHITE)
        student_label_1.move_to(student_arrow_1).shift([0.1, -0.3, 0]) #This might get tricky as we rotate!    

        theta_0_label_1=math_tex(r"\theta^0", font_size=38)
        theta_0_label_1.set_color(CHILL_BROWN)
        theta_0_label_1.move_to(theta_0_dot).shift([-0.3, 0, 0]) #This might get tricky as we rotate!       


        geq_eq_1=math_tex(r"\geq 0", font_size=42)
        geq_eq_1.set_color(YELLOW)
        geq_eq_1.next_to(central_dot_product_eq, gx.RIGHT, buff=0.2)

        eq_0_eq_1=math_tex(r"= 0", font_size=42)
        eq_0_eq_1.set_color(YELLOW)
        eq_0_eq_1.next_to(central_dot_product_eq, gx.RIGHT, buff=0.2)

        leq_eq_1=math_tex(r"\leq 0", font_size=42)
        leq_eq_1.set_color(YELLOW)
        leq_eq_1.next_to(central_dot_product_eq, gx.RIGHT, buff=0.2)


        self.wait()
        self.play(gx.Create(p30_48_to_manim_1), gx.Write(paramter_space_text))
        self.play(gx.Create(theta_0_dot), gx.Write(theta_0_label_1))
        self.wait()

        self.play(gx.ReplacementTransform(central_dot_product_eq[:3].copy(), teacher_label_1),
                  gx.ReplacementTransform(central_dot_product_eq[4:].copy(), student_label_1),
                  gx.GrowArrow(teacher_arrow_1),
                  gx.GrowArrow(student_arrow_1),run_time=2.5)
        self.wait()

        self.add(p30_48_to_manim_2)
        self.wait()
        self.play(gx.Write(geq_eq_1))
        self.wait()


        #Ok now make vectors orthogonal. 
        self.wait()
        self.remove(p30_48_to_manim_2)
        self.play(teacher_arrow_1.animate.rotate(gx.DEGREES*40, [0, 0, 1], theta_0_dot.get_center()),
                  teacher_label_1.animate.shift([-0.45, 0, 0]), run_time=2.5)
        self.add(theta_0_dot)
        self.wait()

        self.play(gx.ReplacementTransform(geq_eq_1, eq_0_eq_1))
        self.wait()

        #Now they move in different directions
        self.wait()
        self.play(teacher_arrow_1.animate.rotate(gx.DEGREES*40, [0, 0, 1], theta_0_dot.get_center()),
                  teacher_label_1.animate.shift([-0.45, -0.3, 0]), 
                  theta_0_label_1.animate.shift([0, -0.1, 0]), run_time=2.5)
        self.add(theta_0_dot)
        self.wait()

        self.play(gx.ReplacementTransform(eq_0_eq_1, leq_eq_1))
        self.wait()        


        ## ---- P43 ----- ##
        # litte resent, bring dot product over to left side, and = ?, bring back in everything else!
        left_dot_product_eq=math_tex(r"\Delta \theta_T \cdot \Delta \theta_S = \: ?", font_size=37)
        left_dot_product_eq.move_to([-2.25, -2, 0])
        left_dot_product_eq[:3].set_color(CYAN)

        self.wait()
        self.play(gx.ReplacementTransform(central_dot_product_eq[4:7], left_dot_product_eq[4:7]),
                  gx.ReplacementTransform(central_dot_product_eq[:3], left_dot_product_eq[:3]),
                  gx.FadeOut(central_dot_product_eq[3]),
                  gx.FadeOut(leq_eq_1),
                  gx.FadeOut(p30_48_to_manim_1),
                  gx.FadeOut(paramter_space_text),
                  gx.FadeOut(theta_0_label_1),
                  gx.FadeOut(teacher_label_1),
                  gx.FadeOut(student_label_1),
                  gx.FadeOut(teacher_arrow_1),
                  gx.FadeOut(student_arrow_1),
                  gx.FadeOut(theta_0_dot),
                  gx.FadeIn(legend_footer_group),
                  gx.FadeIn(delta_theta_s_eq_4),
                  gx.FadeIn(delta_theta_s_eq_3),
                  gx.FadeIn(cross_out_line_3),
                  gx.FadeIn(squared_error_b),
                  gx.FadeIn(cross_out_line_2),
                  gx.FadeIn(gradient_descent_eq_3b),
                  gx.FadeIn(gradient_descent_eq), run_time=2.5)
        self.add(left_dot_product_eq)
        self.wait()

        left_dot_product_eq_2=math_tex(r"\Delta \theta_S \cdot \Delta \theta_T = [\alpha (\nabla_{\theta} g_0 \cdot \Delta \theta_T) \nabla_{\theta} g_0] \cdot \Delta \theta_T", font_size=37)
        left_dot_product_eq_2.move_to([-0.15, -2.77, 0])
        left_dot_product_eq_2[4:7].set_color(CYAN)
        left_dot_product_eq_2[11:13].set_color(YELLOW)
        left_dot_product_eq_2[13:15].set_color(COOL_GREEN)
        left_dot_product_eq_2[16:19].set_color(CYAN)
        left_dot_product_eq_2[20:22].set_color(YELLOW)
        left_dot_product_eq_2[22:24].set_color(COOL_GREEN)
        left_dot_product_eq_2[-3:].set_color(CYAN)


        self.wait()
        #Ok I think I have a good solution to match the VO, do the left side bring down switch action at after the rist side!

        self.play(gx.ReplacementTransform(delta_theta_s_eq_4[1:].copy(), left_dot_product_eq_2[9:-5]), run_time=2.5)
        self.add(left_dot_product_eq_2[7:8],left_dot_product_eq_2[-4],left_dot_product_eq_2[8], left_dot_product_eq_2[-5])
        self.play(gx.ReplacementTransform(left_dot_product_eq[:3].copy(), left_dot_product_eq_2[-3:]), run_time=2.5)
        self.wait()
        self.play(gx.ReplacementTransform(left_dot_product_eq[:3].copy(), left_dot_product_eq_2[4:7]), #The ole switcheroo
                  gx.ReplacementTransform(left_dot_product_eq[4:7].copy(), left_dot_product_eq_2[:3]), run_time=2.5)
        self.add(left_dot_product_eq_2[3]) #Add that dot!
        self.wait()
        

        p30_48_to_manim_4 = gx.SVGMobject(scene_asset('p40_48_to_manim-04.svg'))[1:].scale(3.6)
        p30_48_to_manim_4.move_to([0.15, -2.38, 0])


        self.play(gx.Write(p30_48_to_manim_4))
        self.wait()

        #Ok now I need my penultimate equation here...
        left_dot_product_eq_3=math_tex(r" = \alpha (\nabla_{\theta} g_0 \cdot \Delta \theta_T) (\nabla_{\theta} g_0 \cdot \Delta \theta_T)", font_size=37)
        left_dot_product_eq_3.move_to([0.8, -3.55, 0])
        left_dot_product_eq_3[3:5].set_color(YELLOW)
        left_dot_product_eq_3[5:7].set_color(COOL_GREEN)
        left_dot_product_eq_3[8:11].set_color(CYAN)
        left_dot_product_eq_3[13:15].set_color(YELLOW)
        left_dot_product_eq_3[15:17].set_color(COOL_GREEN)
        left_dot_product_eq_3[-4:-1].set_color(CYAN)


        self.wait()
        self.play(gx.ReplacementTransform(left_dot_product_eq_2[7].copy(), left_dot_product_eq_3[0]),
                  gx.ReplacementTransform(left_dot_product_eq_2[9:20].copy(), left_dot_product_eq_3[1:12]),
                  gx.ReplacementTransform(left_dot_product_eq_2[20:24].copy(), left_dot_product_eq_3[13:17]),
                  gx.ReplacementTransform(left_dot_product_eq_2[-3:].copy(), left_dot_product_eq_3[-4:-1]),
                  run_time=2.5)
        self.add(left_dot_product_eq_3)
        self.wait()

        p30_48_to_manim_4b = gx.SVGMobject(scene_asset('p40_48_to_manim-04.svg'))[1:].scale(3.5)
        p30_48_to_manim_4b.move_to([0.1, -3.17, 0])
        p30_48_to_manim_4c=p30_48_to_manim_4b.copy()
        p30_48_to_manim_4c.move_to([2.15, -3.17, 0])


        self.play(gx.Write(p30_48_to_manim_4b), gx.Write(p30_48_to_manim_4c))


        #Ok almost through p43 here...
        left_dot_product_eq_4=math_tex(r" = \alpha (\nabla_{\theta} g_0 \cdot \Delta \theta_T)^2 \geq 0", font_size=37)
        left_dot_product_eq_4.move_to([0.22, -4.3, 0])
        left_dot_product_eq_4[3:5].set_color(YELLOW)
        left_dot_product_eq_4[5:7].set_color(COOL_GREEN)
        left_dot_product_eq_4[8:11].set_color(CYAN)

        self.wait()
        self.play(gx.ReplacementTransform(left_dot_product_eq_3[:12].copy(), left_dot_product_eq_4[:12]), run_time=2.0)
        self.play(gx.ReplacementTransform(left_dot_product_eq_3[12:].copy(), left_dot_product_eq_4[2:12]), run_time=2.0)
        self.add(left_dot_product_eq_4[-3])
        self.wait()
                  

        self.play(gx.Write(left_dot_product_eq_4[-2:]))
        self.wait()

        dot_prod_eq_copy=left_dot_product_eq_2[:7].copy()
        self.add(dot_prod_eq_copy)

        self.play(dot_prod_eq_copy.animate.next_to(left_dot_product_eq_4, gx.LEFT, buff=0.15),
                 gx.FadeOut(p30_48_to_manim_4), gx.FadeOut(p30_48_to_manim_4b), gx.FadeOut(p30_48_to_manim_4c), run_time=2.0
                 )
        self.wait()

        result_box_1=gx.RoundedRectangle(width=5.7, height=0.7, corner_radius=0.1)
        result_box_1.set_stroke(color=YELLOW, width=2.5)
        result_box_1.move_to([-0.66, -4.32, 0])

        self.play(gx.Write(result_box_1))
        self.wait()

        ## --- P45 ---- ##
        ## Ok now i need to collapse down right side, bringing in a couple important elements. 

        p30_48_to_manim_1b = gx.SVGMobject(scene_asset('p40_48_to_manim-01.svg'))[1:].scale(4.2)


        p45_eq_1=math_tex(r"\Delta \theta_S = \alpha (\nabla_{\theta} g_0 \cdot \Delta \theta_T) \nabla_{\theta} g_0", font_size=28)
        p45_eq_1.set_color(CHILL_BROWN)
        p45_eq_1.move_to([7, -1.6, 0])
        p30_48_to_manim_1b.move_to([7, -3.5, 0])

        self.remove(squared_error_b, gradient_descent_eq[4:], gradient_descent_eq_3b, delta_theta_s_eq_4[0])
        self.remove(delta_theta_s_eq_3, cross_out_line_2, cross_out_line_3)
        self.play(gx.ReplacementTransform(gradient_descent_eq[:4], p45_eq_1[:4]),
                  gx.ReplacementTransform(delta_theta_s_eq_4[1:], p45_eq_1[4:]), run_time=2.5)
        self.wait()

        #Ah yeah should construct various parts bringing down bariables from equation above...

        theta_0_dot_2=gx.Dot(radius=0.07).set_color(CHILL_BROWN)
        theta_0_dot_2.move_to([7, -3.65, 0])

        theta_0_label_2=math_tex(r"\theta^0", font_size=30)
        theta_0_label_2.set_color(CHILL_BROWN)
        theta_0_label_2.move_to(theta_0_dot_2).shift([0.05, -0.25, 0]) #This might get tricky as we rotate! 

        # Ok now I need a nice gradient arrow that extends through my points, and 
        # I want to bring down the label from the equation
        gradient_vector_angle=10 #degrees
        gradient_vector_len=4 # go in both directions around theta 0
        gradient_vector_end=theta_0_dot_2.get_center()+np.array([gradient_vector_len/2*np.cos(np.pi*gradient_vector_angle/180), 
                                                             gradient_vector_len/2*np.sin(np.pi*gradient_vector_angle/180),0])
        gradient_vector_start=theta_0_dot_2.get_center()-np.array([gradient_vector_len/2*np.cos(np.pi*gradient_vector_angle/180), 
                                                             gradient_vector_len/2*np.sin(np.pi*gradient_vector_angle/180),0])
        gradient_arrow_1 = diagram_arrow(gradient_vector_start, gradient_vector_end, thickness=2.5, buff=0)
        gradient_arrow_1.set_color(YELLOW)

        teacher_arrow_angle=60 #Degrees
        teacher_arrow_len=1.5
        teacher_arrow_start=theta_0_dot_2.get_center()
        teacher_arrow_end=theta_0_dot_2.get_center()+np.array([teacher_arrow_len*np.cos(np.pi*teacher_arrow_angle/180), 
                                                            teacher_arrow_len*np.sin(np.pi*teacher_arrow_angle/180),0])
        teacher_arrow_2 = diagram_arrow(teacher_arrow_start, teacher_arrow_end, thickness=2.5, buff=0)
        teacher_arrow_2.set_color(CYAN)


        gradient_vec = gradient_vector_end - gradient_vector_start
        teacher_vec = teacher_arrow_end - teacher_arrow_start
        gradient_direction = gradient_vec / np.linalg.norm(gradient_vec)
        projection_length = np.dot(teacher_vec, gradient_direction)
        projected_vector = projection_length * gradient_direction
        projection_point = teacher_arrow_start + projected_vector


        student_arrow_start=theta_0_dot_2.get_center()
        student_arrow_end=projection_point
        student_arrow_2 = diagram_arrow(student_arrow_start, student_arrow_end, thickness=2.5, buff=0)
        student_arrow_2.set_color(gx.WHITE)


        perp_connector=gx.DashedLine(teacher_arrow_2.get_end(), projection_point, dash_length=0.05, dashed_ratio=0.5)
        perp_connector.set_stroke(color=CYAN, width=3)


        gradient_direction_point = projection_point - 0.5 * (gradient_vec / np.linalg.norm(gradient_vec))
        right_angle_symbol = create_right_angle_symbol(
            vertex=projection_point,
            side1_end=teacher_arrow_2.get_end(),  # perpendicular line
            side2_end=gradient_direction_point,   # gradient line direction
            size=0.2
        )
        right_angle_symbol.set_stroke(color=CYAN, width=3)


        delta_theta_s_copy_2=p45_eq_1[:3].copy()
        delta_theta_s_copy_2.scale(1.2)
        delta_theta_s_copy_2.set_color(gx.WHITE)
        delta_theta_s_copy_2.move_to(student_arrow_2).shift([0.15, -0.25, 0])


        delta_theta_t_copy_2=p45_eq_1[11:14].copy()
        delta_theta_t_copy_2.scale(1.2)
        delta_theta_t_copy_2.set_color(CYAN)
        delta_theta_t_copy_2.move_to(teacher_arrow_2).shift([-0.4, 0.1, 0])

        grad_copy_2=p45_eq_1[-4:].copy()
        grad_copy_2.scale(1.2)
        grad_copy_2[:2].set_color(YELLOW)
        grad_copy_2[2:].set_color(COOL_GREEN)
        grad_copy_2.move_to(gradient_arrow_1.get_end()).shift([0.4, 0., 0])


        #Ok yeah I think that getting the gradient vector and axis down first is less overwhelming
        self.wait()
        self.play(gx.Write(p30_48_to_manim_1b),
                  gx.Write(gradient_arrow_1),
                  gx.ReplacementTransform(p45_eq_1[-4:].copy(), grad_copy_2), run_time=2.5)

        self.play(gx.Write(teacher_arrow_2),
                  gx.Write(student_arrow_2),
                  gx.Write(perp_connector),
                  gx.Write(right_angle_symbol),
                  gx.Write(theta_0_dot_2),
                  gx.Write(theta_0_label_2),
                  gx.ReplacementTransform(p45_eq_1[:3].copy(), delta_theta_s_copy_2),
                  gx.ReplacementTransform(p45_eq_1[11:14].copy(), delta_theta_t_copy_2), run_time=2.5
                  )
        self.wait()


        ## --- P46 ---- ##
        ## Ok ok ok ok ok ok ok ok 
        ## I think from here I rotate/move everythign over to like 120 degrees
        ## And then back to 100 degrees (perpendicular), then back to starting point 

        ## --- P46 ---- ##
        # Animate rotation from 60 degrees to 140 degrees
        new_teacher_angle = 150  # degrees

        # Create updater functions for all dependent elements
        def update_teacher_arrow(mob, alpha):
            current_angle = gx.interpolate(teacher_arrow_angle, new_teacher_angle, alpha)
            new_end = theta_0_dot_2.get_center() + np.array([
                teacher_arrow_len * np.cos(np.pi * current_angle / 180), 
                teacher_arrow_len * np.sin(np.pi * current_angle / 180), 
                0
            ])
            mob.put_start_and_end_on(teacher_arrow_start, new_end)

        def update_projections_and_student(alpha):
            # Calculate current teacher vector
            current_angle = gx.interpolate(teacher_arrow_angle, new_teacher_angle, alpha)
            current_teacher_end = theta_0_dot_2.get_center() + np.array([
                teacher_arrow_len * np.cos(np.pi * current_angle / 180), 
                teacher_arrow_len * np.sin(np.pi * current_angle / 180), 
                0
            ])
            current_teacher_vec = current_teacher_end - teacher_arrow_start
            
            # Recalculate projection
            current_projection_length = np.dot(current_teacher_vec, gradient_direction)
            current_projected_vector = current_projection_length * gradient_direction
            current_projection_point = teacher_arrow_start + current_projected_vector
            
            # Update student arrow
            student_arrow_2.put_start_and_end_on(student_arrow_start, current_projection_point)
            
            # Recreate perpendicular connector (since DashedLine doesn't update well)
            new_perp_connector = gx.DashedLine(current_teacher_end, current_projection_point, dash_length=0.05, dashed_ratio=0.5)
            new_perp_connector.set_stroke(color=CYAN, width=3)
            perp_connector.become(new_perp_connector)
            
            # Update right angle symbol - switch direction based on angle
            if current_angle > 90:  # When teacher angle is obtuse, flip the gradient direction
                gradient_direction_point = current_projection_point + 0.5 * (gradient_vec / np.linalg.norm(gradient_vec))
            else:
                gradient_direction_point = current_projection_point - 0.5 * (gradient_vec / np.linalg.norm(gradient_vec))
                
            new_right_angle = create_right_angle_symbol(
                vertex=current_projection_point,
                side1_end=current_teacher_end,
                side2_end=gradient_direction_point,
                size=0.2
            )
            new_right_angle.set_stroke(color=CYAN, width=3)
            right_angle_symbol.become(new_right_angle)
            
            # Update label positions
            delta_theta_s_copy_2.move_to(student_arrow_2).shift([0.15, -0.25, 0])
            delta_theta_t_copy_2.move_to(current_teacher_end).shift([-0.4, 0.1, 0])

        # Run the animation
        self.play(
            gx.UpdateFromAlphaFunc(teacher_arrow_2, update_teacher_arrow),
            gx.UpdateFromAlphaFunc(gx.VGroup(), lambda mob, alpha: update_projections_and_student(alpha)),
            run_time=2
        )
        self.wait()

        # Update the teacher_arrow_angle variable for future use
        teacher_arrow_angle = new_teacher_angle

        new_teacher_angle = 100  # degrees
        self.play(
            gx.UpdateFromAlphaFunc(teacher_arrow_2, update_teacher_arrow),
            gx.UpdateFromAlphaFunc(gx.VGroup(), lambda mob, alpha: update_projections_and_student(alpha)),
            run_time=2
        )
        self.wait()
        teacher_arrow_angle = new_teacher_angle

        new_teacher_angle = 60  # degrees
        self.play(
            gx.UpdateFromAlphaFunc(teacher_arrow_2, update_teacher_arrow),
            gx.UpdateFromAlphaFunc(gx.VGroup(), lambda mob, alpha: update_projections_and_student(alpha)),
            run_time=2
        )
        self.wait()
        teacher_arrow_angle = new_teacher_angle


        #Damn claude did a nice job!
        self.wait()





        self.wait(20)
        # Scene authoring stops here in the interactive editor.

        # self.add(gradient_arrow_1


        # self.add(delta_theta_s_copy_2, delta_theta_t_copy_2, grad_copy_2)        

        # self.add(right_angle_symbol)


        # self.add(perp_connector)

        # self.add(teacher_arrow_2, student_arrow_2)
        # self.add(p30_48_to_manim_1b, theta_0_dot_2, theta_0_label_2, gradient_arrow_1)
        # self.add(theta_0_dot_2)


        # self.remove(theta_0_dot_2, theta_0_label_2, gradient_arrow_1)
        # self.remove(right_angle_symbol)
        # self.remove(delta_theta_t_copy_2)



        # #Lerbels move down after we add those suckers
        # teacher_label_1=Tex(r"\Delta \theta_T", font_size=38)
        # teacher_label_1.set_color(CYAN)
        # teacher_label_1.move_to(teacher_arrow_1).shift([-0.5, 0.1, 0]) #This might get tricky as we rotate!

        # student_label_1=Tex(r"\Delta \theta_S", font_size=38)
        # student_label_1.set_color(WHITE)
        # student_label_1.move_to(student_arrow_1).shift([0.1, -0.3, 0]) #This might get tricky as we rotate!    
      






        # self.add(p30_48_to_manim_1b)

        # self.add(p45_eq_1)

        # self.remove(p45_eq_1)


        # self.add(result_box_1)

        # self.remove(result_box_1)


        #Now draw a nice rounded rectangle



        # self.add(dot_prod_eq_copy)


        # self.remove(dot_prod_eq_copy)


        # self.remove(left_dot_product_eq_4)









        # self.add(p30_48_to_manim_4b, p30_48_to_manim_4c)
        # self.remove(p30_48_to_manim_4b, p30_48_to_manim_4c)

        # self.remove
        # self.add(left_dot_product_eq_3)
        # self.remove(left_dot_product_eq_3)

        # self.add(left_dot_product_eq_2)


        # self.add(p30_48_to_manim_4)
        # self.remove(p30_48_to_manim_4)




        # self.add(geq_eq_1)
        # self.add(p30_48_to_manim_1)
        # self.add(paramter_space_text)
        # self.add(teacher_arrow_1)
        # self.add(student_arrow_1)
        # self.add(theta_0_dot)
        # self.add(teacher_label_1)
        # self.add(student_label_1)
        # self.add(p30_48_to_manim_2)
        # self.add(theta_0_label_1)


        # self.remove(teacher_arrow_1)



        # self.add(central_dot_product_eq)
        # self.remove(central_dot_product_eq)



        # self.play(FadeOut(legend_footer_group), 
        #           FadeOut(delta_theta_s_term_copy), #Dont' bring this one back
        #           FadeOut(delta_theta_s_eq_4), 
        #           FadeOut(delta_theta_s_eq_3), 
        #           FadeOut(delta_theta_s_eq_2), 
        #           FadeOut(gradient_descent_eq))









        # self.add(delta_theta_s_term_copy)
        # self.remove(gradient_descent_eq)
        # self.add(gradient_descent_eq)



        # self.add(delta_theta_s_eq_4)
        # self.remove(delta_theta_s_eq_4)


        # self.add(delta_theta_s_eq_3)
        # self.remove(delta_theta_s_eq_3)
        # self.remove(gradient_descent_eq)
        # self.add(gradient_descent_eq)
        # self.remove(gradient_descent_eq_3b)
        # self.add(gradient_descent_eq_3b)


        # self.add(taylor_expansion_2)
        # self.add(p33_40_to_manim_10)


        # self.play(Write(p33_40_to_manim_07[-1]))
        # self.add(p33_40_to_manim_07[-1])

        # self.remove(taylor_expansion_plot_1)



        #reorient(0.0, 0.0, 0.0, (3.7340972, -1.8347028, 0.0), 8.74)  
        # reorient(0.0, 0.0, 0.0, (3.73, -1.84, 0.0), 8.74)  



        # self.add(gradient_descent_eq_3)
        # self.remove(gradient_descent_eq_2)
        # self.remove(gradient_descent_eq_2_copy_a)
        # self.remove(gradient_descent_eq_2_copy_b)
        # self.remove(nabla_copy)
        # self.remove(gs_copy)
        # self.remove(funky_arrow_2)
        # self.remove(gradient_vector_2)


        # self.add(funky_arrow_2)
        # self.add(gradient_vector_2)
        # self.remove(gradient_vector_2)




        # self.add(cross_out_line_1)
        # self.remove(cross_out_line_1)

        # self.play(gradient_descent_eq_2_copy_b[-1].animate


        # self.play(gradient_descent_eq_2_copy_b[-1].animate.move_to(gradient_descent_eq_2_copy_b[0].get_center()), 
        #           FadeOut(gradient_descent_eq_2_copy_b[0]))


        # self.add(gradient_descent_eq_2_copy_a, gradient_descent_eq_2_copy_b)

        # self.remove(gradient_descent_eq_2_copy_a, gradient_descent_eq_2_copy_b)

        # gradient_descent_eq_3 = Tex(r" = - \alpha (g_T - g_S)  \bigg]")
        # gradient_descent_eq_3.scale(0.9)


    #



class P27_30b(WelchScene):
    def construct(self):
        # p27_to_manim_1 = SVGMobject("p27_to_manim-01.svg")[1:].scale(4)
        p27_to_manim_2 = gx.SVGMobject(scene_asset('p27_to_manim-02.svg'))[1:].scale(4)
        p27_to_manim_3 = gx.SVGMobject(scene_asset('p27_to_manim-03.svg'))[1:].scale(4)
        # p27_to_manim_4 = SVGMobject("p27_to_manim-04.svg")[1:].scale(4)
        # p27_to_manim_5 = SVGMobject("p27_to_manim-05.svg")[1:].scale(4)
        p27_to_manim_6 = gx.SVGMobject(scene_asset('p27_to_manim-06.svg'))[1:].scale(4)
        # p27_to_manim_7 = SVGMobject("p27_to_manim-07.svg")[1:].scale(4)
        p28_to_manim_4 = gx.SVGMobject(scene_asset('p28_to_manim-04.svg'))[1:].scale(4)
        p29_to_32_to_manim_1 = gx.SVGMobject(scene_asset('p29_to_32_to_manim-01.svg'))[1:].scale(4)

        mnist_network = image_at_height(scene_asset('mnist_network.png'))

        n2_1 = p27_to_manim_2.submobjects[10]
        n3_2 = p27_to_manim_2.submobjects[9]
        n1_1 = p27_to_manim_2.submobjects[11]
        n1_2 = p27_to_manim_2.submobjects[12]
        n2_2 = p27_to_manim_2.submobjects[13]
        n3_1 = p27_to_manim_2.submobjects[8]

        n3_1_arrow = p27_to_manim_2.submobjects[14]
        n3_2_arrow = p27_to_manim_2.submobjects[15]

        n2_1_graph = p27_to_manim_2.submobjects[16]
        n2_2_graph = p27_to_manim_2.submobjects[21]

        n2_1_text = gx.VGroup(
            p27_to_manim_2.submobjects[17],
            p27_to_manim_2.submobjects[18],
            p27_to_manim_2.submobjects[19],
            p27_to_manim_2.submobjects[20],
        )
        n2_2_text = gx.VGroup(
            p27_to_manim_2.submobjects[22],
            p27_to_manim_2.submobjects[23],
            p27_to_manim_2.submobjects[24],
            p27_to_manim_2.submobjects[25],
        )

        left_dimension = gx.VGroup(
            p27_to_manim_3[0], p27_to_manim_3[1], p27_to_manim_3[2]
        ).next_to(gx.VGroup(n1_1, n1_2), gx.LEFT)
        top_dimension = gx.VGroup(
            p27_to_manim_3[3], p27_to_manim_3[4], p27_to_manim_3[5]
        ).next_to(gx.VGroup(n2_1, n3_1), gx.UP)

        x1 = math_tex(r"x_1").set_color(FRESH_TAN).move_to(n1_1.get_center())
        x2 = math_tex(r"x_2").set_color(FRESH_TAN).move_to(n1_2.get_center())

        model_parameters = (
            math_tex(
                r"\theta = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
            )
            .set_color(FRESH_TAN)
            .scale(0.75)
            .next_to(n2_2, gx.DOWN, buff=0.4)
        )

        model_parameters_text = (
            gx.Text("Model Parameters*", font="Myriad Pro", font_size=24, weight=gx.BOLD)
            .next_to(model_parameters, gx.DOWN)
            .set_color(CHILL_BROWN)
        )

        def edge_point(c1, c2):
            c1_center = c1.get_center()
            c2_center = c2.get_center()
            direction = gx.normalize(c2_center - c1_center)

            # ManimGL compatible approach - use get_corner instead of get_critical_point
            ul = c1.get_corner(gx.UP + gx.LEFT)
            ur = c1.get_corner(gx.UP + gx.RIGHT)
            dl = c1.get_corner(gx.DOWN + gx.LEFT)
            dr = c1.get_corner(gx.DOWN + gx.RIGHT)

            width = np.linalg.norm(ur - ul)
            height = np.linalg.norm(ul - dl)
            radius = max(width, height) / 2

            return c1_center + radius * direction

        def split_line(line, break_width, split_ratio):
            start = line.get_start()
            end = line.get_end()
            direction = (end - start) / line.get_length()

            split_point = start + direction * line.get_length() * split_ratio

            half_gap = direction * (break_width / 2)

            line1 = gx.Line(start, split_point - half_gap)
            line2 = gx.Line(split_point + half_gap, end)

            return line1, line2, split_point

        def animate_split_line(line, split_line_1, split_line_2, split_ratio):
            line_part_1, line_part_2, _ = split_line(line, 0, split_ratio)
            line_part_1.set_color(split_line_1.get_color())
            line_part_2.set_color(split_line_2.get_color())
            self.remove(line)
            return gx.AnimationGroup(
                gx.ReplacementTransform(line_part_1, split_line_1),
                gx.ReplacementTransform(line_part_2, split_line_2),
            )

        connections = gx.VGroup(
            gx.Line(edge_point(n1_1, n2_1), edge_point(n2_1, n1_1), color=FRESH_TAN),
            gx.Line(edge_point(n1_1, n2_2), edge_point(n2_2, n1_1), color=FRESH_TAN),
            gx.Line(edge_point(n1_2, n2_1), edge_point(n2_1, n1_2), color=FRESH_TAN),
            gx.Line(edge_point(n1_2, n2_2), edge_point(n2_2, n1_2), color=FRESH_TAN),
            gx.Line(edge_point(n2_1, n3_1), edge_point(n3_1, n2_1), color=FRESH_TAN),
            gx.Line(edge_point(n2_1, n3_2), edge_point(n3_2, n2_1), color=CHILL_BROWN),
            gx.Line(edge_point(n2_2, n3_1), edge_point(n3_1, n2_2), color=FRESH_TAN),
            gx.Line(edge_point(n2_2, n3_2), edge_point(n3_2, n2_2), color=CHILL_BROWN),
        )

        ln1_1_n2_1 = connections[0]
        ln1_1_n2_2 = connections[1]
        ln1_2_n2_1 = connections[2]
        ln1_2_n2_2 = connections[3]
        ln2_1_n3_1 = connections[4]
        ln2_1_n3_2 = connections[5]
        ln2_2_n3_1 = connections[6]
        ln2_2_n3_2 = connections[7]

        theta_1 = math_tex(r"\theta_1").scale(0.6).set_color(FRESH_TAN)
        theta_2 = math_tex(r"\theta_2").scale(0.6).set_color(FRESH_TAN)
        theta_3 = math_tex(r"\theta_3").scale(0.6).set_color(FRESH_TAN)
        theta_4 = math_tex(r"\theta_4").scale(0.6).set_color(FRESH_TAN)
        theta_5 = math_tex(r"\theta_5").scale(0.6).set_color(FRESH_TAN)
        theta_6 = math_tex(r"\theta_6").scale(0.6).set_color(FRESH_TAN)
        theta_7 = math_tex(r"\theta_7").scale(0.6).set_color(CHILL_BROWN)
        theta_8 = math_tex(r"\theta_8").scale(0.6).set_color(CHILL_BROWN)

        ln1_1_n2_1_left, ln1_1_n2_1_right, theta_1_position = split_line(
            ln1_1_n2_1, theta_1.width + 0.2, 0.5
        )
        theta_1.move_to(theta_1_position)
        ln1_1_n2_1_left.set_color(FRESH_TAN)
        ln1_1_n2_1_right.set_color(FRESH_TAN)

        ln1_2_n2_1_left, ln1_2_n2_1_right, theta_2_position = split_line(
            ln1_2_n2_1, theta_2.width + 0.2, 0.75
        )
        theta_2.move_to(theta_2_position)
        ln1_2_n2_1_left.set_color(FRESH_TAN)
        ln1_2_n2_1_right.set_color(FRESH_TAN)

        ln1_1_n2_2_left, ln1_1_n2_2_right, theta_3_position = split_line(
            ln1_1_n2_2, theta_3.width + 0.2, 0.75
        )
        theta_3.move_to(theta_3_position)
        ln1_1_n2_2_left.set_color(FRESH_TAN)
        ln1_1_n2_2_right.set_color(FRESH_TAN)

        ln1_2_n2_2_left, ln1_2_n2_2_right, theta_4_position = split_line(
            ln1_2_n2_2, theta_4.width + 0.2, 0.5
        )
        theta_4.move_to(theta_4_position)
        ln1_2_n2_2_left.set_color(FRESH_TAN)
        ln1_2_n2_2_right.set_color(FRESH_TAN)

        ln2_1_n3_1_left, ln2_1_n3_1_right, theta_5_position = split_line(
            ln2_1_n3_1, theta_5.width + 0.2, 0.5
        )
        theta_5.move_to(theta_5_position)
        ln2_1_n3_1_left.set_color(FRESH_TAN)
        ln2_1_n3_1_right.set_color(FRESH_TAN)

        ln2_2_n3_1_left, ln2_2_n3_1_right, theta_6_position = split_line(
            ln2_2_n3_1, theta_6.width + 0.2, 0.75
        )
        theta_6.move_to(theta_6_position)
        ln2_2_n3_1_left.set_color(FRESH_TAN)
        ln2_2_n3_1_right.set_color(FRESH_TAN)

        ln2_1_n3_2_left, ln2_1_n3_2_right, theta_7_position = split_line(
            ln2_1_n3_2, theta_7.width + 0.2, 0.75
        )
        theta_7.move_to(theta_7_position)
        ln2_1_n3_2_left.set_color(CHILL_BROWN)
        ln2_1_n3_2_right.set_color(CHILL_BROWN)

        ln2_2_n3_2_left, ln2_2_n3_2_right, theta_8_position = split_line(
            ln2_2_n3_2, theta_8.width + 0.2, 0.5
        )
        theta_8.move_to(theta_8_position)
        ln2_2_n3_2_left.set_color(CHILL_BROWN)
        ln2_2_n3_2_right.set_color(CHILL_BROWN)

        group1 = gx.AnimationGroup(gx.FadeIn(gx.VGroup(n1_1, n1_2, x1, x2)))

        group2 = gx.AnimationGroup(
            gx.GrowArrow(ln1_1_n2_1),
            gx.GrowArrow(ln1_1_n2_2),
            gx.GrowArrow(ln1_2_n2_1),
            gx.GrowArrow(ln1_2_n2_2),
        )

        group3 = gx.AnimationGroup(gx.FadeIn(gx.VGroup(n2_1, n2_2)))

        group4 = gx.AnimationGroup(
            gx.FadeIn(gx.VGroup(n2_1_graph, n2_2_graph)),
            gx.Write(n2_1_text),
            gx.Write(n2_2_text),
        )

        group5 = gx.AnimationGroup(
            gx.GrowArrow(ln2_1_n3_1),
            gx.GrowArrow(ln2_1_n3_2),
            gx.GrowArrow(ln2_2_n3_1),
            gx.GrowArrow(ln2_2_n3_2),
        )

        group6 = gx.AnimationGroup(
            gx.FadeIn(gx.VGroup(n3_1, n3_2)),
            gx.GrowFromEdge(n3_1_arrow, gx.LEFT),
            gx.GrowFromEdge(n3_2_arrow, gx.LEFT),
        )

        self.wait()
        self.play(
            gx.LaggedStart(
                group1,
                group2,
                group3,
                group4,
                group5,
                group6,
                lag_ratio=0.2,
                run_time=2.5,
            )
        )

        self.wait(1)

        self.add(left_dimension, top_dimension)

        self.wait(1)

        theta1_group = gx.AnimationGroup(
            animate_split_line(ln1_1_n2_1, ln1_1_n2_1_left, ln1_1_n2_1_right, 0.5),
            gx.Write(theta_1),
            lag_ratio=0.5,
        )

        theta2_group = gx.AnimationGroup(
            animate_split_line(ln1_2_n2_1, ln1_2_n2_1_left, ln1_2_n2_1_right, 0.75),
            gx.Write(theta_2),
            lag_ratio=0.5,
        )

        theta3_group = gx.AnimationGroup(
            animate_split_line(ln1_1_n2_2, ln1_1_n2_2_left, ln1_1_n2_2_right, 0.75),
            gx.Write(theta_3),
            lag_ratio=0.5,
        )

        theta4_group = gx.AnimationGroup(
            animate_split_line(ln1_2_n2_2, ln1_2_n2_2_left, ln1_2_n2_2_right, 0.5),
            gx.Write(theta_4),
            lag_ratio=0.5,
        )

        theta5_group = gx.AnimationGroup(
            animate_split_line(ln2_1_n3_1, ln2_1_n3_1_left, ln2_1_n3_1_right, 0.5),
            gx.Write(theta_5),
            lag_ratio=0.5,
        )

        theta6_group = gx.AnimationGroup(
            animate_split_line(ln2_2_n3_1, ln2_2_n3_1_left, ln2_2_n3_1_right, 0.75),
            gx.Write(theta_6),
            lag_ratio=0.5,
        )

        theta7_group = gx.AnimationGroup(
            animate_split_line(ln2_1_n3_2, ln2_1_n3_2_left, ln2_1_n3_2_right, 0.75),
            gx.Write(theta_7),
            lag_ratio=0.5,
        )

        theta8_group = gx.AnimationGroup(
            animate_split_line(ln2_2_n3_2, ln2_2_n3_2_left, ln2_2_n3_2_right, 0.5),
            gx.Write(theta_8),
            lag_ratio=0.5,
        )

        self.play(
            gx.LaggedStart(
                theta1_group,
                theta2_group,
                theta3_group,
                theta4_group,
                theta5_group,
                theta6_group,
                theta7_group,
                theta8_group,
                lag_ratio=0.3,
                run_time=5,
            )
        )

        self.wait(1)

        model_parameters_1 = gx.AnimationGroup(gx.FadeIn(model_parameters[0:3]))

        model_parameters_2 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_1.copy(), model_parameters[3:5]),
            gx.FadeIn(model_parameters[5]),
        )

        model_parameters_3 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_2.copy(), model_parameters[6:8]),
            gx.FadeIn(model_parameters[8]),
        )

        model_parameters_4 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_3.copy(), model_parameters[9:11]),
            gx.FadeIn(model_parameters[11]),
        )

        model_parameters_5 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_4.copy(), model_parameters[12:14]),
            gx.FadeIn(model_parameters[14]),
        )

        model_parameters_6 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_5.copy(), model_parameters[15:17]),
            gx.FadeIn(model_parameters[17]),
        )

        model_parameters_7 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_6.copy(), model_parameters[18:20]),
            gx.FadeIn(model_parameters[20]),
        )

        model_parameters_8 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_7.copy(), model_parameters[21:23]),
            gx.FadeIn(model_parameters[23]),
        )

        model_parameters_9 = gx.AnimationGroup(
            gx.ReplacementTransform(theta_8.copy(), model_parameters[24:26]),
            gx.FadeIn(model_parameters[26]),
        )

        self.play(
            gx.LaggedStart(
                model_parameters_1,
                model_parameters_2,
                model_parameters_3,
                model_parameters_4,
                model_parameters_5,
                model_parameters_6,
                model_parameters_7,
                model_parameters_8,
                model_parameters_9,
                lag_ratio=0.3,
                run_time=5,
            )
        )

        self.wait(1)

        p27_to_manim_6[29].next_to(n3_1_arrow, gx.RIGHT, buff=0.1)
        p27_to_manim_6[28].next_to(n3_2_arrow, gx.RIGHT, buff=0.1)

        p27_to_manim_6[0:13].next_to(p27_to_manim_6[29], gx.DOWN, buff=0.1)
        p27_to_manim_6[13:28].next_to(p27_to_manim_6[28], gx.DOWN, buff=0.1)

        primary_output = gx.VGroup(p27_to_manim_6[29], p27_to_manim_6[0:13])
        auxiliary_output = gx.VGroup(p27_to_manim_6[28], p27_to_manim_6[13:28])

        outputs = gx.VGroup(primary_output, auxiliary_output)

        """self.play(FadeIn(primary_output), FadeIn(auxiliary_output), model_parameters.animate.move_to([ 2.81398236e-01, -1.57497560e+00, -2.99721055e-17]))

        self.wait(1)"""

        # p28_to_manim_4.move_to([(primary_output.get_x() + p28_to_manim_4.width/2) - (p28_to_manim_4.submobjects[1].width/2), 0, 0])

        p28_to_manim_4.move_to(
            [
                (primary_output.get_x() + p28_to_manim_4.width / 2)
                - (p28_to_manim_4.submobjects[1].width / 2),
                0,
                0,
            ]
        )

        top_arrow = gx.VGroup(
            p28_to_manim_4.submobjects[0],
            p28_to_manim_4.submobjects[1],
            p28_to_manim_4.submobjects[2],
        )
        bottom_arrow = gx.VGroup(
            p28_to_manim_4.submobjects[3],
            p28_to_manim_4.submobjects[4],
            p28_to_manim_4.submobjects[5],
        )

        top_outputs = outputs.get_top()[1]
        bottom_outputs = outputs.get_bottom()[1]
        bottom_submob1 = p28_to_manim_4.submobjects[1].get_bottom()[1]
        top_submob5 = p28_to_manim_4.submobjects[5].get_top()[1]
        vertical_shift = (
            top_outputs - bottom_submob1 - top_submob5 + bottom_outputs
        ) / 2

        # Apply the shift
        p28_to_manim_4.shift([0, vertical_shift, 0])

        # MATH TIME ;(
        """
        p28_to_manim_4.width = 5.860444245515045
        was about to pull out the ti84 for this lol
        """

        mnist_network.move_to([7.325, -0.175, 0])  # TODO: Adjust position

        self.camera.frame.save_state()

        # self.play(FadeIn(model_parameters_text))

        self.wait()
        self.play(gx.LaggedStart(gx.FadeIn(primary_output)))
        self.wait()

        self.play(
            self.camera.frame.animate.move_to(
                [3, 0, 0]
            ),  # TODO: Update camera position
            gx.FadeOut(top_dimension),
            gx.FadeOut(left_dimension),
            # FadeOut(model_parameters_text),
            run_time=2,
        )
        self.play(gx.FadeIn(top_arrow), gx.FadeIn(mnist_network))


        self.wait(1)

        self.play(
            gx.LaggedStart(gx.FadeIn(auxiliary_output), gx.FadeIn(bottom_arrow), lag_ratio=0.75)
        )

        # when mnist gets updated
        # self.play(FadeIn(mnist_network))

        self.wait(1)

        # P29

        self.play(
            self.camera.frame.animate.move_to([0, 0, 0]),
            gx.FadeOut(mnist_network),
            gx.FadeOut(p28_to_manim_4),
            run_time=2,
        )

        self.wait(3)

        t_network_copy = gx.VGroup(
            # Input neurons with labels
            n1_1.copy(),
            n1_2.copy(),
            x1.copy(),
            x2.copy(),
            # Hidden layer neurons and their activation functions
            n2_1.copy(),
            n2_2.copy(),
            n2_1_graph.copy(),
            n2_2_graph.copy(),
            n2_1_text.copy(),
            n2_2_text.copy(),
            # Output layer neurons and arrows
            n3_1.copy(),
            n3_2.copy(),
            n3_1_arrow.copy(),
            n3_2_arrow.copy(),
            # Split lines between input and hidden layer
            ln1_1_n2_1_left.copy(),
            ln1_1_n2_1_right.copy(),
            ln1_2_n2_1_left.copy(),
            ln1_2_n2_1_right.copy(),
            ln1_1_n2_2_left.copy(),
            ln1_1_n2_2_right.copy(),
            ln1_2_n2_2_left.copy(),
            ln1_2_n2_2_right.copy(),
            # Split lines between hidden and output layer
            ln2_1_n3_1_left.copy(),
            ln2_1_n3_1_right.copy(),
            ln2_2_n3_1_left.copy(),
            ln2_2_n3_1_right.copy(),
            ln2_1_n3_2_left.copy(),
            ln2_1_n3_2_right.copy(),
            ln2_2_n3_2_left.copy(),
            ln2_2_n3_2_right.copy(),
            # Thetas (weights) on the connections
            theta_1.copy(),
            theta_2.copy(),
            theta_3.copy(),
            theta_4.copy(),
            theta_5.copy(),
            theta_6.copy(),
            theta_7.copy(),
            theta_8.copy(),
            model_parameters.copy(),
            auxiliary_output.copy(),
            primary_output.copy(),
        )


        # Assign individual variables for each element in the VGroup
        s_n1_1 = t_network_copy[0]
        s_n1_2 = t_network_copy[1]
        s_x1 = t_network_copy[2]
        s_x2 = t_network_copy[3]
        s_n2_1 = t_network_copy[4]
        s_n2_2 = t_network_copy[5]
        s_n2_1_graph = t_network_copy[6]
        s_n2_2_graph = t_network_copy[7]
        s_n2_1_text = t_network_copy[8]
        s_n2_2_text = t_network_copy[9]
        s_n3_1 = t_network_copy[10]
        s_n3_2 = t_network_copy[11]
        s_n3_1_arrow = t_network_copy[12]
        s_n3_2_arrow = t_network_copy[13]
        s_ln1_1_n2_1_left = t_network_copy[14]
        s_ln1_1_n2_1_right = t_network_copy[15]
        s_ln1_2_n2_1_left = t_network_copy[16]
        s_ln1_2_n2_1_right = t_network_copy[17]
        s_ln1_1_n2_2_left = t_network_copy[18]
        s_ln1_1_n2_2_right = t_network_copy[19]
        s_ln1_2_n2_2_left = t_network_copy[20]
        s_ln1_2_n2_2_right = t_network_copy[21]
        s_ln2_1_n3_1_left = t_network_copy[22]
        s_ln2_1_n3_1_right = t_network_copy[23]
        s_ln2_2_n3_1_left = t_network_copy[24]
        s_ln2_2_n3_1_right = t_network_copy[25]
        s_ln2_1_n3_2_left = t_network_copy[26]
        s_ln2_1_n3_2_right = t_network_copy[27]
        s_ln2_2_n3_2_left = t_network_copy[28]
        s_ln2_2_n3_2_right = t_network_copy[29]
        s_theta_1 = t_network_copy[30]
        s_theta_2 = t_network_copy[31]
        s_theta_3 = t_network_copy[32]
        s_theta_4 = t_network_copy[33]
        s_theta_5 = t_network_copy[34]
        s_theta_6 = t_network_copy[35]
        s_theta_7 = t_network_copy[36]
        s_theta_8 = t_network_copy[37]
        s_model_parameters = t_network_copy[38]
        s_auxiliary_output = t_network_copy[39]
        s_primary_output = t_network_copy[40]

        self.wait()

        self.play(
            # self.camera.frame.animate.scale_to_fit_width(16).move_to([3.9, -1.4, 0]),
            self.animate_view(0.0, 0.0, 0.0, (3.73, -1.84, 0.0), 8.74), #SW little different framing
            t_network_copy.animate.shift(gx.RIGHT * 7),
            run_time=3,  # TODO: Find best camera position
        )

        self.wait(1)

        # self.play(t_network_copy.animate.shift(RIGHT * 7), run_time=2.5)

        self.wait(1)




        self.wait()
        self.wait(20)







