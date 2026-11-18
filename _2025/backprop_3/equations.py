import manimgx as m
from welch_animation import TransformByGlyphMap
from welch_scene import WelchScene
from welch_text import math_tex

CHILL_BROWN='#cabba6'
COOL_BLUE = '#00ffff'
COOL_YELLOW = '#ffd35a'
COOL_GREEN = '#00a14b'
MIXED_GREEN = '#80E9AD'


class Equations(WelchScene):
    def construct(self):
        hidden1_layer1 = math_tex(r"h_{1}^{(1)} = m_{11}^{(1)} x_{1} + m_{12}^{(1)} x_{2} + b_{1}^{(1)}").scale(1.5).set_color(COOL_BLUE)
        hidden2_layer1 = math_tex(r"h_{2}^{(1)} = m_{21}^{(1)} x_{1} + m_{22}^{(1)} x_{2} + b_{2}^{(1)}").scale(1.5).set_color(COOL_YELLOW)
        
        hidden1_layer2 = math_tex(r"h_{1}^{(2)} = m_{11}^{(2)} h_{1}^{(1)} + m_{12}^{(2)} h_{2}^{(1)} + b_{1}^{(2)}").scale(1.5)
        hidden1_layer2[12:17].set_color(COOL_BLUE)
        hidden1_layer2[24:29].set_color(COOL_YELLOW)
        
        
        hidden2_layer2 = math_tex(r"h_{2}^{(2)} = m_{21}^{(2)} h_{1}^{(1)} + m_{22}^{(2)} h_{2}^{(1)} + b_{2}^{(2)}").scale(1.5)
        # Additional expanded/grouped equations
        hidden1_layer2_expanded_brackets = math_tex(
            r"h_{1}^{(2)} = m_{11}^{(2)}\bigg( m_{11}^{(1)}x_{1}+m_{12}^{(1)}x_{2}+b_1^{(1)} \bigg) "
            r"+ m_{12}^{(2)}\bigg( m_{21}^{(1)}x_{1}+m_{22}^{(1)}x_{2}+b_2^{(1)} \bigg) + b_1^{(2)}"
        ).scale(1.5).move_to(hidden1_layer2.get_center())
        
        hidden1_layer2_expanded_brackets[13:36].set_color(COOL_BLUE)
        hidden1_layer2_expanded_brackets[45:68].set_color(COOL_YELLOW)
        
        hidden1_layer2_expanded_brackets_part1 = math_tex(
            r"h_{1}^{(2)} = m_{11}^{(2)}\bigg( m_{11}^{(1)}x_{1}+m_{12}^{(1)}x_{2}+b_1^{(1)} \bigg) "
            r"+ m_{12}^{(2)} h_{2}^{(1)} + b_{1}^{(2)}"
        ).scale(1.5).move_to(hidden1_layer2.get_center())
        
        hidden1_layer2_expanded_brackets_part1[13:36].set_color(COOL_BLUE)
        hidden1_layer2_expanded_brackets_part1[44:49].set_color(COOL_YELLOW)
        
        hidden1_layer2_expanded = math_tex(
            r"h_{1}^{(2)} = m_{11}^{(2)} m_{11}^{(1)} x_{1} + m_{11}^{(2)} m_{12}^{(1)} x_{2} + m_{11}^{(2)} b_1^{(1)} "
            r"+ m_{12}^{(2)} m_{21}^{(1)} x_{1} + m_{12}^{(2)} m_{22}^{(1)} x_{2} + m_{12}^{(2)} b_2^{(1)} + b_1^{(2)}"
        ).scale(1.5)
        
        hidden1_layer2_expanded[12:20].set_color(COOL_BLUE)
        hidden1_layer2_expanded[27:35].set_color(COOL_BLUE)
        hidden1_layer2_expanded[42:47].set_color(COOL_BLUE)
        hidden1_layer2_expanded[54:62].set_color(COOL_YELLOW)
        hidden1_layer2_expanded[69:77].set_color(COOL_YELLOW)
        hidden1_layer2_expanded[84:89].set_color(COOL_YELLOW)
        
        hidden1_layer2_expanded_part1 = math_tex(
            r"h_{1}^{(2)} = m_{11}^{(2)} m_{11}^{(1)} x_{1} + m_{11}^{(2)} m_{12}^{(1)} x_{2} + m_{11}^{(2)} b_1^{(1)} "
            r"+ m_{12}^{(2)}\bigg( m_{21}^{(1)}x_{1}+m_{22}^{(1)}x_{2}+b_2^{(1)} \bigg) + b_1^{(2)}"
        ).scale(1.5)
        
        hidden1_layer2_expanded_part1[12:20].set_color(COOL_BLUE)
        hidden1_layer2_expanded_part1[27:35].set_color(COOL_BLUE)
        hidden1_layer2_expanded_part1[42:47].set_color(COOL_BLUE)
        hidden1_layer2_expanded_part1[55:78].set_color(COOL_YELLOW)
        
        
        hidden1_layer2_expanded_rearranged = math_tex(
            r"h_{1}^{(2)} = m_{11}^{(2)} m_{11}^{(1)} x_{1} + m_{12}^{(2)} m_{21}^{(1)} x_{1} + m_{11}^{(2)} m_{12}^{(1)} x_{2} + m_{12}^{(2)} m_{22}^{(1)} x_{2} + m_{11}^{(2)} b_1^{(1)} + m_{12}^{(2)} b_2^{(1)} + b_1^{(2)}"
        ).scale(1.5)
        
        hidden1_layer2_expanded_rearranged[12:20].set_color(COOL_BLUE)
        hidden1_layer2_expanded_rearranged[27:35].set_color(COOL_YELLOW)
        hidden1_layer2_expanded_rearranged[42:50].set_color(COOL_BLUE)
        hidden1_layer2_expanded_rearranged[57:65].set_color(COOL_YELLOW)
        hidden1_layer2_expanded_rearranged[72:77].set_color(COOL_BLUE)
        hidden1_layer2_expanded_rearranged[84:89].set_color(COOL_YELLOW)
        
        hidden1_layer2_grouped = math_tex(
            r"h_{1}^{(2)} = \bigg(m_{11}^{(2)} m_{11}^{(1)} + m_{12}^{(2)} m_{21}^{(1)}\bigg) x_{1} "
            r"+ \bigg(m_{11}^{(2)} m_{12}^{(1)} + m_{12}^{(2)} m_{22}^{(1)}\bigg) x_{2} "
            r"+ \bigg(m_{11}^{(2)} b_1^{(1)} + m_{12}^{(2)} b_2^{(1)} + b_1^{(2)}\bigg)"
        ).scale(1.5)
        
        hidden1_layer2_grouped[13:19].set_color(COOL_BLUE)
        hidden1_layer2_grouped[26:32].set_color(COOL_YELLOW)
        hidden1_layer2_grouped[43:49].set_color(COOL_BLUE)
        hidden1_layer2_grouped[56:62].set_color(COOL_YELLOW)
        hidden1_layer2_grouped[73:78].set_color(COOL_BLUE)
        hidden1_layer2_grouped[85:90].set_color(COOL_YELLOW)
        
        
        softmax_output = math_tex(r"\hat{y}_i = \text{Softmax}(h_i^{(2)}) = \frac{e^{h_i^{(2)}}}{e^{h_{1}^{(2)}} + e^{h_{2}^{(2)}}}")



        

        self.play(m.FadeIn(hidden1_layer1), run_time=2)
        
        self.wait()
        
        self.play(m.FadeIn(hidden2_layer1), m.VGroup(hidden1_layer1, hidden2_layer1).animate.arrange(m.DOWN), run_time=2)
        
        self.wait()

        self.play(m.VGroup(hidden1_layer1, hidden2_layer1).animate.shift(m.UP * 2.5), m.FadeIn(hidden1_layer2), run_time=2)

        self.wait()


        
        # self.add(index_labels(hidden2_layer2).set_color(GREEN).set_backstroke(BLACK, 5))
        # self.add(index_labels(softmax_output).set_color(BLUE).set_backstroke(BLACK, 5))
        
        # im going to include a couple of animations that maybe useful later on
        
        '''self.play(  
            FlashAround(hidden1_layer1[0:5]),
            FlashAround(hidden1_layer2[12:17])
        )
        
        
        
        
        
        
        self.play(
            FlashAround(hidden2_layer1[0:5]),
            FlashAround(hidden1_layer2[24:29])
        )'''
        
        
        
        self.camera.frame.save_state()
                
        self.remove(hidden1_layer2[12:17])
        
        
        self.play(
            m.ReplacementTransform(hidden1_layer2[0:12], hidden1_layer2_expanded_brackets_part1[0:12]),
            m.ReplacementTransform(hidden1_layer2[17:35], hidden1_layer2_expanded_brackets_part1[37:55]),
            m.ReplacementTransform(hidden1_layer1[6:29], hidden1_layer2_expanded_brackets_part1[13:36]),
            
            m.FadeOut(hidden1_layer1[0:6]),
            self.camera.frame.animate.scale_to_fit_width(hidden1_layer2_expanded_brackets_part1.width * 1.1), run_time=2
        )
        
        self.add(m.VGroup(hidden1_layer2_expanded_brackets_part1[12], hidden1_layer2_expanded_brackets_part1[36]))
        
        
        
        
        
        self.wait()
        
        self.remove(hidden1_layer2_expanded_brackets_part1[44:49])
        
        
        self.play(
            m.ReplacementTransform(hidden1_layer2_expanded_brackets_part1[0:44], hidden1_layer2_expanded_brackets[0:44]),
            m.ReplacementTransform(hidden1_layer2_expanded_brackets_part1[49:55], hidden1_layer2_expanded_brackets[69:75]),
            m.ReplacementTransform(hidden2_layer1[6:29], hidden1_layer2_expanded_brackets[45:68]),
            m.FadeOut(hidden2_layer1[0:6]),
            self.camera.frame.animate.scale_to_fit_width(hidden1_layer2_expanded_brackets.width * 1.1), run_time=2
        )
        
        self.add(m.VGroup(hidden1_layer2_expanded_brackets[44], hidden1_layer2_expanded_brackets[68]))

        
        self.wait()
        
        
        
        self.play(
            TransformByGlyphMap(hidden1_layer2_expanded_brackets, hidden1_layer2_expanded_part1,
                (list(range(37, 75)), list(range(47, 85))),
                (list(range(0, 6)), list(range(0, 6))),
                ([12, 36],  [], {"run_time":0.0000001}),
                ([21], [20]),
                ([30], [35]),
                (list(range(13, 21)), list(range(12, 20))),
                (list(range(22, 30)), list(range(27, 35))),
                (list(range(31, 36)), list(range(42, 47))),
                (list(range(6, 12)), list(range(6, 12))), # basically no point to put an arc on this because it is in the same spot
                (list(range(6, 12)), list(range(21, 27)), {"path_arc":-2/3*m.PI}),
                (list(range(6, 12)), list(range(36, 42)), {"path_arc":-1/3*m.PI}),
            ),
            self.camera.frame.animate.scale_to_fit_width(hidden1_layer2_expanded_part1.width * 1.1), run_time=2
        )

        
        self.wait()
        
        self.remove()
        
        
        
        self.play(
            TransformByGlyphMap(hidden1_layer2_expanded_part1, hidden1_layer2_expanded,
                (list(range(0, 48)), list(range(0, 48))),
                ([54, 78], [], {"run_time":0.0000001}),
                ([63], [62]),
                ([72], [77]),
                (list(range(55, 63)), list(range(54, 62))),
                (list(range(64, 72)), list(range(69, 77))),
                (list(range(73, 78)), list(range(84, 89))),
                (list(range(48, 54)), list(range(48, 54))), # basically no point to put an arc on this because it is in the same spot
                (list(range(48, 54)), list(range(63, 69)), {"path_arc":-2/3*m.PI}),
                (list(range(48, 54)), list(range(78, 84)), {"path_arc":-1/3*m.PI}),
            ),
            self.camera.frame.animate.scale_to_fit_width(hidden1_layer2_expanded.width * 1.1), run_time=2
        )
        
        self.wait()
        
        
        
        self.play(
            TransformByGlyphMap(hidden1_layer2_expanded, hidden1_layer2_expanded_rearranged,
                (list(range(0, 6)), list(range(0, 6))),
                ([20], [20]),
                ([35], [35]),
                ([47], [50]),
                ([62], [65]),
                ([77], [77]),
                ([89], [89]),
                (list(range(78, 89)), list(range(78, 89))),
                (list(range(90, 95)), list(range(90, 95))),
                (list(range(6, 20)), list(range(6, 20))),
                (list(range(21, 35)), list(range(36, 50)), {"path_arc":1/3*m.PI}),
                (list(range(36, 47)), list(range(66, 77)), {"path_arc":2/3*m.PI}),
                (list(range(48, 62)), list(range(21, 35)), {"path_arc":2/3*m.PI}),
                (list(range(63, 77)), list(range(51, 65)), {"path_arc":1/3*m.PI}),
            ),
            self.camera.frame.animate.scale_to_fit_width(hidden1_layer2_expanded_rearranged.width * 1.1), run_time=2
        )
        
        self.wait()
        
        
        #self.embed()
        
                
        
        '''self.play(
            ReplacementTransform(hidden1_layer2_expanded_rearranged[0:6], hidden1_layer2_grouped[0:6]),
            ReplacementTransform(VGroup(hidden1_layer2_expanded_rearranged[18], hidden1_layer2_expanded_rearranged[19]), hidden1_layer2_grouped[33:35].copy()),
            ReplacementTransform(VGroup(hidden1_layer2_expanded_rearranged[33], hidden1_layer2_expanded_rearranged[34]), hidden1_layer2_grouped[33:35]),
            ReplacementTransform(hidden1_layer2_expanded_rearranged[6:18], hidden1_layer2_grouped[7:19]),
            ReplacementTransform(hidden1_layer2_expanded_rearranged[20:33], hidden1_layer2_grouped[19:32]),
            ReplacementTransform(hidden1_layer2_expanded_rearranged[35], hidden1_layer2_grouped[35]),
            ReplacementTransform(hidden1_layer2_expanded_rearranged[36:48], hidden1_layer2_grouped[37:49]),
            ReplacementTransform(hidden1_layer2_expanded_rearranged[50:63], hidden1_layer2_grouped[49:62]),
            ReplacementTransform(VGroup(hidden1_layer2_expanded_rearranged[48], hidden1_layer2_expanded_rearranged[49]), hidden1_layer2_grouped[63:65].copy()),
            ReplacementTransform(VGroup(hidden1_layer2_expanded_rearranged[63], hidden1_layer2_expanded_rearranged[64]), hidden1_layer2_grouped[63:65]),
            ReplacementTransform(hidden1_layer2_expanded_rearranged[65], hidden1_layer2_grouped[65]),
            ReplacementTransform(hidden1_layer2_expanded_rearranged[66:95], hidden1_layer2_grouped[67:96]),
            self.camera.frame.animate.scale_to_fit_width(hidden1_layer2_grouped.width * 1.1)
        )'''
        
        self.play(
            TransformByGlyphMap(hidden1_layer2_expanded_rearranged, hidden1_layer2_grouped,
                (list(range(0, 6)), list(range(0, 6))),
                (list(range(6, 18)), list(range(7, 19))),
                (list(range(20, 33)), list(range(19, 32))),
                (list(range(36, 48)), list(range(37, 49))),
                (list(range(66, 95)), list(range(67, 96))),
                ([35], [35]),
                ([65], [65]),
                (m.FadeIn, [6, 32, 36, 62, 66, 96], {"delay":2}),
                ([18, 19], [33, 34], {"path_arc":2/3*m.PI}),
                ([33, 34], [33, 34]),
                ([48, 49], [63, 64], {"path_arc":2/3*m.PI}),
                ([63, 64], [63, 64]),

                
                
            ),
            self.camera.frame.animate.scale_to_fit_width(hidden1_layer2_grouped.width * 1.1)
        )
                
        self.wait()
        

        pass
        
