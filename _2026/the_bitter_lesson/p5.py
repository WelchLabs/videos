from manimlib import *
from tqdm import tqdm
import re
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import wavfile


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' #6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

SCALE_FACTOR=0.4
SCALE_FACTOR_2=0.5


audio_fn='/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/exports/tell_me_about_china.wav'
spectra_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/hacking/spectral_envelopes_final.npy'

#Small graph configuration - Phone, node index, x coord, y coord
nodes=[['start', 0, 0, 0],
       ['T',     1, 4, 4], #Tell
       ['AH',    2, 6, 7],
       ['EL',    3, 8, 4], 
       ['M',     4, 11, 7], #Me
       ['IY',    5, 15, 7],
       ['IH',    6, 11, 1], #Us
       ['S',     7, 15, 1],
       ['OW',    8, 18, 4], #all
       ['EL',    9, 22, 4],
       ['A',     10, 26, 4], #About
       ['B',     11, 30, 4],
       ['AW',    12, 34, 4],
       ['T',     13, 38, 4],
       ['SH',    14, 41, 7], #CHINA
       ['AY',    15, 45, 7],
       ['N',     16, 49, 7],
       ['UH',    17, 53, 7],
       ['N',     18, 41, 1], #NIXON
       ['IH',    19, 45, 1],
       ['X',     20, 49, 1],
       ['EN',    21, 53, 1],
       ['G',     22, 4, -6], #GIVE
       ['IH',    23, 8, -6],
       ['V',     24, 12, -6],
       ['M',     25, 16, -6], #ME
       ['IY',    26, 20, -6],
       ['TH',    27, 24, -6], #THE
       ['UH',    28, 27, -3],
       ['EE',    29, 27, -9],
       ['H',     30, 31, -3], #HEADLINES
       ['AA',    31, 34.5, -3],
       ['D',     32, 38, -3],
       ['L',     33, 41, -3],
       ['AY',    34, 45, -3],
       ['N',     35, 49, -3],
       ['S',     36, 53, -3],
       ['N',     37, 31, -9], #NEWS
       ['OO',    38, 42, -9],
       ['S',     39, 53, -9],
       ['end',   40, 58, 0]
       ]

#Matches order of nodes
spectra_keys_1=['', 'mc_sent1_tell_T', 'mc_sent1_tell_AH', 'mc_sent1_tell_EL', 
                'mc_sent1_me_M', 'mc_sent1_me_IY', 'mc_sent1_us_IH', 'mc_sent1_us_S', 
                'mc_sent1_all_OW', 'mc_sent1_all_EL', 'mc_sent1_about_A', 
                'mc_sent1_about_B', 'mc_sent1_about_AW', 'mc_sent1_about_T', 
                'mc_sent1_china_SH', 'mc_sent1_china_AY', 'mc_sent1_china_N', 'mc_sent1_china_UH', 
                'mc_sent1_nixon_N', 'mc_sent1_nixon_IH', 'mc_sent1_nixon_X', 'mc_sent1_nixon_EN', 
                'mc_sent2_give_G', 'mc_sent2_give_IH', 'mc_sent2_give_V', 
                'mc_sent2_me_M', 'mc_sent2_me_IY', 
                'mc_sent2_the_TH', 'mc_sent2_the_UH', 'mc_sent2_the_UH', #Duplicated for EE
                'mc_sent2_headlines_H', 'mc_sent2_headlines_AA', 'mc_sent2_headlines_D', 'mc_sent2_headlines_L', 'mc_sent2_headlines_AY', 'mc_sent2_headlines_N', 'mc_sent2_headlines_S',
                'mc_sent2_news_N', 'mc_sent2_news_OO', 'mc_sent2_news_S', '']

edges=[[0, 1], 
          [1, 2],
          [2, 3],
          [1, 3],
          [3, 4],
          [3, 6],
          [4, 5],
          [6, 7],
          [5, 8],
          [5, 10],
          [7, 10],
          [7, 8],
          [8, 9],
          [9, 10],
          [10, 11],
          [11, 12],
          [12, 13],
          [13, 14],
          [14, 15],
          [15, 16],
          [16, 17],      
          [13, 18], 
          [18, 19], 
          [19, 20],  
          [20, 21], 
          [17, 40], 
          [21, 40],
          [0, 22],
          [22, 23],   
          [23, 24],  
          [24, 25],  
          [25, 26],  
          [26, 27],  
          [27, 28], 
          [27, 29],
          [28, 30],
          [28, 37],
          [29, 30],
          [29, 37],
          [30, 31],
          [31, 32],
          [32, 33],
          [33, 34],
          [34, 35],
          [35, 36],
          [36, 40],
          [37, 38], 
          [38, 39], 
          [39, 40]

        ]

spectra_keys_2 = ["stephen_tell_T", "stephen_tell_EL", 
               "stephen_me_M", "stephen_me_IY", "stephen_about_A", 
               "stephen_about_B", "stephen_about_AW", "stephen_about_T", 
               "stephen_china_SH", "stephen_china_AY", "stephen_china_N", 
               "stephen_china_UH"]


def get_rect_edge_point(rect, direction):
    """
    Get the point on a rectangle's edge in a given direction from its center.
    direction should be a unit vector.
    """
    center = rect.get_center()
    w = rect.get_width() / 2
    h = rect.get_height() / 2
    
    dx, dy = direction[0], direction[1]
    
    # Avoid division by zero
    if abs(dx) < 1e-8:
        # Vertical line
        t = h / abs(dy) if abs(dy) > 1e-8 else 0
    elif abs(dy) < 1e-8:
        # Horizontal line
        t = w / abs(dx)
    else:
        # Find intersection with both edges and take the closer one
        t_x = w / abs(dx)  # time to hit vertical edge
        t_y = h / abs(dy)  # time to hit horizontal edge
        t = min(t_x, t_y)
    
    return center + t * direction




class P5(InteractiveScene):
    def construct(self): 
        fast=False #False for final render

        node_mobjects = {}
        buff = 0.2
        for label, idx, x, y in nodes:
            text = Text(label, font="American Typewriter", color=CHILL_BROWN)
            text.set_color(CHILL_BROWN)
            box = RoundedRectangle(
                width=text.get_width() + 2 * buff,
                height=text.get_height() + 2 * buff,
                corner_radius=0.15,
                color=CHILL_BROWN,
            )
            node = VGroup(box, text)
            node.move_to([x*SCALE_FACTOR, y*SCALE_FACTOR, 0])
            node_mobjects[idx] = node
        
        arrows = VGroup()
        arrow_dict={}
        for start_idx, end_idx in edges:
            start_node = node_mobjects[start_idx]
            end_node = node_mobjects[end_idx]
            
            start_box = start_node[0]  # The RoundedRectangle
            end_box = end_node[0]
            
            # Direction from start to end
            direction = end_node.get_center() - start_node.get_center()
            direction = direction / np.linalg.norm(direction)  # normalize
            
            # Get edge points
            start_point = get_rect_edge_point(start_box, direction)
            end_point = get_rect_edge_point(end_box, -direction)
            
            # Small additional buffer for visual breathing room
            gap = 0.05
            start_point = start_point + gap * direction
            end_point = end_point - gap * direction
            
            arrow = Arrow(start_point, end_point, buff=0)
            arrow.set_color(CHILL_BROWN)
            arrows.add(arrow)
            arrow_dict[(start_idx, end_idx)]=arrow

        # Group everything
        all_nodes = VGroup(*node_mobjects.values())
        graph = VGroup(arrows, all_nodes)
        graph.center()
        
        self.frame.reorient(0, 0, 0, (-0.11, 0.17, 0.0), 14.76)

        self.wait()
        self.add(graph)

        #Ok lets trace some phrases here
        #“Tell me about China”
        self.wait()
        nodes_to_trace=[0, 1, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 40]
        for i, n in enumerate(nodes_to_trace): 
            node_mobjects[n].set_color(BLUE)
            if not fast: self.wait(0.1)
            if i<len(nodes_to_trace)-1:
                arrow_dict[nodes_to_trace[i], nodes_to_trace[i+1]].set_color(BLUE)
            if not fast: self.wait(0.1)
        self.wait()

        self.play(graph.animate.set_color(CHILL_BROWN))

        #“Tell me about Nixon”
        self.wait()
        nodes_to_trace=[0, 1, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21, 40]
        for i, n in enumerate(nodes_to_trace): 
            node_mobjects[n].set_color(BLUE)
            if not fast: self.wait(0.1)
            if i<len(nodes_to_trace)-1:
                arrow_dict[nodes_to_trace[i], nodes_to_trace[i+1]].set_color(BLUE)
            if not fast: self.wait(0.1)
        self.wait()

        self.play(graph.animate.set_color(CHILL_BROWN))

        # “Give me the headlines"
        self.wait()
        nodes_to_trace=[0, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 40]
        for i, n in enumerate(nodes_to_trace): 
            node_mobjects[n].set_color(BLUE)
            if not fast: self.wait(0.1)
            if i<len(nodes_to_trace)-1:
                arrow_dict[nodes_to_trace[i], nodes_to_trace[i+1]].set_color(BLUE)
            if not fast: self.wait(0.1)
        self.wait()

        self.play(graph.animate.set_color(CHILL_BROWN))

        # Ok now stuff is going to get kinda crazy! It's time to add little freq 
        # curves inside each node! No idea how this is going to work with spacing etc.
        # So i feel like it might not be insane to just evenly dilate the spacing 
        # for the frequency graph? 
        # Probably makes sense to build the freq graph separately, then come back  
        # And figure out how to animate between them. 
        # self.remove(graph)

        spectral_envelope_outputs_loaded = np.load(spectra_path, allow_pickle=True)
        spectral_envelopes_loaded_dict = spectral_envelope_outputs_loaded.item()

        # self.wait()

        node_mobjects_2 = {}
        buff = 0.2
        for spectra_key, (label, idx, x, y) in zip(spectra_keys_1, nodes):

            n=Group()
            if spectra_key == '':
                text = Text(label, font="American Typewriter", color=CHILL_BROWN)
                text.set_color(CHILL_BROWN)
                box = RoundedRectangle(
                    width=text.get_width() + 2 * buff,
                    height=text.get_height() + 2 * buff,
                    corner_radius=0.15,
                    color=CHILL_BROWN,
                )
                box.set_stroke(width=2)
                node = VGroup(text, box)
                # node.move_to([x*SCALE_FACTOR, y*SCALE_FACTOR, 0])
                node.move_to([x*SCALE_FACTOR_2-12, y*SCALE_FACTOR_2, 0])
                node_mobjects_2[idx] = node

            else:
                axes = Axes(
                    x_range=[0, 25000, 5000],
                    y_range=[0, 1, 0.5],
                    width=0.9,
                    height=0.6,
                    axis_config={
                        "color": CHILL_BROWN,
                        "stroke_width": 3.0,
                        "include_ticks": False,
                        "include_tip": True,
                        "tip_config": {"width":0.01, "length":0.01}
                    },
                )

                curr_result = spectral_envelopes_loaded_dict.get(spectra_key)
                w=curr_result.get("w")
                lpc_envelope = curr_result.get("lpc_envelope"),
                s=np.log10(lpc_envelope / np.max(lpc_envelope))[0]
                s=(s-s.min())/(s.max()-s.min())

                spectra_1 = VMobject()
                spectra_1.set_stroke(YELLOW, width=4)

                points = [axes.c2p(w[j], s[j]) for j in range(len(s))]
                spectra_1.set_points_as_corners(points)

                n.add(axes)
                n.add(spectra_1)
                node_mobjects_2[idx]=n

                text = Text(label, font="American Typewriter", color=CHILL_BROWN)
                text.set_color(CHILL_BROWN)
                text.scale(0.8)

                box = RoundedRectangle(
                    width=axes.get_width() + 2 * buff,
                    height=axes.get_height() + 2 * buff,
                    corner_radius=0.15,
                    color=CHILL_BROWN,
                )
                box.set_stroke(width=2)

                node = VGroup(axes, spectra_1, text, box)
                node.move_to([x*SCALE_FACTOR_2-12, y*SCALE_FACTOR_2, 0])
                text.move_to([x*SCALE_FACTOR_2+0.25-12, y*SCALE_FACTOR_2+0.2, 0])
                node_mobjects_2[idx] = node


        arrows_2 = VGroup()
        arrow_dict_2={}
        for start_idx, end_idx in edges:
            start_node = node_mobjects_2[start_idx]
            end_node = node_mobjects_2[end_idx]
            
            start_box = start_node[-1]  # The RoundedRectangle
            end_box = end_node[-1]
            
            # Direction from start to end
            direction = end_node.get_center() - start_node.get_center()
            direction = direction / np.linalg.norm(direction)  # normalize
            
            # Get edge points
            start_point = get_rect_edge_point(start_box, direction)
            end_point = get_rect_edge_point(end_box, -direction)
            
            # Small additional buffer for visual breathing room
            gap = 0.05
            start_point = start_point + gap * direction
            end_point = end_point - gap * direction
            
            arrow = Arrow(start_point, end_point, buff=0)
            arrow.set_color(CHILL_BROWN)
            arrows_2.add(arrow)
            arrow_dict_2[(start_idx, end_idx)]=arrow

        # Group everything
        all_nodes_2 = VGroup(*node_mobjects_2.values())
        graph_2 = VGroup(arrows_2, all_nodes_2)
        graph_2.center()


        self.wait()
        #Hell yeah - so this not perfect, but I think totally workable, and easy!
        self.play(ReplacementTransform(graph, graph_2), 
                 self.frame.animate.reorient(0, 0, 0, (0.18, 0.5, 0.0), 18.15), 
                 run_time=5)
        self.wait()


        self.play(self.frame.animate.reorient(0, 0, 0, (-7.31, 0.11, 0.0), 9.90), 
                  run_time=5)
        self.wait(0)

        self.play(self.frame.animate.reorient(0, 0, 0, (-0.08, 2.77, 0.0), 17.74),
                 run_time=5)
        self.wait()


        # self.frame.reorient(0, 0, 0, (-0.08, 3.68, 0.0), 17.74)
        # self.frame.reorient(0, 0, 0, (-0.08, 2.77, 0.0), 17.74)


        ## --- Ok! Now we can introduce some waveform action!
        axes_width = 22 #Initial axis width
        sample_rate, data = wavfile.read(audio_fn)
        data = data / np.max(np.abs(data))
        
        downsample_factor=12
        data_ds = data[::downsample_factor]
        print(len(data_ds))
        
        duration = len(data) / sample_rate
        t = np.linspace(0, duration, len(data_ds))

        axes = Axes(
            x_range=[0, duration, duration/4],
            y_range=[-1, 1, 0.5],
            width=axes_width,
            height=2.5,
            axis_config={
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_ticks": False,
                "tick_size": 0.05,
                "include_tip": True,
                "tip_config": {"width":0.02, "length":0.02}
            },
        )
        axes.move_to([0, 9, 0])
        # self.add(axes)
        
        waveform = VMobject()
        waveform.set_stroke(BLUE, width=3)
        self.add(waveform)

        self.wait()
        N = 20
        if not fast:
            ## Audio "playing in" -
            for i in range(N, len(data_ds), N):
                points = [axes.c2p(t[j], data_ds[j]) for j in range(i)]
                waveform.set_points_as_corners(points)
                self.wait(1/30) 
        else:
            self.add(waveform)
        
        # Final frame with all samples
        points = [axes.c2p(t[j], data_ds[j]) for j in range(len(data_ds))]
        waveform.set_points_as_corners(points)

        self.wait()
        # self.remove(waveform)

        ## --- Now break apart waveform
        cut_points=[0, 4500, 12000, 15500, 19000, 21500, 25000, 31500, 35000, 40500, 46000, 51500]
        phones_1=["T", "EL", "M", "IY", "AH", "B", "AW", "T", "SH", "AY", "N", "UH"]

        # Convert to downsampled indices
        cut_points_ds = [cp // downsample_factor for cp in cut_points]

        # Make sure we cover all data
        if cut_points_ds[-1] < len(data_ds):
            cut_points_ds.append(len(data_ds))

        # Get scaling info from axes
        axes_left = axes.c2p(0, 0)[0]
        y_center = axes.c2p(0, 0)[1]
        y_scale = axes.c2p(0, 1)[1] - y_center
        total_samples = len(data_ds)

        num_steps=30
        spacing=0.5 #0.35

        self.wait()
        self.remove(waveform)
        # gap_width = 0.3  # scene units between blocks
        for count, gap_width in enumerate(np.linspace(0, spacing, num_steps)):
            segments = VGroup()
            x_cursor = axes_left
            
            for i in range(len(cut_points_ds) - 1):
                start_idx = cut_points_ds[i]
                end_idx = cut_points_ds[i + 1]
                n_samples = end_idx - start_idx
                
                # Width proportional to sample count
                seg_width = (n_samples / total_samples) * axes_width
                
                segment_data = data_ds[start_idx:end_idx]
                
                seg = VMobject()
                seg.set_stroke(BLUE, width=3)
                
                local_x = np.linspace(0, seg_width, n_samples)
                points = [[x_cursor + local_x[j], y_center + segment_data[j] * y_scale, 0] 
                          for j in range(n_samples)]
                seg.set_points_as_corners(points)
                
                segments.add(seg)
                x_cursor += seg_width + gap_width

            # total_new_width = x_cursor - axes_left - gap_width  # subtract last gap
            # scale_factor = axes_width / total_new_width
            # segments.scale(scale_factor, about_point=axes.c2p(0, 0))
            # segments.move_to(axes.get_center())
            segments.set_x(axes.get_x())

            self.add(segments)
            self.wait(0.1)
            if count<num_steps-1:
                self.remove(segments)

        self.wait()
        # self.remove(segments)

        #No convert to sepctogrames
        spectral_envelope_outputs_loaded = np.load(spectra_path, allow_pickle=True)
        spectral_envelopes_loaded_dict = spectral_envelope_outputs_loaded.item()

        horizontal_spacing = 2.3

        spectra_plots=Group()
        for count, phone_key in enumerate(spectra_keys_2):

            axes_2 = Axes(
                x_range=[0, 25000, 5000],
                y_range=[0, 1, 0.5],
                width=1.9,
                height=1.3,
                axis_config={
                    "color": CHILL_BROWN,
                    "stroke_width": 4.0,
                    "include_ticks": False,
                    "include_tip": True,
                    "tip_config": {"width":0.01, "length":0.01}
                },
            )
            axes_2.move_to([-12.7+horizontal_spacing*count, 6.5, 0])

            curr_result = spectral_envelopes_loaded_dict.get(phone_key)
            w=curr_result.get("w")
            lpc_envelope = curr_result.get("lpc_envelope"),
            s=np.log10(lpc_envelope / np.max(lpc_envelope))[0]
            s=(s-s.min())/(s.max()-s.min())

            spectra_1 = VMobject()
            spectra_1.set_stroke(BLUE, width=4)

            points = [axes_2.c2p(w[j], s[j]) for j in range(len(s))]
            spectra_1.set_points_as_corners(points)

            spectra_plot=Group()
            spectra_plot.add(axes_2)
            spectra_plot.add(spectra_1)
            spectra_plots.add(spectra_plot)

        # self.add(spectra_plots)
        # self.remove(spectra_plots)

        self.wait()
        segments_copy=segments.copy()
        for i in range(len(segments)):
            self.play(ReplacementTransform(segments_copy[i], spectra_plots[i][1]), 
                      ShowCreation(spectra_plots[i][0]), #axis
                      run_time=3)
        self.wait()


        # Ok now start doing comparison and building out path in blue or maybe magenta!
        # So I kinda had two punch ins -> but I think just doing 1 is the way to go
        # self.play(self.frame.animate.reorient(0, 0, 0, (-5.66, 2.1, 0.0), 11.14),
        #           run_time=5)
        # self.wait()

        # self.remove(all_nodes_2[1][1])
        # self.add(all_nodes_2[1])

        # all_nodes_2[1].get_center()

        # self.remove(spectra_plots[0][0])

        spectra_curve_copy=spectra_plots[0][1].copy()
        spectra_curve_copy_2=spectra_plots[0][1].copy()
        self.wait()
        self.play(spectra_curve_copy.animate.move_to(all_nodes_2[1].get_center()).scale(0.5), 
                  spectra_curve_copy_2.animate.move_to(all_nodes_2[22].get_center()).scale(0.5), 
                  self.frame.animate.reorient(0, 0, 0, (-9.17, 0.08, 0.0), 7.06), #Eh?
                  run_time=7)
        self.wait()


        #Ok start to highlight path in majenta i recon?
        graph_2[0][0].set_color(MAGENTA)
        graph_2[1][0].set_color(MAGENTA)
        graph_2[1][1][2].set_color(MAGENTA)
        graph_2[1][1][3].set_color(MAGENTA)
        self.wait()


        self.play(self.frame.animate.reorient(0, 0, 0, (-8.74, 3.66, 0.0), 7.59), 
                  run_time=5)
        self.wait()

        #"EL"
        spectra_curve_copy=spectra_plots[1][1].copy()
        spectra_curve_copy_2=spectra_plots[1][1].copy()
        self.play(spectra_curve_copy.animate.move_to(all_nodes_2[2].get_center()).scale(0.5), 
                  spectra_curve_copy_2.animate.move_to(all_nodes_2[3].get_center()).scale(0.5), 
                  run_time=5)
        self.wait()

        graph_2[0][3].set_color(MAGENTA)
        graph_2[1][3][2].set_color(MAGENTA)
        graph_2[1][3][3].set_color(MAGENTA)
        self.wait()

        #M
        spectra_curve_copy=spectra_plots[2][1].copy()
        spectra_curve_copy_2=spectra_plots[2][1].copy()
        self.play(spectra_curve_copy.animate.move_to(all_nodes_2[4].get_center()).scale(0.5), 
                  spectra_curve_copy_2.animate.move_to(all_nodes_2[6].get_center()).scale(0.5), 
                  run_time=5)
        self.wait()

        graph_2[0][4].set_color(MAGENTA) #Arrow
        graph_2[1][4][2].set_color(MAGENTA)
        graph_2[1][4][3].set_color(MAGENTA)
        self.wait()

        # IY - I guess there's only one option here 
        # spectra_curve_copy=spectra_plots[3][1].copy()
        # spectra_curve_copy_2=spectra_plots[3][1].copy()
        # self.play(spectra_curve_copy.animate.move_to(all_nodes_2[5].get_center()).scale(0.5), 
        #           spectra_curve_copy_2.animate.move_to(all_nodes_2[7].get_center()).scale(0.5), 
        #           run_time=5)
        # self.wait()

        graph_2[0][6].set_color(MAGENTA) #Arrow
        graph_2[1][5][2].set_color(MAGENTA)
        graph_2[1][5][3].set_color(MAGENTA)
        self.wait()

        # Ok since the script goes kinda fast here, I kinda feel like we 
        # go to the full wide zoom out. 
        self.play(self.frame.animate.reorient(0, 0, 0, (-0.02, 3.14, 0.0), 17.59), 
                  run_time=4)

        #A
        spectra_curve_copy=spectra_plots[4][1].copy()
        spectra_curve_copy_2=spectra_plots[4][1].copy()
        self.play(spectra_curve_copy.animate.move_to(all_nodes_2[8].get_center()).scale(0.5), 
                  spectra_curve_copy_2.animate.move_to(all_nodes_2[10].get_center()).scale(0.5), 
                  run_time=5)
        self.wait()

        graph_2[0][9].set_color(MAGENTA) #Arrow
        graph_2[1][10][2].set_color(MAGENTA)
        graph_2[1][10][3].set_color(MAGENTA)
        self.wait()        

        #B
        graph_2[0][14].set_color(MAGENTA) #Arrow
        graph_2[1][11][2].set_color(MAGENTA)
        graph_2[1][11][3].set_color(MAGENTA)
        self.wait()    

        #AW
        graph_2[0][15].set_color(MAGENTA) #Arrow
        graph_2[1][12][2].set_color(MAGENTA)
        graph_2[1][12][3].set_color(MAGENTA)
        self.wait()   

        #T
        graph_2[0][16].set_color(MAGENTA) #Arrow
        graph_2[1][13][2].set_color(MAGENTA)
        graph_2[1][13][3].set_color(MAGENTA)
        self.wait()   

        #SH
        spectra_curve_copy=spectra_plots[8][1].copy()
        spectra_curve_copy_2=spectra_plots[8][1].copy()
        self.play(spectra_curve_copy.animate.move_to(all_nodes_2[14].get_center()).scale(0.5), 
                  spectra_curve_copy_2.animate.move_to(all_nodes_2[18].get_center()).scale(0.5), 
                  run_time=5)
        self.wait()        

        graph_2[0][17].set_color(MAGENTA) #Arrow
        graph_2[1][14][2].set_color(MAGENTA)
        graph_2[1][14][3].set_color(MAGENTA)
        self.wait()   

        #AY
        graph_2[0][18].set_color(MAGENTA) #Arrow
        graph_2[1][15][2].set_color(MAGENTA)
        graph_2[1][15][3].set_color(MAGENTA)
        self.wait()   

        #N
        graph_2[0][19].set_color(MAGENTA) #Arrow
        graph_2[1][16][2].set_color(MAGENTA)
        graph_2[1][16][3].set_color(MAGENTA)
        self.wait()   

        #UH
        graph_2[0][20].set_color(MAGENTA) #Arrow
        graph_2[1][17][2].set_color(MAGENTA)
        graph_2[1][17][3].set_color(MAGENTA)
        self.wait()   

        #END
        graph_2[0][25].set_color(MAGENTA) #Arrow
        graph_2[1][40].set_color(MAGENTA)
        self.wait()    

        # I'm a bit unsure if i want to show matching fo each block or not
        # It's kinda complicated due to beam search,and I have the footnote
        # I might leave it like it is for now. 


        self.wait(20)
        self.embed()







