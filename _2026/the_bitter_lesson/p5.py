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


audio_fn='/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/exports/tell_me_about_china.wav'
spectra_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/bitter_lesson/hacking/spectral_envelopes_final.npy'


class P5b(InteractiveScene):
    def construct(self): 

        '''
        Ok let me hack a bit on how to do the waveform here. I guess first I need to 
        get the take from the final a-roll, glad that's done!


        '''
        axes_width = 9 #Initial axis width

        sample_rate, data = wavfile.read(audio_fn)
        data = data / np.max(np.abs(data))
        
        # Downsample for plotting (adjust factor as needed)
        # downsample_factor = max(1, len(data) // 5000)
        downsample_factor=12
        data_ds = data[::downsample_factor]
        print(len(data_ds))
        
        # Time array
        duration = len(data) / sample_rate
        t = np.linspace(0, duration, len(data_ds))
        
        # Create axes
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
        axes.move_to([0, 2.5, 0])
        # self.add(axes)
        
        waveform = VMobject()
        waveform.set_stroke(BLUE, width=3)

        self.wait()
        self.add(waveform)

        N = 20
        ## Audio "playing in" - Uncomment for final render
        # for i in range(N, len(data_ds), N):
        #     points = [axes.c2p(t[j], data_ds[j]) for j in range(i)]
        #     waveform.set_points_as_corners(points)
        #     self.wait(1/30) 
        
        # Final frame with all samples
        points = [axes.c2p(t[j], data_ds[j]) for j in range(len(data_ds))]
        waveform.set_points_as_corners(points)

        self.wait()

        #Ok, matthew is going to provide exact cutpoints, let me fake a few for now
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
        spacing=0.35

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
            self.wait()
            if count<num_steps-1:
                self.remove(segments)

        self.wait()

        # Ok that's nice, now I need 12 little spectrograms. 
        # I think it's going to make the most sense to space them evenly
        # instead of trying to line them up with each audio chunk
        # I can add arrows connecting wavesforms to spectograms, probably in 
        # illustrator. Or I can warp each chunk to make them even in the 
        # previous step -> we'l see what make sense here!

        spectral_envelope_outputs_loaded = np.load(spectra_path, allow_pickle=True)
        spectral_envelopes_loaded_dict = spectral_envelope_outputs_loaded.item()

        phone_keys = ["stephen_tell_T", "stephen_tell_EL", 
                       "stephen_me_M", "stephen_me_IY", "stephen_about_A", 
                       "stephen_about_B", "stephen_about_AW", "stephen_about_T", 
                       "stephen_china_SH", "stephen_china_AY", "stephen_china_N", 
                       "stephen_china_UH"]

        horizontal_spacing = 1.1

        spectra_plots=Group()
        for count, phone_key in enumerate(phone_keys):

            axes_2 = Axes(
                x_range=[0, 25000, 5000],
                y_range=[0, 1, 0.5],
                width=0.9,
                height=0.6,
                axis_config={
                    "color": CHILL_BROWN,
                    "stroke_width": 2.0,
                    "include_ticks": False,
                    "include_tip": True,
                    "tip_config": {"width":0.01, "length":0.01}
                },
            )
            axes_2.move_to([-6+horizontal_spacing*count, 1, 0])

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


        # Ok specta look nice!
        # Now do I wan to use cool arrows, or try to show the mapping with animation?
        # Ok one at time like this looks nice, but it might be too slow 
        # in the edit. If it does end up being too slow, 
        # then we can change to lag ratio or all at once setup. 

        self.wait()
        segments_copy=segments.copy()
        for i in range(len(segments)):
            self.play(ReplacementTransform(segments_copy[i], spectra_plots[i][1]), 
                      ShowCreation(spectra_plots[i][0]), #axis
                      run_time=3)

        self.wait()

        # Ok making progress here. Now I recon that I really need the 
        # graph right? 



        self.wait(20)
        self.embed()






class P5a(InteractiveScene):
    def construct(self): 

        #I think I want to use the typewriter font!
        
        #Phone, node index, x coord, y coord
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

        #Connections between node ids
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
        
        # Create arrows
        arrows = VGroup()
        for start_idx, end_idx in edges:
            start_node = node_mobjects[start_idx]
            end_node = node_mobjects[end_idx]
            
            arrow = Arrow(
                start_node.get_center(),
                end_node.get_center(),
                buff=0.5,  # keeps arrow from overlapping rounded rects
            )
            arrow.set_color(CHILL_BROWN)
            arrows.add(arrow)
        
        # Group everything
        all_nodes = VGroup(*node_mobjects.values())
        graph = VGroup(arrows, all_nodes)
        graph.center()
        
        self.wait()
        self.frame.reorient(0, 0, 0, (-0.06, -0.62, 0.0), 14.46)
        self.add(graph)



        self.wait(20)
        self.embed()