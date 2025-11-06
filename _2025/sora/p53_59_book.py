from manimlib import *
import glob

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'

from torch.utils.data import DataLoader
from smalldiffusion import (
    ScheduleLogLinear, samples, Swissroll, ModelMixin, ScheduleDDPM
)

from typing import Callable
from tqdm import tqdm
import torch
from itertools import pairwise

def manual_camera_interpolation(start_orientation, end_orientation, num_steps):
    """
    Linearly interpolate between two camera orientations.
    
    Parameters:
    - start_orientation: List containing camera parameters with a tuple at index 3
    - end_orientation: List containing camera parameters with a tuple at index 3
    - num_steps: Number of interpolation steps (including start and end)
    
    Returns:
    - List of interpolated orientations
    """
    result = []
    
    for step in range(num_steps):
        # Calculate interpolation factor (0 to 1)
        t = step / (num_steps - 1) if num_steps > 1 else 0
        
        # Create a new orientation for this step
        interpolated = []
        
        for i in range(len(start_orientation)):
            if i == 3:  # Handle the tuple at position 3
                start_tuple = start_orientation[i]
                end_tuple = end_orientation[i]
                
                # Interpolate each element of the tuple
                interpolated_tuple = tuple(
                    start_tuple[j] + t * (end_tuple[j] - start_tuple[j])
                    for j in range(len(start_tuple))
                )
                
                interpolated.append(interpolated_tuple)
            else:  # Handle regular numeric values
                start_val = start_orientation[i]
                end_val = end_orientation[i]
                interpolated_val = start_val + t * (end_val - start_val)
                interpolated.append(interpolated_val)
        
        result.append(interpolated)
    
    return result

def get_color_wheel_colors(n_colors, saturation=1.0, value=1.0, start_hue=0.0):
    """
    Generate N evenly spaced colors from the color wheel.
    
    Args:
        n_colors: Number of colors to generate
        saturation: Color saturation (0.0 to 1.0)
        value: Color brightness/value (0.0 to 1.0) 
        start_hue: Starting hue position (0.0 to 1.0)
    
    Returns:
        List of Manim-compatible hex color strings
    """
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = (start_hue + i / n_colors) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

class CustomTracedPath(VMobject):
    """
    A custom traced path that supports:
    - Reverse playback with segment removal
    - Variable opacity based on distance from end
    - Manual control over path segments
    """
    def __init__(
        self,
        traced_point_func,
        stroke_width=2.0,
        stroke_color=YELLOW,
        opacity_range=(0.1, 0.8),  # (min_opacity, max_opacity)
        fade_length=20,  # Number of segments to fade over
        **kwargs
    ):
        super().__init__(**kwargs)
        self.traced_point_func = traced_point_func
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.opacity_range = opacity_range
        self.fade_length = fade_length
        
        # Store path segments as individual VMobjects
        self.segments = VGroup()
        self.traced_points = []
        self.is_tracing = True
        
        # Add updater for forward tracing
        self.add_updater(lambda m, dt: m.update_path(dt))
    
    def update_path(self, dt=0):
        """Update path during forward animation"""
        if not self.is_tracing or dt == 0:
            return
            
        point = self.traced_point_func()
        self.traced_points.append(point.copy())
        
        if len(self.traced_points) >= 2:
            # Create a new segment
            segment = Line(
                self.traced_points[-2], 
                self.traced_points[-1],
                stroke_width=self.stroke_width,
                stroke_color=self.stroke_color
            )
            
            # Apply opacity gradient
            self.segments.add(segment)
            self.update_segment_opacities()
            self.add(segment)
    
    def update_segment_opacities(self):
        """Update opacity of all segments based on their position"""
        n_segments = len(self.segments)
        if n_segments == 0:
            return
            
        min_op, max_op = self.opacity_range
        
        for i, segment in enumerate(self.segments):
            if i >= n_segments - self.fade_length:
                # Calculate fade based on distance from end
                fade_progress = (i - (n_segments - self.fade_length)) / self.fade_length
                opacity = min_op + (max_op - min_op) * fade_progress
            else:
                opacity = min_op
            segment.set_opacity(opacity)
    
    def remove_last_segment(self):
        """Remove the last segment (for reverse playback)
        Kinda hacky but just run 2x to fix bug
        """
        if len(self.segments) > 0:
            last_segment = self.segments[-1]
            self.segments.remove(last_segment)
            self.remove(last_segment)
            if len(self.traced_points) > 0:
                self.traced_points.pop()
            # self.update_segment_opacities()

        if len(self.segments) > 0:
            last_segment = self.segments[-1]
            self.segments.remove(last_segment)
            self.remove(last_segment)
            if len(self.traced_points) > 0:
                self.traced_points.pop()

        self.update_segment_opacities()
    
    def stop_tracing(self):
        """Stop the automatic tracing updater"""
        self.is_tracing = False
    
    def start_tracing(self):
        """Resume automatic tracing"""
        self.is_tracing = True
    
    def get_num_segments(self):
        """Get the current number of segments"""
        return len(self.segments)

class TrackerControlledVectorField(VectorField):
    def __init__(self, time_tracker, max_radius=2.0, min_opacity=0.1, max_opacity=0.7, **kwargs):
        self.time_tracker = time_tracker
        self.max_radius = max_radius  # Maximum radius for opacity calculation
        self.min_opacity = min_opacity  # Minimum opacity at max radius
        self.max_opacity = max_opacity  # Maximum opacity at origin
        super().__init__(**kwargs)
        
        # Add updater that triggers when tracker changes
        self.add_updater(self.update_from_tracker)
    
    def update_from_tracker(self, mob, dt):
        """Update vectors when tracker value changes"""
        # Only update if tracker value has changed significantly
        current_time = self.time_tracker.get_value()
        if not hasattr(self, '_last_time') or abs(current_time - self._last_time) > 0.01:
            self._last_time = current_time
            self.update_vectors()  # Redraw vectors with new time
            self.apply_radial_opacity()  # Apply opacity falloff after updating
    
    def apply_radial_opacity(self):
        """Apply radial opacity falloff from origin"""
        # Get the stroke opacities array (this creates it if it doesn't exist)
        opacities = self.get_stroke_opacities()
        
        # In ManimGL VectorField, each vector is represented by 8 points
        # Points 0,2,4,6 are the key points of each vector, with point 0 being the base
        n_vectors = len(self.sample_points)
        
        for i in range(n_vectors):
            # Get the base point of this vector (every 8th point starting from 0)
            base_point = self.sample_points[i]
            
            # Calculate distance from origin (assuming origin is at [0,0,0])
            distance = np.linalg.norm(base_point[:2])  # Only use x,y components
            
            # Calculate opacity based on distance
            # Linear falloff: opacity decreases linearly with distance
            opacity_factor = max(0, 1 - distance / self.max_radius)
            final_opacity = self.min_opacity + (self.max_opacity - self.min_opacity) * opacity_factor
            
            # Apply the opacity to all 8 points of this vector (except the last one)
            start_idx = i * 8
            end_idx = min(start_idx + 8, len(opacities))
            opacities[start_idx:end_idx] = final_opacity
        
        # Make sure the data is marked as changed
        self.note_changed_data()


class follow_trajectory_2b(InteractiveScene):
    def construct(self):
        '''
        Ok ok ok need to do a direct transition from p47b after fading out all the traces etc -> then bring
        in the full vector field - I think this is going to be dope!
        '''
        batch_size=2130
        dataset = Swissroll(np.pi/2, 5*np.pi, 100)
        loader = DataLoader(dataset, batch_size=batch_size)
        batch=next(iter(loader)).numpy()

        axes = Axes(
            x_range=[-1.2, 1.2, 0.5],
            y_range=[-1.2, 1.2, 0.5],
            height=7,
            width=7,
            axis_config={
                "color": CHILL_BROWN, 
                "stroke_width": 2,
                "include_tip": True,
                "include_ticks": False,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )

        axes.set_opacity(0.5)

        # Create extended axes with SAME center point and proportional scaling
        extended_axes = Axes(
            x_range=[-2.0, 2.0, 0.5],    # Extended range
            y_range=[-2.0, 2.0, 0.5],    # Extended range
            height=7 * (4.0/2.4),        # Scale height proportionally: original_height * (new_range/old_range)
            width=7 * (4.0/2.4),         # Scale width proportionally: original_width * (new_range/old_range)
            axis_config={"stroke_width": 0}  # Make invisible
        )

        # Move extended axes to same position as original axes
        extended_axes.move_to(axes.get_center())


        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.05)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(CHILL_BROWN)
        dots.set_opacity(0.5)


        # model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_20_1.pt') #Trained on 256 levels
        model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_24_1.pt') #Trained on 64 levels

        schedule = ScheduleLogLinear(N=64, sigma_min=0.01, sigma_max=1) #N=200
        bound=2.0 #Need to match extended axes bro
        num_heatmap_steps=30
        grid=[]
        for i, x in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
            for j, y in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
                grid.append([x,y])
        grid=torch.tensor(grid).float()

        gam=1
        mu=0.5 #0.5 is DDPM
        cfg_scale=0.0
        cond=None
        sigmas=schedule.sample_sigmas(64)
        xt_history=[]
        history_pre_noise=[]
        heatmaps=[]
        eps=None
        torch.manual_seed(2)

        with torch.no_grad():
            model.eval();
            xt=torch.randn((batch_size,) + model.input_dims)*sigmas[0] #Scaling by sigma here matters a lot - why is that???
            xt_history.append(xt.numpy())
            
            for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
                eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
                # eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
                sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
                eta = (sig_prev**2 - sig_p**2).sqrt()
                history_pre_noise.append(xt - (sig - sig_p) * eps)
                xt = xt - (sig - sig_p) * eps + eta * model.rand_input(xt.shape[0]).to(xt) #Straight remove adding random noise
                
                xt_history.append(xt.numpy())
                heatmaps.append(model.forward(grid, sig, cond=None))
        xt_history=np.array(xt_history)
        history_pre_noise=np.array(history_pre_noise)


        time_tracker = ValueTracker(0.0)  # Start at time 0

        def vector_function_with_tracker(coords_array):
            """Vector function that uses the ValueTracker for time"""
            current_time = time_tracker.get_value()
            max_time = 8.0  # Map time 0-8 to sigma indices 0-255
            sigma_idx = int(np.clip(current_time * 63 / max_time, 0, 63)) #Needs to be N-1
            
            try:
                res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=None)
                return -res.detach().numpy()
            except:
                return np.zeros((len(coords_array), 2))


        vector_field = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_with_tracker,
            coordinate_system=extended_axes,
            density=5.0,
            stroke_width=2.0,
            max_radius=6.0,      # Vectors fade to min_opacity at this distance
            min_opacity=0.5,     # Minimum opacity at max_radius
            max_opacity=0.5,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=CHILL_BROWN
        )
        

        # Ok so I'll need to noodle with a few different starting points - and am tempted ot start not quite at point 100, ya know?
        #Ok yeah so I need to find path I like...
        path_index=25 #Ok I think i like 25? 3 is my fav so far. path 1 is not too shabby, could work. doesn't land quite on the spiral. 
        dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[0, path_index, :], [0]))), radius=0.075)
        dot_to_move.set_color('#ED1C24')

        path_segments=VGroup()
        for k in range(64):
            segment1 = Line(
                axes.c2p(*[xt_history[k, path_index, 0], xt_history[k, path_index, 1]]), 
                axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]),
                stroke_width=5.0,
                stroke_color='#00AEEF' #'YELLOW' - Trying cyan in v2
            )
            segment2 = Line(
                axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]), 
                axes.c2p(*[xt_history[k+1, path_index, 0], xt_history[k+1, path_index, 1]]),
                stroke_width=5.0,
                stroke_color=BLACK, #CHILL_BROWN, 
            )
            segment2.set_opacity(0.9)
            segment1.set_opacity(0.8)
            path_segments.add(segment1)
            path_segments.add(segment2)
        self.add(path_segments) #Add now for layering. 
        path_segments.set_opacity(0.0)


        self.frame.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.25)
        self.add(axes)
        self.wait()
        self.play(ShowCreation(dots),
                  self.frame.animate.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.0), 
                  run_time=3.0)
        self.wait()

        self.play(self.frame.animate.reorient(0, 0, 0, (-1.54, 2.65, 0.0), 6.16),
                  run_time=3.0,
                  )
        self.add(dot_to_move)
        self.wait()

        a0=Arrow(dot_to_move.get_center(), 
                 dot_to_move.get_center()+np.array([2.5, -3.2, 0]), 
                 thickness=3.5,
                 tip_width_ratio=5)
        a0.set_color(YELLOW)
        self.play(FadeIn(a0))
        self.wait()
        self.play(FadeOut(a0))
        self.wait()

        dot_coords=Tex("("+str(round(xt_history[0, path_index, 0], 1))+', '+str(round(xt_history[0, path_index, 1], 1))+")",
                      font_size=32)
        dot_coords.next_to(dot_to_move, DOWN, buff=0.15)
        self.play(Write(dot_coords))
        self.wait()

        self.play(FadeIn(vector_field))
        self.wait()

        #Arrow here or cool variable opacity trail thin here? 
        # a1=Arrow(axes.c2p(*[xt_history[0, path_index, 0], xt_history[0, path_index, 1]]), 
        #          axes.c2p(*[history_pre_noise[0, path_index, 0], history_pre_noise[0, path_index, 1]]),
        #          thickness=3.5,
        #          tip_width_ratio=5)

        self.remove(dot_coords)
        self.play(dot_to_move.animate.move_to(axes.c2p(*[history_pre_noise[0, path_index, 0], 
                                                         history_pre_noise[0, path_index, 1]])),
                  ShowCreation(path_segments[0]),
                  path_segments[0].animate.set_opacity(0.8),
                  run_time=2.0)
        self.wait()

        self.play(dot_to_move.animate.move_to(axes.c2p(*[xt_history[1, path_index, 0], 
                                                         xt_history[1, path_index, 1]])),
                  ShowCreation(path_segments[1]),
                  path_segments[1].animate.set_opacity(0.5),
                  run_time=2.0)
        self.wait()

        #Book?

        # self.play(time_tracker.animate.set_value(8.0*(1.0/64.0)), run_time=0.5) #This move is really small, maybe roll it in and actually mention it a little later?

        #Might be nice to lower opacity on older segements as we go? We'll see. 
        # for k in range(1, 64):
        #     self.play(dot_to_move.animate.move_to(axes.c2p(*[history_pre_noise[k, path_index, 0], 
        #                                                      history_pre_noise[k, path_index, 1]])),
        #               ShowCreation(path_segments[2*k]),
        #               path_segments[2*k].animate.set_opacity(0.8),
        #               run_time=0.4)
        #     self.wait(0.1)

        #     self.play(dot_to_move.animate.move_to(axes.c2p(*[xt_history[k+1, path_index, 0], 
        #                                                      xt_history[k+1, path_index, 1]])),
        #               ShowCreation(path_segments[2*k+1]),
        #               path_segments[2*k+1].animate.set_opacity(0.5),
        #               run_time=0.4)
        #     # self.wait(0.1)   
        #     self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.1)
               

        self.wait()

        # Book first pause?
        # I need the layer stuff to I think...
        # Could spend more time in manim getting a few scenes going 
        # Let me do a little branching though, I don't wait to wait around on all this shit lol. 



        ## ok ok ok ok now zoom out, reset, add a bunch of particles and animate them all!
        ## Everthing in yellow or just to do rainbow hue vibes?
        ## Maybe try rainbow/hue first?
        ## Would be cool it we "landed on" the right colowheel arrangement on the spiral - I think that would
        ## be kinda tricky to code though actually - let me get into it and well see. 

        self.play(FadeOut(path_segments), FadeOut(dot_to_move), 
                  FadeOut(vector_field), 
                  self.frame.animate.reorient(0, 0, 0, (0.0, 0.0, 0.0), 10), 
                  run_time=4.0)
        self.wait()


        #50/50 if i like saturated colowheel colors, let's see how it feels in aggregate!
        num_dots=256 #Start small for testing and crank for final animation. 
        colors=get_color_wheel_colors(num_dots)
        all_path_segments=VGroup()
        all_dots_to_move=VGroup()
        colored_path_segments=VGroup()

        for path_index in range(num_dots): 
            dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[0, path_index, :], [0]))), radius=0.06)
            dot_to_move.set_color(colors[path_index])
            all_dots_to_move.add(dot_to_move)

            path_segments=VGroup()
            for k in range(64):
                segment1 = Line(
                    axes.c2p(*[xt_history[k, path_index, 0], xt_history[k, path_index, 1]]), 
                    axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]),
                    stroke_width=3.0,
                    stroke_color=colors[path_index]
                )
                segment2 = Line(
                    axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]), 
                    axes.c2p(*[xt_history[k+1, path_index, 0], xt_history[k+1, path_index, 1]]),
                    stroke_width=3.0,
                    stroke_color=colors[path_index], 
                )
                segment2.set_opacity(0.2)
                segment1.set_opacity(0.2)
                path_segments.add(segment1)
                path_segments.add(segment2)
                colored_path_segments.add(segment1)
            self.add(path_segments) #Add now for layering. 
            path_segments.set_opacity(0.0)
            all_path_segments.add(path_segments)

        self.wait()

        #Book
        # Eh not that cool - maybe just dots?
        # self.add(colored_path_segments)
        # colored_path_segments.set_opacity(1.0)

        #Ok my kinda gut right now is that I show 8-16 paths, but a bunch of final particle locations?





        # self.add(all_dots_to_move)

        self.play(FadeIn(all_dots_to_move))
        self.wait()
        # self.play(time_tracker.animate.set_value(0.0), run_time=0.1)
        time_tracker.set_value(0.0)
        # self.play(FadeIn(vector_field))
        self.wait()

        history_length=64
        # dots_indices_with_tails=[0, 30, 60, 90, 120, 150, 180, 210, 240, 255]
        dots_indices_with_tails = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255]
        for k in range(0,64):
            self.play(*[all_dots_to_move[path_index].animate.move_to(axes.c2p(*[history_pre_noise[k, path_index, 0], 
                                history_pre_noise[k, path_index, 1]])) for path_index in range(len(all_dots_to_move))], 
                      *[ShowCreation(all_path_segments[path_index][2*k]) for path_index in dots_indices_with_tails],
                      *[all_path_segments[path_index][2*k].animate.set_opacity(0.4) for path_index in dots_indices_with_tails],
                      *[all_path_segments[path_index][2*k-history_length].animate.set_opacity(0.4) for path_index in dots_indices_with_tails],
                      run_time=0.4)

            self.play(*[all_dots_to_move[path_index].animate.move_to(axes.c2p(*[xt_history[k+1, path_index, 0], 
                                xt_history[k+1, path_index, 1]])) for path_index in range(len(all_dots_to_move))], 
                      *[ShowCreation(all_path_segments[path_index][2*k+1]) for path_index in dots_indices_with_tails],
                      *[all_path_segments[path_index][2*k+1].animate.set_opacity(0.4) for path_index in dots_indices_with_tails],
                      *[all_path_segments[path_index][2*k+1-history_length].animate.set_opacity(0.4) for path_index in dots_indices_with_tails],
                      run_time=0.4)

            self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.2)
        self.wait()


        #Ok now does this go straight into the noise free variant?



        self.play(FadeOut(all_path_segments))
        # self.wait()
        self.play(FadeOut(all_dots_to_move))
        # self.wait()

        # self.play(time_tracker.animate.set_value(0.0), run_time=1.0)
        time_tracker.set_value(0.0)

        xt_history=np.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/ddpm_no_noise_1.npy')


        # num_dots=16 #Start small for testing and crank for final animation. 
        num_dots=96 #256 is kinda overwhelming here
        colors=get_color_wheel_colors(num_dots)
        all_traced_paths=VGroup()
        all_dots_to_move=VGroup()
        for path_index in range(num_dots): 
            dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[0, path_index, :], [0]))), radius=0.06)
            dot_to_move.set_color(colors[path_index])
            all_dots_to_move.add(dot_to_move)

            traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=colors[path_index], stroke_width=3.0, 
                                          opacity_range=(0.4, 0.4), fade_length=24) #Tryin opaicty 0 and longer fade length in v2
            # traced_path.set_opacity(0.5)
            # traced_path.set_fill(opacity=0)
            all_traced_paths.add(traced_path)
        self.add(all_traced_paths)

        # self.wait()

        # self.play(FadeIn(all_dots_to_move), self.frame.animate.reorient(0, 0, 0, (-0.06, 0.01, 0.0), 7.10), run_time=3.0)
        self.wait()

        for k in range(64):
            self.play(time_tracker.animate.set_value(8.0*(k/64.0)), 
                      *[all_dots_to_move[path_index].animate.move_to(axes.c2p(*[xt_history[k, path_index, 0], 
                                                                                xt_history[k, path_index, 1]])) for path_index in range(len(all_dots_to_move))],
                     rate_func=linear, run_time=0.2)
            # for traced_path in all_traced_paths:
            #     traced_path.update_path(0.1) 

            # self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.1)

        self.wait()
        #BOook?

        self.frame.reorient(0, 0, 0, (-0.23, 0.03, 0.0), 7.34)
        self.wait()

        self.remove(axes)
        self.frame.reorient(0, 0, 0, (-0.39, -0.12, 0.0), 7.81)
        self.wait()

        self.frame.reorient(0, 0, 0, (-0.26, -0.33, 0.0), 6.40)
        self.wait()


        self.remove(all_traced_paths)
        self.wait()


        # self.wait()
        # self.play(FadeOut(all_traced_paths), FadeOut(vector_field), FadeOut(axes),
        #           self.frame.animate.reorient(0, 0, 0, (-0.11, -0.32, 0.0), 6.34), 
        #           run_time=2.5)
        # self.wait()

        # #Ok so to get me into p57, I think i just want to go back to that sam path I showed at the beggining of 40?
        # # self.play(FadeIn(axes), FadeOut(all_dots_to_move))

        # time_tracker.set_value(8.0)
        # time_tracker.set_value(0.0) #Doesn't seem liek this is taking?
        # self.play(FadeIn(axes), 
        #           FadeOut(all_dots_to_move), 
        #           # FadeIn(vector_field), #Lets actually fade in the vector field in p57
        #           self.frame.animate.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.25), run_time=3.0)
        # self.wait()

        self.embed()


class follow_trajectory_1(InteractiveScene):
    def construct(self):
        '''
        Ok ok ok need to do a direct transition from p47b after fading out all the traces etc -> then bring
        in the full vector field - I think this is going to be dope!
        '''
        batch_size=2130
        dataset = Swissroll(np.pi/2, 5*np.pi, 100)
        loader = DataLoader(dataset, batch_size=batch_size)
        batch=next(iter(loader)).numpy()

        axes = Axes(
            x_range=[-1.2, 1.2, 0.5],
            y_range=[-1.2, 1.2, 0.5],
            height=7,
            width=7,
            axis_config={
                "color": CHILL_BROWN, 
                "stroke_width": 2,
                "include_tip": True,
                "include_ticks": False,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )

        axes.set_opacity(0.5)

        # Create extended axes with SAME center point and proportional scaling
        extended_axes = Axes(
            x_range=[-2.0, 2.0, 0.5],    # Extended range
            y_range=[-2.0, 2.0, 0.5],    # Extended range
            height=7 * (4.0/2.4),        # Scale height proportionally: original_height * (new_range/old_range)
            width=7 * (4.0/2.4),         # Scale width proportionally: original_width * (new_range/old_range)
            axis_config={"stroke_width": 0}  # Make invisible
        )

        # Move extended axes to same position as original axes
        extended_axes.move_to(axes.get_center())


        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.05)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(CHILL_BROWN)
        dots.set_opacity(0.5)


        # model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_20_1.pt') #Trained on 256 levels
        model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_24_1.pt') #Trained on 64 levels

        schedule = ScheduleLogLinear(N=64, sigma_min=0.01, sigma_max=1) #N=200
        bound=2.0 #Need to match extended axes bro
        num_heatmap_steps=30
        grid=[]
        for i, x in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
            for j, y in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
                grid.append([x,y])
        grid=torch.tensor(grid).float()

        gam=1
        mu=0.5 #0.5 is DDPM
        cfg_scale=0.0
        cond=None
        sigmas=schedule.sample_sigmas(64)
        xt_history=[]
        history_pre_noise=[]
        heatmaps=[]
        eps=None
        torch.manual_seed(2)

        with torch.no_grad():
            model.eval();
            xt=torch.randn((batch_size,) + model.input_dims)*sigmas[0] #Scaling by sigma here matters a lot - why is that???
            xt_history.append(xt.numpy())
            
            for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
                eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
                # eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
                sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
                eta = (sig_prev**2 - sig_p**2).sqrt()
                history_pre_noise.append(xt - (sig - sig_p) * eps)
                xt = xt - (sig - sig_p) * eps + eta * model.rand_input(xt.shape[0]).to(xt) #Straight remove adding random noise
                
                xt_history.append(xt.numpy())
                heatmaps.append(model.forward(grid, sig, cond=None))
        xt_history=np.array(xt_history)
        history_pre_noise=np.array(history_pre_noise)


        time_tracker = ValueTracker(0.0)  # Start at time 0

        def vector_function_with_tracker(coords_array):
            """Vector function that uses the ValueTracker for time"""
            current_time = time_tracker.get_value()
            max_time = 8.0  # Map time 0-8 to sigma indices 0-255
            sigma_idx = int(np.clip(current_time * 63 / max_time, 0, 63)) #Needs to be N-1
            
            try:
                res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=None)
                return -res.detach().numpy()
            except:
                return np.zeros((len(coords_array), 2))


        vector_field = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_with_tracker,
            coordinate_system=extended_axes,
            density=5.0,
            stroke_width=2.0,
            max_radius=6.0,      # Vectors fade to min_opacity at this distance
            min_opacity=0.5,     # Minimum opacity at max_radius
            max_opacity=0.5,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=CHILL_BROWN
        )
        

        # Ok so I'll need to noodle with a few different starting points - and am tempted ot start not quite at point 100, ya know?
        #Ok yeah so I need to find path I like...
        path_index=25 #Ok I think i like 25? 3 is my fav so far. path 1 is not too shabby, could work. doesn't land quite on the spiral. 
        dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[0, path_index, :], [0]))), radius=0.075)
        dot_to_move.set_color('#ED1C24')

        path_segments=VGroup()
        for k in range(64):
            segment1 = Line(
                axes.c2p(*[xt_history[k, path_index, 0], xt_history[k, path_index, 1]]), 
                axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]),
                stroke_width=5.0,
                stroke_color='#00AEEF' #'YELLOW' - Trying cyan in v2
            )
            segment2 = Line(
                axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]), 
                axes.c2p(*[xt_history[k+1, path_index, 0], xt_history[k+1, path_index, 1]]),
                stroke_width=5.0,
                stroke_color=BLACK, #CHILL_BROWN, 
            )
            segment2.set_opacity(0.9)
            segment1.set_opacity(0.8)
            path_segments.add(segment1)
            path_segments.add(segment2)
        self.add(path_segments) #Add now for layering. 
        path_segments.set_opacity(0.0)


        self.frame.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.25)
        self.add(axes)
        self.wait()
        self.play(ShowCreation(dots),
                  self.frame.animate.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.0), 
                  run_time=3.0)
        self.wait()

        self.play(self.frame.animate.reorient(0, 0, 0, (-1.54, 2.65, 0.0), 6.16),
                  run_time=3.0,
                  )
        self.add(dot_to_move)
        self.wait()

        a0=Arrow(dot_to_move.get_center(), 
                 dot_to_move.get_center()+np.array([2.5, -3.2, 0]), 
                 thickness=3.5,
                 tip_width_ratio=5)
        a0.set_color(YELLOW)
        self.play(FadeIn(a0))
        self.wait()
        self.play(FadeOut(a0))
        self.wait()

        dot_coords=Tex("("+str(round(xt_history[0, path_index, 0], 1))+', '+str(round(xt_history[0, path_index, 1], 1))+")",
                      font_size=32)
        dot_coords.next_to(dot_to_move, DOWN, buff=0.15)
        self.play(Write(dot_coords))
        self.wait()

        self.play(FadeIn(vector_field))
        self.wait()

        #Arrow here or cool variable opacity trail thin here? 
        # a1=Arrow(axes.c2p(*[xt_history[0, path_index, 0], xt_history[0, path_index, 1]]), 
        #          axes.c2p(*[history_pre_noise[0, path_index, 0], history_pre_noise[0, path_index, 1]]),
        #          thickness=3.5,
        #          tip_width_ratio=5)

        self.remove(dot_coords)
        self.play(dot_to_move.animate.move_to(axes.c2p(*[history_pre_noise[0, path_index, 0], 
                                                         history_pre_noise[0, path_index, 1]])),
                  ShowCreation(path_segments[0]),
                  path_segments[0].animate.set_opacity(0.8),
                  run_time=2.0)
        self.wait()

        self.play(dot_to_move.animate.move_to(axes.c2p(*[xt_history[1, path_index, 0], 
                                                         xt_history[1, path_index, 1]])),
                  ShowCreation(path_segments[1]),
                  path_segments[1].animate.set_opacity(0.5),
                  run_time=2.0)
        self.wait()

        #Book?

        # self.play(time_tracker.animate.set_value(8.0*(1.0/64.0)), run_time=0.5) #This move is really small, maybe roll it in and actually mention it a little later?

        #Might be nice to lower opacity on older segements as we go? We'll see. 
        for k in range(1, 64):
            self.play(dot_to_move.animate.move_to(axes.c2p(*[history_pre_noise[k, path_index, 0], 
                                                             history_pre_noise[k, path_index, 1]])),
                      ShowCreation(path_segments[2*k]),
                      path_segments[2*k].animate.set_opacity(0.8),
                      run_time=0.4)
            self.wait(0.1)

            self.play(dot_to_move.animate.move_to(axes.c2p(*[xt_history[k+1, path_index, 0], 
                                                             xt_history[k+1, path_index, 1]])),
                      ShowCreation(path_segments[2*k+1]),
                      path_segments[2*k+1].animate.set_opacity(0.5),
                      run_time=0.4)
            # self.wait(0.1)   
            self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.1)
               

        self.wait()
        self.embed()



class p57_58v4(InteractiveScene):
    def construct(self):

        #Smooth transition/pickup from spiral and axes in p56
        batch_size=2130
        dataset = Swissroll(np.pi/2, 5*np.pi, 100)
        loader = DataLoader(dataset, batch_size=batch_size)
        batch=next(iter(loader)).numpy()

        axes = Axes(
            x_range=[-1.2, 1.2, 0.5],
            y_range=[-1.2, 1.2, 0.5],
            height=7,
            width=7,
            axis_config={
                "color": CHILL_BROWN, 
                "stroke_width": 2,
                "include_tip": True,
                "include_ticks": False,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )
        axes.set_opacity(0.8)

        # Create extended axes with SAME center point and proportional scaling
        extended_axes = Axes(
            x_range=[-2.0, 2.0, 0.5],    # Extended range
            y_range=[-2.0, 2.0, 0.5],    # Extended range
            height=7 * (4.0/2.4),        # Scale height proportionally: original_height * (new_range/old_range)
            width=7 * (4.0/2.4),         # Scale width proportionally: original_width * (new_range/old_range)
            axis_config={"stroke_width": 0}  # Make invisible
        )

        # Move extended axes to same position as original axes
        extended_axes.move_to(axes.get_center())


        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(YELLOW)
        dots.set_opacity(0.3)


        # Hmm can i use the same forward path I used earlier, and then show the reversal of it?
        # I think that might work pretty well? Script wise I'll probably want to bring the vector field and and then take 
        # it out maybe when I zoom in,  
        # Ok yeah iterating through script and visuals here -> i think it's like fade in vector field, remove it, and 
        # then play that same forward diffusion process, then like one step of reverse process, and then maybe full
        # reverse process. 


        #Forward Process I want to show the reversal of
        i=75
        dot_to_move=dots[i].copy()
        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=2.0, 
                                      opacity_range=(0.25, 0.9), fade_length=15)
        # traced_path.set_opacity(0.5)
        traced_path.set_fill(opacity=0)


        np.random.seed(485) #485 is nice, 4 is maybe best so far, #52 is ok
        random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
        random_walk[0]=np.array([0.2, 0.12]) #make first step go up and to the right
        # random_walk[-1]=np.array([0.15, -0.04])
        random_walk[-1]=np.array([0.19, -0.05])
        random_walk=np.cumsum(random_walk,axis=0) 

        random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        
        dot_history=VGroup()
        dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
        # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)
        # dot_to_move.move_to(axes.c2p(*random_walk_shifted[1]))
        traced_path.update_path(0.1)


        # model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_20_1.pt') #Trained on 256 levels
        model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_24_1.pt') #Trained on 64 levels

        schedule = ScheduleLogLinear(N=64, sigma_min=0.01, sigma_max=1) #N=200
        bound=2.0 #Need to match extended axes bro
        num_heatmap_steps=30
        grid=[]
        for i, x in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
            for j, y in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
                grid.append([x,y])
        grid=torch.tensor(grid).float()

        gam=1
        mu=0.5 #0.5 is DDPM
        cfg_scale=0.0
        cond=None
        sigmas=schedule.sample_sigmas(64)
        xt_history=[]
        history_pre_noise=[]
        heatmaps=[]
        eps=None
        torch.manual_seed(2)

        with torch.no_grad():
            model.eval();
            xt=torch.randn((batch_size,) + model.input_dims)*sigmas[0] #Scaling by sigma here matters a lot - why is that???
            #xt.shape is torch.Size([2130, 2])  
            xt[0,0]=random_walk_shifted[-1][0] #Copy over end of test path into first path to reverse. 
            xt[0,1]=random_walk_shifted[-1][1]
            xt_history.append(xt.numpy())
            
            for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
                eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
                # eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
                sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
                eta = (sig_prev**2 - sig_p**2).sqrt()
                history_pre_noise.append(xt - (sig - sig_p) * eps)
                xt = xt - (sig - sig_p) * eps + eta * model.rand_input(xt.shape[0]).to(xt) #Straight remove adding random noise
                
                xt_history.append(xt.numpy())
                heatmaps.append(model.forward(grid, sig, cond=None))
        xt_history=np.array(xt_history)
        history_pre_noise=np.array(history_pre_noise)

        time_tracker = ValueTracker(0.0)  # Start at time 0

        def vector_function_with_tracker(coords_array):
            """Vector function that uses the ValueTracker for time"""
            current_time = time_tracker.get_value()
            max_time = 8.0  # Map time 0-8 to sigma indices 0-255
            sigma_idx = int(np.clip(current_time * 63 / max_time, 0, 63)) #Needs to be N-1
            
            try:
                res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=None)
                return -res.detach().numpy()
            except:
                return np.zeros((len(coords_array), 2))


        # Create the tracker-controlled vector field
        vector_field = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_with_tracker,
            coordinate_system=extended_axes,
            density=3.0,
            stroke_width=2,
            max_radius=6.0,      # Vectors fade to min_opacity at this distance
            min_opacity=0.15,     # Minimum opacity at max_radius
            max_opacity=1.0,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=WHITE
        )
        


        self.wait()
        self.frame.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.25)
        self.add(axes, dots)
        self.wait()
        self.play(FadeIn(vector_field)) #, self.frame.animate.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.0))
        self.wait()


        path_index=0 
        dot_to_move_2=Dot(axes.c2p(*np.concatenate((xt_history[0, path_index, :], [0]))), radius=0.06)
        dot_to_move_2.set_color(WHITE)

        path_segments=VGroup()
        for k in range(64):
            segment1 = Line(
                axes.c2p(*[xt_history[k, path_index, 0], xt_history[k, path_index, 1]]), 
                axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]),
                stroke_width=4.0,
                stroke_color=YELLOW
            )
            segment2 = Line(
                axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]), 
                axes.c2p(*[xt_history[k+1, path_index, 0], xt_history[k+1, path_index, 1]]),
                stroke_width=4.0,
                stroke_color=WHITE, 
            )
            segment2.set_opacity(0.4)
            segment1.set_opacity(0.9)
            path_segments.add(segment1)
            path_segments.add(segment2)
        self.add(path_segments) #Add now for layering. 
        path_segments.set_opacity(0.0)


        self.wait()

        # self.frame.animate.reorient(0, 0, 0, (2.92, 1.65, 0.0), 4.19)
        self.play(FadeOut(vector_field))
        dot_to_move.set_opacity(1.0)
        self.add(dot_to_move, traced_path)

        start_orientation=[0, 0, 0, (0.00, 0.00, 0.0), 8.25]
        # end_orientation=[0, 0, 0, (2.92, 1.65, 0.0), 4.19]
        end_orientation=[0, 0, 0, (3.48, 1.88, 0.0), 4.26]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, 100)

        self.wait()
        for j in range(100):
            dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
            dot_to_move.move_to(axes.c2p(*random_walk_shifted[j]))
            traced_path.update_path(0.1)
            self.frame.reorient(*interp_orientations[j])
            self.wait(0.1)
            # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)
        traced_path.stop_tracing()


        x100=Tex('x_{100}', font_size=24).set_color(YELLOW)
        x100.next_to(dot_to_move, 0.07*UP+0.001*RIGHT)

        x99=Tex('x_{99}', font_size=24).set_color(CHILL_BROWN)
        x99.next_to(dot_history[-1], 0.1*UP+0.01*RIGHT)
        dot99=Dot(dot_history[-1].get_center(), radius=0.04)
        dot99.set_color(CHILL_BROWN)

        self.play(FadeIn(x100), FadeIn(x99), FadeIn(dot99))
        self.wait()

        a1=Arrow(dot_history[-1].get_center(),
                 dot_to_move.get_center(),
                 thickness = 2.0,
                 tip_width_ratio= 5, 
                 buff=0.03)
        a1.set_color(YELLOW)

        #Turn path to chiil brown when I do the reversal??
        # I think I want an arrow on that last step here?

        eq_1=Tex("p(x_{100} | x_{99}) = \mathcal{N} (0, \sigma^2)", font_size=22)
        eq_1.set_color('#FFFFFF')
        eq_1[2:6].set_color(YELLOW)
        eq_1[7:10].set_color(CHILL_BROWN)
        eq_1.move_to([5.2, 3.65, 0])
        self.add(eq_1, a1)
        

        eq_2=Tex("p(x_{99} | x_{100}) = \mathcal{N} (\mu, \sigma^2)", font_size=22)
        eq_2.set_color('#FFFFFF')
        eq_2[2:5].set_color(CHILL_BROWN)
        eq_2[6:10].set_color(YELLOW)
        eq_2[14].set_color('#00AEEF')
        eq_2.move_to([5.2, 2.3, 0])       

        pre_point_coords=dot_to_move.get_center()-np.array([0.6, 0.18, 0])
        a2=Arrow(dot_to_move.get_center(),
                 pre_point_coords,
                 thickness = 2.0,
                 tip_width_ratio= 5, 
                 buff=0.035)
        a2.set_color('#00AEEF')

        self.wait()
        self.play(ReplacementTransform(a1, a2), 
                  traced_path.animate.set_color(CHILL_BROWN).set_opacity(0.2), 
                  x99.animate.set_opacity(0.5), 
                  dot99.animate.set_opacity(0.25))
        self.wait()

        self.play(ReplacementTransform(eq_1[:2].copy(), eq_2[:2]), 
                  ReplacementTransform(eq_1[2:6].copy(), eq_2[6:10]),
                  ReplacementTransform(eq_1[7:10].copy(), eq_2[2:5]),
                  ReplacementTransform(eq_1[6].copy(), eq_2[5]),
                  ReplacementTransform(eq_1[10:].copy(), eq_2[10:]),
                  run_time=4)
        # Ok now add in little overlay from illustrator here
        # And I think i know how i want to finish the scent
        # Totally remove forward path, add in next (random) step of reverse path
        # And I think add labels -> mu and a zero mean normal to the two steps/arrows
        # From there maybe one more set of illustrator label
        # I don' think it makes sense to play the full reverse path here. 

        dot2=Dot(pre_point_coords, radius=0.04)
        dot2.set_color('#00AEEF')
        a3=Arrow(pre_point_coords,
                 pre_point_coords+np.array([-0.7, 0.24, 0]),
                 thickness = 2.0,
                 tip_width_ratio= 5, 
                 buff=0.035)
        a3.set_color('#777777')

        self.wait()
        self.play(FadeOut(traced_path), FadeOut(x99), FadeOut(dot99))
        self.add(a3, dot2)
        self.wait()

        # mu_label=Tex('\mu', font_size=22)
        # mu_label.set_color('#00AEEF')
        # mu_label.next_to(a2, DOWN, buff=0.1)
        mu_label=eq_2[14].copy()

        self.play(mu_label.animate.move_to(a2.get_center()+0.16*DOWN+0.05*RIGHT), run_time=2.0)
        self.wait()

        eq_3=Tex("\mathcal{N} (0, \sigma^2)", font_size=18)
        eq_3.set_color('#777777')
        eq_3.move_to(a3.get_center()+0.22*DOWN+0.2*LEFT)

        self.play(ReplacementTransform(eq_2[12:].copy(), eq_3), run_time=2.0)
        self.wait()

        self.play(FadeOut(eq_1), FadeOut(eq_1), FadeOut(eq_2), FadeOut(eq_3), 
                  FadeOut(a2), FadeOut(a3), FadeOut(dot2), FadeOut(dot_to_move),
                  FadeOut(mu_label), FadeOut(x100))

        #Now fade in vector field while we zoom out!
        self.add(vector_field)
        self.play(time_tracker.animate.set_value(8.0), run_time=2.0)
        self.remove(vector_field)


        self.play(FadeIn(vector_field), self.frame.animate.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.0), run_time=4.0)
        self.wait()

        self.play(time_tracker.animate.set_value(0.0), run_time=8.0)
        self.wait()



        self.wait(20)
        self.embed()


class book_vector_field_progression_1(InteractiveScene):
    def construct(self):
        '''
        Ok ok ok need to do a direct transition from p47b after fading out all the traces etc -> then bring
        in the full vector field - I think this is going to be dope!
        '''
        batch_size=2130
        dataset = Swissroll(np.pi/2, 5*np.pi, 100)
        loader = DataLoader(dataset, batch_size=batch_size)
        batch=next(iter(loader)).numpy()

        axes = Axes(
            x_range=[-1.2, 1.2, 0.5],
            y_range=[-1.2, 1.2, 0.5],
            height=7,
            width=7,
            axis_config={
                "color": CHILL_BROWN, 
                "stroke_width": 2,
                "include_tip": True,
                "include_ticks": False,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )

        axes.set_opacity(0.5)

        # Create extended axes with SAME center point and proportional scaling
        extended_axes = Axes(
            x_range=[-2.0, 2.0, 0.5],    # Extended range
            y_range=[-2.0, 2.0, 0.5],    # Extended range
            height=7 * (4.0/2.4),        # Scale height proportionally: original_height * (new_range/old_range)
            width=7 * (4.0/2.4),         # Scale width proportionally: original_width * (new_range/old_range)
            axis_config={"stroke_width": 0}  # Make invisible
        )

        # Move extended axes to same position as original axes
        extended_axes.move_to(axes.get_center())


        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.05)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(CHILL_BROWN)
        dots.set_opacity(0.5)


        # model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_20_1.pt') #Trained on 256 levels
        model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_24_1.pt') #Trained on 64 levels

        schedule = ScheduleLogLinear(N=64, sigma_min=0.01, sigma_max=1) #N=200
        bound=2.0 #Need to match extended axes bro
        num_heatmap_steps=30
        grid=[]
        for i, x in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
            for j, y in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
                grid.append([x,y])
        grid=torch.tensor(grid).float()

        gam=1
        mu=0.5 #0.5 is DDPM
        cfg_scale=0.0
        cond=None
        sigmas=schedule.sample_sigmas(64)
        xt_history=[]
        history_pre_noise=[]
        heatmaps=[]
        eps=None
        torch.manual_seed(2)

        with torch.no_grad():
            model.eval();
            xt=torch.randn((batch_size,) + model.input_dims)*sigmas[0] #Scaling by sigma here matters a lot - why is that???
            xt_history.append(xt.numpy())
            
            for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
                eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
                # eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
                sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
                eta = (sig_prev**2 - sig_p**2).sqrt()
                history_pre_noise.append(xt - (sig - sig_p) * eps)
                xt = xt - (sig - sig_p) * eps + eta * model.rand_input(xt.shape[0]).to(xt) #Straight remove adding random noise
                
                xt_history.append(xt.numpy())
                heatmaps.append(model.forward(grid, sig, cond=None))
        xt_history=np.array(xt_history)
        history_pre_noise=np.array(history_pre_noise)


        time_tracker = ValueTracker(0.0)  # Start at time 0

        def vector_function_with_tracker(coords_array):
            """Vector function that uses the ValueTracker for time"""
            current_time = time_tracker.get_value()
            max_time = 8.0  # Map time 0-8 to sigma indices 0-255
            sigma_idx = int(np.clip(current_time * 63 / max_time, 0, 63)) #Needs to be N-1
            
            try:
                res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=None)
                return -res.detach().numpy()
            except:
                return np.zeros((len(coords_array), 2))


        # Create the tracker-controlled vector field
        vector_field = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_with_tracker,
            coordinate_system=extended_axes,
            density=5.0,
            stroke_width=1.5,
            max_radius=6.0,      # Vectors fade to min_opacity at this distance
            min_opacity=0.6,     # Minimum opacity at max_radius
            max_opacity=0.6,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=CHILL_BROWN
        )
        
        #Book pause here or too early?
        self.add(axes)
        self.add(dots)
        self.frame.reorient(0, 0, 0, (0.24, -0.02, 0.0), 12.18)
        self.add(vector_field)
        self.wait()
        # Ok yeah that's not bad. 
        # Now i want like t=1 right?

        time_tracker.set_value(0)
        self.wait()

        for k in range(64):
            self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.2)
            self.wait(0.2)

        self.wait(5)
        self.embed()

        # k=63
        # time_tracker.set_value(8.0*(k/64.0))
        self.wait()


class p53_56v4b(InteractiveScene):
    def construct(self):
        '''
        Ok ok ok need to do a direct transition from p47b after fading out all the traces etc -> then bring
        in the full vector field - I think this is going to be dope!
        '''
        batch_size=2130
        dataset = Swissroll(np.pi/2, 5*np.pi, 100)
        loader = DataLoader(dataset, batch_size=batch_size)
        batch=next(iter(loader)).numpy()

        axes = Axes(
            x_range=[-1.2, 1.2, 0.5],
            y_range=[-1.2, 1.2, 0.5],
            height=7,
            width=7,
            axis_config={
                "color": CHILL_BROWN, 
                "stroke_width": 2,
                "include_tip": True,
                "include_ticks": False,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )

        axes.set_opacity(0.5)

        # Create extended axes with SAME center point and proportional scaling
        extended_axes = Axes(
            x_range=[-2.0, 2.0, 0.5],    # Extended range
            y_range=[-2.0, 2.0, 0.5],    # Extended range
            height=7 * (4.0/2.4),        # Scale height proportionally: original_height * (new_range/old_range)
            width=7 * (4.0/2.4),         # Scale width proportionally: original_width * (new_range/old_range)
            axis_config={"stroke_width": 0}  # Make invisible
        )

        # Move extended axes to same position as original axes
        extended_axes.move_to(axes.get_center())


        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.05)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(CHILL_BROWN)
        dots.set_opacity(0.5)


        # model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_20_1.pt') #Trained on 256 levels
        model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_24_1.pt') #Trained on 64 levels

        schedule = ScheduleLogLinear(N=64, sigma_min=0.01, sigma_max=1) #N=200
        bound=2.0 #Need to match extended axes bro
        num_heatmap_steps=30
        grid=[]
        for i, x in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
            for j, y in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
                grid.append([x,y])
        grid=torch.tensor(grid).float()

        gam=1
        mu=0.5 #0.5 is DDPM
        cfg_scale=0.0
        cond=None
        sigmas=schedule.sample_sigmas(64)
        xt_history=[]
        history_pre_noise=[]
        heatmaps=[]
        eps=None
        torch.manual_seed(2)

        with torch.no_grad():
            model.eval();
            xt=torch.randn((batch_size,) + model.input_dims)*sigmas[0] #Scaling by sigma here matters a lot - why is that???
            xt_history.append(xt.numpy())
            
            for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
                eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
                # eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
                sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
                eta = (sig_prev**2 - sig_p**2).sqrt()
                history_pre_noise.append(xt - (sig - sig_p) * eps)
                xt = xt - (sig - sig_p) * eps + eta * model.rand_input(xt.shape[0]).to(xt) #Straight remove adding random noise
                
                xt_history.append(xt.numpy())
                heatmaps.append(model.forward(grid, sig, cond=None))
        xt_history=np.array(xt_history)
        history_pre_noise=np.array(history_pre_noise)


        time_tracker = ValueTracker(0.0)  # Start at time 0

        def vector_function_with_tracker(coords_array):
            """Vector function that uses the ValueTracker for time"""
            current_time = time_tracker.get_value()
            max_time = 8.0  # Map time 0-8 to sigma indices 0-255
            sigma_idx = int(np.clip(current_time * 63 / max_time, 0, 63)) #Needs to be N-1
            
            try:
                res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=None)
                return -res.detach().numpy()
            except:
                return np.zeros((len(coords_array), 2))


        # Create the tracker-controlled vector field
        vector_field = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_with_tracker,
            coordinate_system=extended_axes,
            density=5.0,
            stroke_width=1.5,
            max_radius=6.0,      # Vectors fade to min_opacity at this distance
            min_opacity=0.6,     # Minimum opacity at max_radius
            max_opacity=0.6,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=CHILL_BROWN
        )
        
        #Book pause here or too early?
        # self.add(axes)
        self.add(dots)
        self.frame.reorient(0, 0, 0, (0.24, -0.02, 0.0), 12.18)
        self.add(vector_field)
        self.wait()
        # Ok yeah that's not bad. 
        # Now i want like t=1 right?
        k=63
        time_tracker.set_value(0)
        self.wait()

        for k in range(64):
            self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.2)
            self.wait(0.2)

        self.wait(5)
        self.embed()


        # # Ok so I'll need to noodle with a few different starting points - and am tempted ot start not quite at point 100, ya know?
        # #Ok yeah so I need to find path I like...
        # path_index=25 #Ok I think i like 25? 3 is my fav so far. path 1 is not too shabby, could work. doesn't land quite on the spiral. 
        # dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[0, path_index, :], [0]))), radius=0.06)
        # dot_to_move.set_color(WHITE)

        # path_segments=VGroup()
        # for k in range(64):
        #     segment1 = Line(
        #         axes.c2p(*[xt_history[k, path_index, 0], xt_history[k, path_index, 1]]), 
        #         axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]),
        #         stroke_width=4.0,
        #         stroke_color='#00AEEF' #'YELLOW' - Trying cyan in v2
        #     )
        #     segment2 = Line(
        #         axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]), 
        #         axes.c2p(*[xt_history[k+1, path_index, 0], xt_history[k+1, path_index, 1]]),
        #         stroke_width=4.0,
        #         stroke_color=WHITE, 
        #     )
        #     segment2.set_opacity(0.4)
        #     segment1.set_opacity(0.9)
        #     path_segments.add(segment1)
        #     path_segments.add(segment2)
        # self.add(path_segments) #Add now for layering. 
        # path_segments.set_opacity(0.0)


        # self.frame.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.25)
        # self.add(axes)
        # self.wait()
        # self.play(ShowCreation(dots),
        #           self.frame.animate.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.0), 
        #           run_time=3.0)
        # self.wait()

        # self.play(self.frame.animate.reorient(0, 0, 0, (-1.54, 2.65, 0.0), 6.16),
        #           run_time=3.0,
        #           )
        # self.add(dot_to_move)
        # self.wait()

        # a0=Arrow(dot_to_move.get_center(), 
        #          dot_to_move.get_center()+np.array([2.5, -3.2, 0]), 
        #          thickness=3.5,
        #          tip_width_ratio=5)
        # a0.set_color(YELLOW)
        # self.play(FadeIn(a0))
        # self.wait()
        # self.play(FadeOut(a0))
        # self.wait()

        # dot_coords=Tex("("+str(round(xt_history[0, path_index, 0], 1))+', '+str(round(xt_history[0, path_index, 1], 1))+")",
        #               font_size=32)
        # dot_coords.next_to(dot_to_move, DOWN, buff=0.15)
        # self.play(Write(dot_coords))
        # self.wait()

        # self.play(FadeIn(vector_field))
        # self.wait()

        # #Arrow here or cool variable opacity trail thin here? 
        # # a1=Arrow(axes.c2p(*[xt_history[0, path_index, 0], xt_history[0, path_index, 1]]), 
        # #          axes.c2p(*[history_pre_noise[0, path_index, 0], history_pre_noise[0, path_index, 1]]),
        # #          thickness=3.5,
        # #          tip_width_ratio=5)

        # self.remove(dot_coords)
        # self.play(dot_to_move.animate.move_to(axes.c2p(*[history_pre_noise[0, path_index, 0], 
        #                                                  history_pre_noise[0, path_index, 1]])),
        #           ShowCreation(path_segments[0]),
        #           path_segments[0].animate.set_opacity(0.8),
        #           run_time=2.0)
        # self.wait()

        # self.play(dot_to_move.animate.move_to(axes.c2p(*[xt_history[1, path_index, 0], 
        #                                                  xt_history[1, path_index, 1]])),
        #           ShowCreation(path_segments[1]),
        #           path_segments[1].animate.set_opacity(0.5),
        #           run_time=2.0)
        # self.wait()

        # # self.play(time_tracker.animate.set_value(8.0*(1.0/64.0)), run_time=0.5) #This move is really small, maybe roll it in and actually mention it a little later?

        # #Might be nice to lower opacity on older segements as we go? We'll see. 
        # for k in range(1, 64):
        #     self.play(dot_to_move.animate.move_to(axes.c2p(*[history_pre_noise[k, path_index, 0], 
        #                                                      history_pre_noise[k, path_index, 1]])),
        #               ShowCreation(path_segments[2*k]),
        #               path_segments[2*k].animate.set_opacity(0.8),
        #               run_time=0.4)
        #     self.wait(0.1)

        #     self.play(dot_to_move.animate.move_to(axes.c2p(*[xt_history[k+1, path_index, 0], 
        #                                                      xt_history[k+1, path_index, 1]])),
        #               ShowCreation(path_segments[2*k+1]),
        #               path_segments[2*k+1].animate.set_opacity(0.5),
        #               run_time=0.4)
        #     # self.wait(0.1)   
        #     self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.1)
               

        # self.wait()

        # ## ok ok ok ok now zoom out, reset, add a bunch of particles and animate them all!
        # ## Everthing in yellow or just to do rainbow hue vibes?
        # ## Maybe try rainbow/hue first?
        # ## Would be cool it we "landed on" the right colowheel arrangement on the spiral - I think that would
        # ## be kinda tricky to code though actually - let me get into it and well see. 

        # self.play(FadeOut(path_segments), FadeOut(dot_to_move), 
        #           FadeOut(vector_field), 
        #           self.frame.animate.reorient(0, 0, 0, (0.0, 0.0, 0.0), 10), 
        #           run_time=4.0)
        # self.wait()


        # #50/50 if i like saturated colowheel colors, let's see how it feels in aggregate!
        # num_dots=256 #Start small for testing and crank for final animation. 
        # colors=get_color_wheel_colors(num_dots)
        # all_path_segments=VGroup()
        # all_dots_to_move=VGroup()

        # for path_index in range(num_dots): 
        #     dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[0, path_index, :], [0]))), radius=0.06)
        #     dot_to_move.set_color(colors[path_index])
        #     all_dots_to_move.add(dot_to_move)

        #     path_segments=VGroup()
        #     for k in range(64):
        #         segment1 = Line(
        #             axes.c2p(*[xt_history[k, path_index, 0], xt_history[k, path_index, 1]]), 
        #             axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]),
        #             stroke_width=3.0,
        #             stroke_color=colors[path_index]
        #         )
        #         segment2 = Line(
        #             axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]), 
        #             axes.c2p(*[xt_history[k+1, path_index, 0], xt_history[k+1, path_index, 1]]),
        #             stroke_width=3.0,
        #             stroke_color=WHITE, 
        #         )
        #         segment2.set_opacity(0.4)
        #         segment1.set_opacity(0.9)
        #         path_segments.add(segment1)
        #         path_segments.add(segment2)
        #     self.add(path_segments) #Add now for layering. 
        #     path_segments.set_opacity(0.0)
        #     all_path_segments.add(path_segments)

        # self.wait()

        # # self.add(all_dots_to_move)

        # self.play(FadeIn(all_dots_to_move))
        # self.wait()
        # # self.play(time_tracker.animate.set_value(0.0), run_time=0.1)
        # time_tracker.set_value(0.0)
        # self.play(FadeIn(vector_field))
        # self.wait()

        # history_length=20
        # for k in range(0,64):
        #     self.play(*[all_dots_to_move[path_index].animate.move_to(axes.c2p(*[history_pre_noise[k, path_index, 0], 
        #                         history_pre_noise[k, path_index, 1]])) for path_index in range(len(all_dots_to_move))], 
        #               *[ShowCreation(all_path_segments[path_index][2*k]) for path_index in range(len(all_dots_to_move))],
        #               *[all_path_segments[path_index][2*k].animate.set_opacity(0.7) for path_index in range(len(all_dots_to_move))],
        #               *[all_path_segments[path_index][2*k-history_length].animate.set_opacity(0.0) for path_index in range(len(all_dots_to_move))],
        #               run_time=0.4)

        #     self.play(*[all_dots_to_move[path_index].animate.move_to(axes.c2p(*[xt_history[k+1, path_index, 0], 
        #                         xt_history[k+1, path_index, 1]])) for path_index in range(len(all_dots_to_move))], 
        #               *[ShowCreation(all_path_segments[path_index][2*k+1]) for path_index in range(len(all_dots_to_move))],
        #               *[all_path_segments[path_index][2*k+1].animate.set_opacity(0.4) for path_index in range(len(all_dots_to_move))],
        #               *[all_path_segments[path_index][2*k+1-history_length].animate.set_opacity(0.0) for path_index in range(len(all_dots_to_move))],
        #               run_time=0.4)

        #     self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.2)
        # self.wait()

        # self.play(FadeOut(all_path_segments))
        # self.wait()
        # self.play(FadeOut(all_dots_to_move))
        # self.wait()

        # self.play(time_tracker.animate.set_value(0.0), run_time=1.0)

        # #Ok might want to start a new scene here, but maybe this is fine?
        # #Now, I think that doing traced paths here is probably a good/nice idea visually. let me make that plan A.
        # # model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_25_1.pt') #Trained on 64 levels

        # # gam=1
        # # mu=0.5 #0.5 is DDPM
        # # cfg_scale=0.0
        # # cond=None
        # # sigmas=schedule.sample_sigmas(64)
        # # xt_history=[]
        # # heatmaps=[]

        # # with torch.no_grad():
        # #     model.eval();
        # #     xt=torch.randn((batch_size,) + model.input_dims)*sigmas[0] #Scaling by sigma here matters a lot - why is that???

        # #     for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        # #         eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
        # #         # eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        # #         sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
        # #         eta = (sig_prev**2 - sig_p**2).sqrt()
        # #         xt = xt - (sig - sig_p) * eps #+ eta * model.rand_input(xt.shape[0]).to(xt) #Straight remove adding random noise
        # #         xt_history.append(xt.numpy())
        # #         heatmaps.append(model.forward(grid, sig, cond=None))
        # # xt_history=np.array(xt_history)

        # # Ok, this is pertty wierd, I'm not really able to replicated the trajectoties I'm seeing in jupyter on my 
        # # my linux machine here - Just going to import trajectories I guess? Do I want to do this for DDPM w/ noise too?
        # # I think this is probably ok, but kinda wierd. I'm going to keep moving for now. 
        # xt_history=np.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/ddpm_no_noise_1.npy')


        # # num_dots=16 #Start small for testing and crank for final animation. 
        # colors=get_color_wheel_colors(num_dots)
        # all_traced_paths=VGroup()
        # all_dots_to_move=VGroup()
        # for path_index in range(num_dots): 
        #     dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[0, path_index, :], [0]))), radius=0.06)
        #     dot_to_move.set_color(colors[path_index])
        #     all_dots_to_move.add(dot_to_move)

        #     traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=colors[path_index], stroke_width=2.0, 
        #                                   opacity_range=(0.0, 1.0), fade_length=24) #Tryin opaicty 0 and longer fade length in v2
        #     # traced_path.set_opacity(0.5)
        #     # traced_path.set_fill(opacity=0)
        #     all_traced_paths.add(traced_path)
        # self.add(all_traced_paths)

        # self.wait()

        # self.play(FadeIn(all_dots_to_move), self.frame.animate.reorient(0, 0, 0, (-0.06, 0.01, 0.0), 7.10), run_time=3.0)
        # self.wait()

        # for k in range(64):
        #     self.play(time_tracker.animate.set_value(8.0*(k/64.0)), 
        #               *[all_dots_to_move[path_index].animate.move_to(axes.c2p(*[xt_history[k, path_index, 0], 
        #                                                                         xt_history[k, path_index, 1]])) for path_index in range(len(all_dots_to_move))],
        #              rate_func=linear, run_time=0.2)
        #     # for traced_path in all_traced_paths:
        #     #     traced_path.update_path(0.1) 

        #     # self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.1)

        # self.wait()
        # self.play(FadeOut(all_traced_paths), FadeOut(vector_field), FadeOut(axes),
        #           self.frame.animate.reorient(0, 0, 0, (-0.11, -0.32, 0.0), 6.34), 
        #           run_time=2.5)
        # self.wait()

        # #Ok so to get me into p57, I think i just want to go back to that sam path I showed at the beggining of 40?
        # # self.play(FadeIn(axes), FadeOut(all_dots_to_move))

        # time_tracker.set_value(8.0)
        # time_tracker.set_value(0.0) #Doesn't seem liek this is taking?
        # self.play(FadeIn(axes), 
        #           FadeOut(all_dots_to_move), 
        #           # FadeIn(vector_field), #Lets actually fade in the vector field in p57
        #           self.frame.animate.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.25), run_time=3.0)
        # self.wait()

        # batch_size=2130
        # dataset = Swissroll(np.pi/2, 5*np.pi, 100)
        # loader = DataLoader(dataset, batch_size=batch_size)
        # batch=next(iter(loader)).numpy()
        # dots = VGroup()
        # for point in batch:
        #     # Map the point coordinates to the axes
        #     screen_point = axes.c2p(point[0], point[1])
        #     dot = Dot(screen_point, radius=0.04)
        #     # dot.set_color(YELLOW)
        #     dots.add(dot)
        # dots.set_color(YELLOW)
        # dots.set_opacity(0.3)

        # i=75
        # dot_to_move=dots[i].copy()
        # traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=3.5, 
        #                               opacity_range=(0.25, 0.9), fade_length=15)
        # # traced_path.set_opacity(0.5)
        # traced_path.set_fill(opacity=0)


        # np.random.seed(485) #485 is nice, 4 is maybe best so far, #52 is ok
        # random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
        # random_walk[0]=np.array([0.2, 0.12]) #make first step go up and to the right
        # # random_walk[-1]=np.array([0.15, -0.04])
        # random_walk[-1]=np.array([0.19, -0.05])
        # random_walk=np.cumsum(random_walk,axis=0) 

        # random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        # random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        
        # dot_history=VGroup()
        # dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
        # # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)
        # # dot_to_move.move_to(axes.c2p(*random_walk_shifted[1]))
        # traced_path.update_path(0.1)

        # for j in range(100):
        #     dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
        #     dot_to_move.move_to(axes.c2p(*random_walk_shifted[j]))
        #     traced_path.update_path(0.1)
        #     # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)
        # traced_path.stop_tracing()
        # dot_history.set_opacity(0.0)
        # dot_to_move.set_opacity(1.0)

        # self.wait()
        # self.play(self.frame.animate.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69), 
        #          FadeIn(traced_path), FadeIn(dot_to_move), run_time=3)
        # self.wait()

        
        # self.add(traced_path, dot_to_move)


        #Hmm seeing better results than I expected compared to matplotlib - investigating...


        # [traced_path.update_path(0.1) for traced_path in all_traced_paths]
        # # all_traced_paths.set_opacity(1.0)

        # traced_path.set_fill(opacity=1.0)


        # all_traced_paths[0].update_path(0.1)


        # for k in range(0, 64):
        #     for path_index in range(num_dots): 
        #         self.play(dot_to_move.animate.move_to(axes.c2p(*[history_pre_noise[k, path_index, 0], 
        #                                                          history_pre_noise[k, path_index, 1]])),
        #                   ShowCreation(path_segments[2*k]),
        #                   path_segments[2*k].animate.set_opacity(0.8),
        #                   run_time=0.4)

        #         self.play(dot_to_move.animate.move_to(axes.c2p(*[xt_history[k+1, path_index, 0], 
        #                                                          xt_history[k+1, path_index, 1]])),
        #                   ShowCreation(path_segments[2*k+1]),
        #                   path_segments[2*k+1].animate.set_opacity(0.5),
        #                   run_time=0.4)

        #     self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.1)

        #Don't forget to update vector field as we go! Might want to add a little line in the script about this.
        #Done!
        

        #Look at all dot real quick to get a sanity check on sprial fit
        # for path_index in range(512):
        #     dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[-1, path_index, :], [0]))), radius=0.04)
        #     dot_to_move.set_color(WHITE)
        #     self.add(dot_to_move)       



        # plt.plot([xt_history[k, j, 0], history_pre_noise[k, j, 0]], [xt_history[k, j, 1], history_pre_noise[k, j, 1]], 'm')
        # plt.plot([history_pre_noise[k, j, 0], xt_history[k+1, j, 0]], [history_pre_noise[k, j, 1], xt_history[k+1, j, 1]], 'c')



        # Ok, I think that lowering sigma max for this example defintely makes sense!
        # However I'm not really landing nicely on the spiral! And I want to
        # I think in need to back to jupyter notebook for a bit and tune - mayb revisit Chenyangs original config
        # I want to say he only did 20 steps?!
        # Ok I'm going to back to writing for the weekend I think
        # Made pretty good progress here - will pick back up on tuning spiral DDPM when I'm back on animation - let's go!








        self.wait(20)
        self.embed()

