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


def create_noisy_arrow_animation(self, start_point, end_point, target_point, num_steps=100, noise_level=0.1, overshoot_factor=0.3):
    """
    Creates a sequence of arrow end positions that converge from end_point to target_point
    with parameterizable noise and overshoot past the target direction.
    """
    
    # Calculate initial and target directions
    initial_direction = np.array(end_point) - np.array(start_point)
    target_direction = np.array(target_point) - np.array(start_point)
    
    # Calculate the constant length
    arrow_length = np.linalg.norm(initial_direction)
    
    # Calculate angles
    initial_angle = np.arctan2(initial_direction[1], initial_direction[0])
    target_angle = np.arctan2(target_direction[1], target_direction[0])
    
    # Handle angle wrapping (choose the shorter path)
    angle_diff = target_angle - initial_angle
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    # Create interpolation parameter
    t_values = np.linspace(0, 1, num_steps)
    
    # Generate noise that decreases over time
    np.random.seed(42)
    noise_decay = np.exp(-3 * t_values)
    angle_noise = noise_level * noise_decay * np.random.randn(num_steps)
    
    # Generate overshoot in angle space - this will make it swing past the target angle
    overshoot_frequency = 3.0
    overshoot_decay = np.exp(-2 * t_values)
    overshoot_oscillation = overshoot_factor * overshoot_decay * np.sin(overshoot_frequency * np.pi * t_values)
    
    # The key: let the angle interpolation overshoot past the target
    t_effective = t_values + overshoot_oscillation
    # Ensure final angle is exactly the target
    t_effective[-1] = 1.0
    
    arrow_positions = []
    
    for i, t_eff in enumerate(t_effective):
        # Interpolate angle - this is where the overshoot happens
        current_angle = initial_angle + t_eff * angle_diff
        
        # Add angular noise (but not on final step)
        if i < len(t_effective) - 1:
            current_angle += angle_noise[i]
        
        # Convert back to cartesian coordinates
        end_x = np.array(start_point)[0] + arrow_length * np.cos(current_angle)
        end_y = np.array(start_point)[1] + arrow_length * np.sin(current_angle)
        
        arrow_positions.append([end_x, end_y, 0])
    
    return arrow_positions



class p48_51v4(InteractiveScene):
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

        i=75
        dot_to_move=dots[i].copy()
        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=3.5, 
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

        for j in range(100):
            dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
            dot_to_move.move_to(axes.c2p(*random_walk_shifted[j]))
            traced_path.update_path(0.1)
            # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)
        traced_path.stop_tracing()

        dot_to_move.set_opacity(1.0)

        #Ok let me try to get all the big elements in here
        x100=Tex('x_{100}', font_size=24).set_color(YELLOW)
        x100.next_to(dot_to_move, 0.07*UP+0.001*RIGHT)

        x0=Tex('x_{0}', font_size=24).set_color('#00FFFF')
        x0.next_to(dots[i], 0.2*UP)
        dots[i].set_color('#00FFFF').set_opacity(1.0)

        arrow_x100_to_x0 = Arrow(
            start=dot_to_move.get_center(),
            end=dots[i].get_center(),
            thickness=1,
            tip_width_ratio=5, 
            buff=0.025  # Small buffer so arrow doesn't overlap the dots
        )
        arrow_x100_to_x0.set_color('#00FFFF')
        arrow_x100_to_x0.set_opacity(0.6)


        arrow_x100_to_x99 = Arrow(
            start=dot_to_move.get_center(),
            end=[4.739921625933185, 2.8708813273028455, 0], #Just pul in from previous paragraph, kinda hacky but meh. ,
            thickness=1.5,
            tip_width_ratio=5, 
            buff=0.04  # Small buffer so arrow doesn't overlap the dots
        )
        arrow_x100_to_x99.put_start_and_end_on(dot_to_move.get_center(), [4.739921625933185, 2.8708813273028455, 0]) #Eh?
        arrow_x100_to_x99.set_color(CHILL_BROWN)
        # arrow_x100_to_x99.set_opacity(0.6)


        # arrow_x100_to_x99 = Arrow(
        #     start=dot_to_move.get_center(),
        #     end=dot_history[-1].get_center(),
        #     thickness=1.5,
        #     tip_width_ratio=5, 
        #     buff=0.04  # Small buffer so arrow doesn't overlap the dots
        # )


        self.frame.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69)
        self.add(axes, dots, dot_to_move)
        # self.add(traced_path)
        self.add(x100,  x0, arrow_x100_to_x0, arrow_x100_to_x99)
        self.wait()

        # Ok so the continuity to think/worry about here is the brown arrow! Now I'm a bit worried about it's angle - hmm 
        # Let's see how it shakes out. 
        # I think first it's Fading everythig except that data and brown line (maybe scale of brown arrow changes)
        # I might beg able to get away with some updates to the brown arrows angle on a zoom out as I add stuff, we'll see. 

        self.play(
                # FadeOut(traced_path), 
                  FadeOut(dot_to_move), 
                  FadeOut(x100), 
                  FadeOut(x0), 
                  FadeOut(arrow_x100_to_x0), 
                 dots.animate.set_opacity(1.0).set_color(YELLOW), run_time=1.5)

        # Ok ok ok so I now in need some vector fields. These come from trained models. Do I want to import the model 
        # and sample from it here? Or do I want to exprot the vector fields? 
        # I think it would be nice to fuck with the density etc in manim, so maybe we get a little aggressive and 
        # try to import the full model? 
        # Lets see here....
        # Hmm kinda unclear if this is going to work on mac/CPU -> i guess it's worth a try? Pretty sure I can't train w/o cuda. 

        model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_20_1.pt')

        schedule = ScheduleLogLinear(N=256, sigma_min=0.01, sigma_max=10) #N=200
        bound=2.0 #Need to match extended axes bro
        num_heatmap_steps=30
        grid=[]
        for i, x in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
            for j, y in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
                grid.append([x,y])
        grid=torch.tensor(grid).float()

        gam=1
        mu=0.01 #0.5 is DDPM
        cfg_scale=0.0
        cond=None
        sigmas=schedule.sample_sigmas(256)
        xt_history=[]
        heatmaps=[]
        eps=None

        with torch.no_grad():
            model.eval();
            xt=torch.randn((batch_size,) + model.input_dims)*sigmas[0] #Scaling by sigma here matters a lot - why is that???

            for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
                eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
                # eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
                sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
                eta = (sig_prev**2 - sig_p**2).sqrt()
                xt = xt - (sig - sig_p) * eps + eta * model.rand_input(xt.shape[0]).to(xt)
                xt_history.append(xt.numpy())
                heatmaps.append(model.forward(grid, sig, cond=None))

        xt_history=np.array(xt_history)
        self.wait()

        # Ok nice glad i tried this! Seems like I can sample right in manim - that's great. 
        # Ok now let's draw some arrows, and then try to figure out how to bring thme in as a nice continuous
        # extension of the single arrow I have. 
        final_vectors = heatmaps[-1].detach().numpy()  # Shape should be (num_heatmap_steps^2, 2)

        sigma_index=-1
        def vector_function_direct(coords_array):
            # print(coords_array.shape)
            res=model.forward(torch.tensor(coords_array).float(), sigmas[sigma_index], cond=None)
            return -res.detach().numpy()

        # individual_arrows=extract_individual_arrows(vector_field)
        # Ok so we still need to figure out a smooth transition between the single individual vector and the vector field
        # If I can extract a single vector from the field I could do a replacement transform as I roll in rest of vectors and zoom 
        # out. Before I do that though, let me make sure the time varying version of the vector field looks good. 
        # I need to increment sigma_index from 0 to 255 and redraw the field each time. Claude?
        # Ok so there's some cool time varying stuff I could do -> let me try the super simple loop appraoch first though 
        # Just need to validate I get the fun/interesting temporal behavior when animating this way. 
        # Hmm maybe not actually?


        time_tracker = ValueTracker(0.0)  # Start at time 0

        def vector_function_with_tracker(coords_array):
            """Vector function that uses the ValueTracker for time"""
            current_time = time_tracker.get_value()
            max_time = 8.0  # Map time 0-8 to sigma indices 0-255
            sigma_idx = int(np.clip(current_time * 255 / max_time, 0, 255))
            
            try:
                res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=None)
                return -res.detach().numpy()
            except:
                return np.zeros((len(coords_array), 2))

        # Create a custom VectorField that updates based on the tracker
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


        # Create the tracker-controlled vector field
        vector_field = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_with_tracker,
            coordinate_system=extended_axes,
            density=3.0,
            stroke_width=2,
            max_radius=6.0,      # Vectors fade to min_opacity at this distance
            min_opacity=0.2,     # Minimum opacity at max_radius
            max_opacity=1.0,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=CHILL_BROWN
        )
        self.wait()

        # Ok so I need a smooth transition from arrow_x100_to_x99 to the nearest vector in teh vector field
        # Would like to simultanesouly do the camera move, progressively bring in vector field - radiating out
        # from arrow_x100_to_x99 would be dope, and i need to do like a replacement transform between 
        # arrow_x100_to_x99 and the nearest vector in the field. 


        # self.play(ShowCreation(vector_field), 
        #          arrow_x100_to_x99.animate.rotate(-14*DEGREES).shift([0.17, -0.07, 0]).scale([0.6, 1.2, 1]).set_opacity(0.8),
        #          self.frame.animate.reorient(0, 0, 0, (-0.06, 0.09, 0.0), 8.31), 
        #          run_time=6.0)

        #ok i think fading in teh vector field is probably th emove. 
        self.play(FadeIn(vector_field), 
                 dots.animate.set_opacity(0.75),
                 arrow_x100_to_x99.animate.rotate(-14*DEGREES).shift([0.17, -0.07, 0]).scale([0.6, 1.2, 1]).set_opacity(0.5),
                 self.frame.animate.reorient(0, 0, 0, (-0.21, 0.02, 0.0), 8.08), 
                 run_time=16.0)
        self.remove(arrow_x100_to_x99)
        self.wait()

        # self.frame.reorient(0, 0, 0, (-0.06, 0.09, 0.0), 8.31)
        # self.play(FadeIn(vector_field), run_time=2.0)
        # self.wait()
        # arrow_x100_to_x99.rotate(-14*DEGREES).shift([0.17, -0.07, 0]).scale([0.6, 1.2, 1]).set_opacity(0.8)
        # arrow_x100_to_x99.shift([0.15, -0.08, 0])
        # arrow_x100_to_x99.scale([0.6, 1.2, 1])
        # arrow_x100_to_x99.set_opacity(0.8)

        # arrow_x100_to_x99.rotate(-2*DEGREES)

        # Ok that transition looks nice! Now I need to make things even crazier and run the forward diffusion process
        # for 100 steps (I think it makes sense to have a step counter at the bottom right?)
        # Probably completele fade out vectors while I play forward diffusion, then bring back in??
        # self.wait()

        dots_to_move = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots_to_move.add(dot)
        dots_to_move.set_color(YELLOW)
        dots_to_move.set_opacity(1.0)

        random_walks=[]
        np.random.seed(2)
        schedule2 = ScheduleLogLinear(N=100, sigma_min=0.02, sigma_max=0.09) #Different schedule for viz?
        # schedule2 = ScheduleDDPM(N=100, beta_start=0.02, beta_end=0.25) #Different schedule for viz?
        sigmas100=schedule2.sample_sigmas(99)
        sigmas100=(sigmas100.numpy()[::-1]).reshape(-1,1)
        for i in range(100):
            # rw=0.07*np.random.randn(100,2) #Uniform steps
            rw=sigmas100*np.random.randn(100,2) #Real noise schedule(scaled down)
            rw[0]=np.array([0,0]) #make be the starting point
            # rw[-1]=np.array([0.08, -0.02])
            rw=np.cumsum(rw,axis=0) 
            rw=np.hstack((rw, np.zeros((len(rw), 1))))
            rw_shifted=rw+np.array([batch[i][0], batch[i][1], 0])
            random_walks.append(rw_shifted)

        traced_paths=VGroup()
        for idx, d in enumerate(dots_to_move): 
            tp = CustomTracedPath(
                    d.get_center, 
                    stroke_color=YELLOW, 
                    stroke_width=2,
                    opacity_range=(0.1, 0.5),
                    fade_length=10
                )
            traced_path.set_fill(opacity=0)
            traced_paths.add(tp)
        self.add(traced_paths)

        step_count=MarkupText(str(1), font_size=35)
        step_count.set_color(CHILL_BROWN)
        step_count.move_to([-6.8, -3.3, 0])

        step_label=MarkupText("STEP", font_size=18, font='myriad-pro')  
        step_label.set_color(CHILL_BROWN).set_opacity(0.7)
        step_label.next_to(step_count, DOWN, buff=0.1)


        self.wait()
        self.play(FadeOut(vector_field), FadeIn(step_label), FadeIn(step_count))


        self.wait()
        self.add(dots_to_move)
        dots.set_opacity(0.3)
        for step in range(100):
            self.play(*[dots_to_move[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in range(len(dots_to_move))], 
                     run_time=0.1, rate_func=linear)
            self.remove(step_count)
            step_count=MarkupText(str(step+1), font_size=35)
            step_count.set_color(CHILL_BROWN)
            step_count.move_to([-6.8, -3.3, 0])
            self.add(step_count)

        self.wait()
        
        for tp in traced_paths: tp.stop_tracing()

        #Ok now fade back in Vector field (at t=100)
        vector_field.set_color('#FFFFFF')
        self.play(FadeIn(vector_field),
                  dots_to_move.animate.set_opacity(0.3))
        self.wait()


        # Ok ok ok ok now fade back out vector field, reverse diffusion, and add t=0 vector field!
        # Gotta count down steps as I play backwards too. 
        # Hmm before I go further here - this is probably a good time to go back and figure out 
        # why I can't render paths backwards. 
        # Might try adding a real noise schedule to earlier paths too
        # Ok let me back track and fix that, then will return here. 
        self.play(FadeOut(vector_field))

        # Ok i spent like 30-40 minutes trying to make backwards diffusion work 
        # in export mode, no luck. 
        # I think I can just do it in post? Let me think through it here. 
        self.wait()
        editor_warning=MarkupText('Reverse in Post!')
        self.add(editor_warning)
        self.wait()
        self.remove(editor_warning)
        self.wait()

        self.remove(dots_to_move, traced_paths)

        #Jump cut to step 1, will play back smoothly in post. 
        dots_to_move_2 = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots_to_move_2.add(dot)
        dots_to_move_2.set_color(YELLOW)
        dots_to_move_2.set_opacity(1.0)

        traced_paths_2=VGroup()
        for idx, d in enumerate(dots_to_move_2): 
            tp = CustomTracedPath(
                    d.get_center, 
                    stroke_color=YELLOW, 
                    stroke_width=2,
                    opacity_range=(0.1, 0.5),
                    fade_length=10
                )
            traced_path.set_fill(opacity=0)
            traced_paths_2.add(tp)
        self.add(traced_paths_2)


        self.remove(step_count)
        step_count=MarkupText(str(1), font_size=35)
        step_count.set_color(CHILL_BROWN)
        step_count.move_to([-6.8, -3.3, 0])
        self.add(step_count)

        self.add(dots_to_move_2)
        dots.set_opacity(0.3)
        for step in range(2):
            self.play(*[dots_to_move_2[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in range(len(dots_to_move_2))], 
                     run_time=0.1, rate_func=linear)
        self.wait()

        # time_tracker.set_value(8.0)
        
        self.add(vector_field)
        time_tracker.set_value(8*(99/100))
        self.wait()
        self.remove(vector_field)



        self.play(FadeIn(vector_field), dots_to_move_2.animate.set_opacity(0.3))

        # self.play(time_tracker.animate.set_value(8.0), run_time=8.0)
        # self.play(time_tracker.animate.set_value(0.0), run_time=4.0)

        self.wait()

        # Ok great, this gets us to the end of p49. 
        # Alright so in 50, we want to fade stuff out I think and zoom back in on a single path. 
        # Then I think it's illustrator overlays and a finally a new scene for p51. 


        #Roll all these into big unified move
        self.play(FadeOut(vector_field),
                  FadeOut(dots_to_move_2),
                  FadeOut(traced_paths_2),
                  FadeOut(step_count),
                  FadeOut(step_label),
                  FadeIn(traced_path),
                  FadeIn(dot_to_move),
                  self.frame.animate.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69),
                  run_time=5.0
                  )
        self.wait()

        # self.add(traced_path)
        # self.add(dot_to_move)
        # self.frame.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69)
        # self.add(axes, dots, traced_path, dot_to_move)
        # self.add(x100,  x0, arrow_x100_to_x0, arrow_x100_to_x99)
        # self.wait()


        x100=Tex('x_{100}', font_size=24).set_color(YELLOW)
        x100.next_to(dot_to_move, 0.07*UP+0.001*RIGHT)

        x0=Tex('x_{0}', font_size=24).set_color('#00FFFF')
        x0.next_to(dots[75], 0.2*UP)

        #Probably do a big fade in here
        arrow_x100_to_x0.set_opacity(1.0)
        self.add(arrow_x100_to_x0)
        dots[75].set_color('#00FFFF').set_opacity(1.0)
        self.add(x100, x0)
        self.wait()


        eq_1=Tex("f(x_{100})", font_size=24)
        eq_1.set_color('#00FFFF')
        eq_1.move_to([3.5, 2.2, 0])
        self.add(eq_1)

        eq_2=Tex("f(x_{100}, t)", font_size=24)
        eq_2.set_color('#00FFFF')
        eq_2.move_to(eq_1, aligned_edge=LEFT)
        self.wait()

        # self.add(eq_2)

        # eq_1[-1].move_to([4.03, 2.2, 0])
        self.play(eq_1[-1].animate.move_to([4.03, 2.2, 0]), run_time=1.4)
        self.add(eq_2) #I think just adding might be cleaner
        # self.play(FadeIn(eq_2))
        self.remove(eq_1)
        self.wait()
        #Now fade in , t???
        eq_3=Tex("f(x_{100}, t=1.0)", font_size=24)
        eq_3.set_color('#00FFFF')
        eq_3.move_to(eq_1, aligned_edge=LEFT)
        self.wait()

        # eq_2[-1].move_to([4.65, 2.2, 0])
        self.play(eq_2[-1].animate.move_to([4.65, 2.2, 0]), run_time=1.4)
        self.add(eq_3)
        self.remove(eq_2)
        self.wait()
        #Ok now I want little arrows pointing from x99, and then maybe one from like x3?

        arrow_x99_to_x0 = Arrow(
            start=traced_path.traced_points[-2],
            end=dots[75].get_center(),
            thickness=1,
            tip_width_ratio=5, 
            buff=0.025  # Small buffer so arrow doesn't overlap the dots
        )
        arrow_x99_to_x0.set_color('#FF00FF')
        arrow_x99_to_x0.set_opacity(1.0)
        dot99=Dot(traced_path.traced_points[-2], radius=0.04)
        dot99.set_color("#FF00FF")

        eq_4=Tex("f(x_{99}, t=0.99)", font_size=20)
        eq_4.set_color('#FF00FF')
        eq_4.move_to([3.1, 2.9, 0])
        # self.add(eq_4)

        self.wait()
        self.play(FadeIn(arrow_x99_to_x0), FadeIn(dot99), FadeIn(eq_4))
        self.wait()



        arrow_x3_to_x0 = Arrow(
            start=traced_path.traced_points[2],
            end=dots[75].get_center(),
            thickness=1,
            tip_width_ratio=5, 
            buff=0.025  # Small buffer so arrow doesn't overlap the dots
        )
        arrow_x3_to_x0.set_color(GREEN)
        arrow_x3_to_x0.set_opacity(1.0)
        dot3=Dot(traced_path.traced_points[2], radius=0.04)
        dot3.set_color(GREEN)

        eq_5=Tex("f(x_{2}, t=0.02)", font_size=20)
        eq_5.set_color(GREEN)
        eq_5.move_to([1.93, 2.45, 0])
        # self.add(eq_4)

        self.wait()
        self.play(FadeIn(arrow_x3_to_x0), FadeIn(dot3), FadeIn(eq_5))
        self.wait()

        
        # self.add(arrow_x99_to_x0, )
        # self.add(dot99)
        # self.remove(arrow_x99_to_x0, dot99, eq_4)

        # Alright lets keep rollin here I guess? 
        # Fade basically all of this new stuff out, zoom back to center view, and run vector field animation

        self.play(FadeOut(arrow_x3_to_x0),
                  FadeOut(eq_5),
                  FadeOut(eq_4),
                  FadeOut(eq_3),
                  FadeOut(arrow_x99_to_x0),
                  FadeOut(arrow_x100_to_x0),
                  FadeOut(traced_path),
                  FadeOut(dot_to_move),
                  FadeOut(x100),
                  FadeOut(dot3),
                  FadeOut(dot99),
                  FadeOut(dot_to_move),
                  dots[75].animate.set_color(YELLOW).set_opacity(0.3),
                  FadeOut(x0),
                  self.frame.animate.reorient(0, 0, 0, (-0.21, 0.02, 0.0), 8.08),
                  run_time=4.0)
        self.wait()



        time_tracker.set_value(0)
        # vector_field.set_color(CHILL_BROWN) #Can't decde on color!
        vector_field.set_color('#FFFFFF')
        # self.play(FadeIn(vector_field))
        self.wait()
        # self.play(time_tracker.animate.set_value(8.0), run_time=10.0)

        # Ok so now i need to figure out how to upate a t="" little counter in the lower left corner like I did before. 
        # I need to incrementally move t from 1.0 down to 0.0 as the animation above runs. Claude can you help?
        time_value = ValueTracker(1.0)  # Start at t=1.0
        time_display = DecimalNumber(
            1.0,
            num_decimal_places=2,
            font_size=35,
            color=CHILL_BROWN
        )
        time_display.move_to([-6.3, -3.3, 0])  # Same position as your step counter

        time_label = MarkupText("t =", font_size=35)
        time_label.set_color(CHILL_BROWN)
        time_label.next_to(time_display, LEFT, buff=0.15)

        # Add updater to keep the display synchronized with the tracker
        time_display.add_updater(lambda m: m.set_value(time_value.get_value()))

        # Replace your final animation section with this:
        time_tracker.set_value(0)
        self.play(FadeIn(vector_field), FadeIn(time_label), FadeIn(time_display))
        self.wait()
        # Animate both the vector field time progression AND the display counter
        self.play(
            time_tracker.animate.set_value(8.0),  # Your existing vector field animation
            time_value.animate.set_value(0.0),    # Time counter goes from 1.0 to 0.0
            run_time=10.0, 
            rate_func=linear
        )
        self.wait()



        self.wait(20)
        self.embed()










class p47bv2(InteractiveScene):
    def construct(self):
        '''
        Alright need to pick up where i left off on p44_47, and get ready for another crazy particle fly by lol
        This is a little nuts - but I do think it conveys the point nicely. 
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

        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(YELLOW)
        dots.set_opacity(0.3)

        i=75
        dot_to_move=dots[i].copy()
        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=3.5, 
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

        for j in range(100):
            dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
            dot_to_move.move_to(axes.c2p(*random_walk_shifted[j]))
            traced_path.update_path(0.1)
            # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)
        traced_path.stop_tracing()

        dot_to_move.set_opacity(1.0)
        self.frame.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69)
        self.add(axes, dots, traced_path, dot_to_move)
        self.wait()


        #Ok let me try to get all the big elements in here
        x100=Tex('x_{100}', font_size=24).set_color(YELLOW)
        x100.next_to(dot_to_move, 0.07*UP+0.001*RIGHT)

        x99=Tex('x_{99}', font_size=24).set_color(CHILL_BROWN)
        x99.next_to(dot_history[-1], 0.1*UP+0.01*RIGHT)
        dot99=Dot(dot_history[-1].get_center(), radius=0.04)
        dot99.set_color(CHILL_BROWN)

        x0=Tex('x_{0}', font_size=24).set_color('#00FFFF')
        x0.next_to(dots[i], 0.2*UP)
        dots[i].set_color('#00FFFF').set_opacity(1.0)

        arrow_x100_to_x0 = Arrow(
            start=dot_to_move.get_center(),
            end=dots[i].get_center(),
            thickness=1,
            tip_width_ratio=5, 
            buff=0.025  # Small buffer so arrow doesn't overlap the dots
        )
        arrow_x100_to_x0.set_color('#00FFFF')
        arrow_x100_to_x0.set_opacity(0.6)

        arrow_x100_to_x99 = Arrow(
            start=dot_to_move.get_center(),
            end=dot_history[-1].get_center(),
            thickness=1.5,
            tip_width_ratio=5, 
            buff=0.04  # Small buffer so arrow doesn't overlap the dots
        )
        arrow_x100_to_x99.set_color(CHILL_BROWN)
        # arrow_x100_to_x99.set_opacity(0.6)


        self.add(x100, x99, dot99, x0, arrow_x100_to_x0, arrow_x100_to_x99)
        self.wait()

        # Alright probably need to tweak how I'm adding stuff etc -> but let's get to the main event though
        # So probably lost the labels and yellow path? Yeah maybe like this:
        self.remove(x99, traced_path)
        self.wait()

        # Ok so i gotta send a bunch of particles, I can probably just use the same exact animation
        # First let me figure out how I wanto to move the brown arrow
        # It needs to feel noisy, but not too noisy, and coverge to exactly the x0 direction, and length needs
        # to stay the same. And i need 100 steps. Let's ask my buddy Claude. 
        noise_level = 0.06  # Adjust this parameter to control noise amount
        overshoot_factor = 2.0  # Adjust this to control how much overshoot occurs
        start_delay=20
        early_end=10
        arrow_end_positions = create_noisy_arrow_animation(
            self, 
            start_point=dot_to_move.get_center()[:2],  # x100 position (2D)
            end_point=dot_history[-1].get_center()[:2],  # x99 position (2D) 
            target_point=dots[i].get_center()[:2],  # x0 position (2D)
            num_steps=100-start_delay-early_end,
            noise_level=noise_level,
            overshoot_factor=overshoot_factor
        )


        # self.wait()
        # for end_pos in arrow_end_positions:
        #     arrow_x100_to_x99.put_start_and_end_on(
        #         dot_to_move.get_center(),
        #         end_pos
        #         # np.concatenate((end_pos, [0]))
        #     )
        #     self.wait(0.05)  # 5 seconds total for 100 steps

        #Ok that works! Now I need this motion to happen while all the points fly by! Let's try the same points as last time. 


        random_walks=[]
        np.random.seed(2)
        for j in tqdm(range(int(2e6))):
            rw=0.07*np.random.randn(100,2)
            rw[0]=np.array([0,0]) #make be the starting point
            # rw[-1]=np.array([0.08, -0.02])
            rw=np.cumsum(rw,axis=0) 
            rw=np.hstack((rw, np.zeros((len(rw), 1))))
            rw_shifted=rw+np.array([batch[j%len(batch)][0], batch[j%len(batch)][1], 0])
            # if rw_shifted[-1][0]>1.7 and rw_shifted[-1][0]<2.2 and rw_shifted[-1][1]>1.1 and rw_shifted[-1][1]<1.4:
            if rw_shifted[-1][0]>2.1 and rw_shifted[-1][1]>1.4:
                random_walks.append(rw_shifted)

        print(len(random_walks))
        # random_walks=random_walks[:100] #Comment out to do all the points.
        print(len(random_walks))

        dots_to_move = VGroup()
        for j in range(len(random_walks)):
            # Map the point coordinates to the axes
            screen_point = axes.c2p(batch[j%len(batch)][0], batch[j%len(batch)][1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots_to_move.add(dot)
        dots_to_move.set_color(FRESH_TAN)
        dots_to_move.set_opacity(0.2)


        traced_paths=VGroup()
        for idx, d in enumerate(dots_to_move): 
            # tp = TracedPath(d.get_center, stroke_color=YELLOW, stroke_width=2)
            if idx != 75:  # Skip the already traced dot
                tp = CustomTracedPath(
                        d.get_center, 
                        stroke_color=FRESH_TAN, 
                        stroke_width=2,
                        opacity_range=(0.01, 0.35),
                        fade_length=10
                    )
                traced_path.set_fill(opacity=0)
                traced_paths.add(tp)
        self.add(traced_paths)

        self.wait()
        for step in range(100):
            self.play(*[dots_to_move[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in range(len(random_walks))], 
                     # self.frame.animate.reorient(*interp_orientations[step]), 
                     run_time=0.1, rate_func=linear)
            if step>start_delay:
                arrow_index=np.clip(step-start_delay, 0, len(arrow_end_positions)-1)
                arrow_x100_to_x99.put_start_and_end_on(dot_to_move.get_center(), arrow_end_positions[arrow_index])
        self.wait()

        self.remove(dots_to_move, traced_paths, dot99) #Might be nice to do a fade out

        self.wait(20)
        self.embed()



class p44_47(InteractiveScene):
    def construct(self):
        '''
        Ok going to try a "clean break" here on the full spiral!
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
                "stroke_width": 3,
                "include_tip": True,
                "include_ticks": False,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )
        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.05)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(BLACK)

        i=75
        dot_to_move=dots[i].copy()
        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=CHILL_BROWN, stroke_width=4, 
                                      opacity_range=(0.4, 0.4), fade_length=90)
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

        for i in range(100):
            dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
            dot_to_move.move_to(axes.c2p(*random_walk_shifted[i]))
            traced_path.update_path(0.1)
            # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)

        self.wait()


        self.frame.reorient(0, 0, 0, (-0.07, 0.01, 0.0), 7.59)
        self.add(axes, dots)
        # self.add(traced_path)
        # self.add(dot_to_move)
        # self.add(dot_history)
        self.wait()


        traced_path.stop_tracing()


        #P45, zoom in and fade in walk
        self.play(FadeIn(traced_path), 
                  FadeIn(dot_to_move), 
                  # FadeIn(dot_history)
                  )


        # self.wait()
        # self.play(self.frame.animate.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69),
        #          dots.animate.set_opacity(0.3), run_time=3.0)
        # self.wait()


        

        #Book puase here maybe?
        dots.set_opacity(0.2)
        self.remove(dots[75])

        self.frame.reorient(0, 0, 0, (3.16, 1.7, 0.0), 3.86)
        self.wait(2)


        self.frame.reorient(0, 0, 0, 8.00)
        self.wait(2)



        # Ok so p45 will probably be all illustrator/premiere overlay 
        # From here then, I need to send a bunch of diffusion paths out
        # I'm hoping that it will be apparent visually that things are moving more left to right than the other way 
        # But i don't know yet
        # I might want to do a bit of a zoom out here -> we'll see. 
        # Could also draw a box around the neighborhood, or search for ways to accentuate the motion or somethhing - not sure 
        # yet. Let me try the naive approach and see how it feels. 
        # Hmm yeah this one is taking some noodling -> I do kinda think that drawing the neighborhood box might be good. 


        #Ok looks like I need to filter on paths that go through this neighborhood
        random_walks=[]
        np.random.seed(2)
        for j in tqdm(range(int(2e6))):
            rw=0.07*np.random.randn(100,2)
            rw[0]=np.array([0,0]) #make be the starting point
            # rw[-1]=np.array([0.08, -0.02])
            rw=np.cumsum(rw,axis=0) 
            rw=np.hstack((rw, np.zeros((len(rw), 1))))
            rw_shifted=rw+np.array([batch[j%len(batch)][0], batch[j%len(batch)][1], 0])
            # if rw_shifted[-1][0]>1.7 and rw_shifted[-1][0]<2.2 and rw_shifted[-1][1]>1.1 and rw_shifted[-1][1]<1.4:
            if rw_shifted[-1][0]>2.1 and rw_shifted[-1][1]>1.4:
                random_walks.append(rw_shifted)

        print(len(random_walks))
        # random_walks=random_walks[:100]
        print(len(random_walks))

        dots_to_move = VGroup()
        for j in range(len(random_walks)):
            # Map the point coordinates to the axes
            screen_point = axes.c2p(batch[j%len(batch)][0], batch[j%len(batch)][1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots_to_move.add(dot)
        dots_to_move.set_color(FRESH_TAN)
        dots_to_move.set_opacity(0.3)


        traced_paths=VGroup()
        for idx, d in enumerate(dots_to_move): 
            # tp = TracedPath(d.get_center, stroke_color=YELLOW, stroke_width=2)
            if idx != 75:  # Skip the already traced dot
                tp = CustomTracedPath(
                        d.get_center, 
                        stroke_color=FRESH_TAN, 
                        stroke_width=2,
                        opacity_range=(0.02, 0.5),
                        fade_length=10
                    )
                traced_path.set_fill(opacity=0)
                traced_paths.add(tp)
        self.add(traced_paths)


        remaining_indices=np.concatenate((np.arange(75), np.arange(76,len(batch))))    
        start_orientation=[0, 0, 0, (3.58, 2.57, 0.0), 2.69]
        end_orientation=[0, 0, 0, (4.86, 2.65, 0.0), 3.06]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=100)
        remaining_indices=np.concatenate((np.arange(75), np.arange(76,len(batch))))

        
        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (2.74, 1.72, 0.0), 3.99))

        r=RoundedRectangle(1.5, 1.0, 0.05)
        r.set_stroke(color='#00FFFF', width=2)
        r.move_to(dot_to_move)
        self.add(r)

        self.wait()
        for step in range(100):
            self.play(*[dots_to_move[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in range(len(random_walks))], 
                     # self.frame.animate.reorient(*interp_orientations[step]), 
                     run_time=0.1, rate_func=linear)
        self.wait()
        
        for tp in traced_paths: tp.stop_tracing()

        self.remove(dots_to_move, traced_paths)
        # self.play(FadeOut(dots_to_move), FadeOut(traced_paths), FadeOut(r))
        self.play(FadeOut(r))
        self.wait()

        #Zoom back in
        self.play(self.frame.animate.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69), run_time=3)
        self.wait()


        self.wait(20)
        self.embed()




class p40_44v2_black(InteractiveScene):
    def construct(self):
        '''
        May want to adopt an actual noise schedule here so we don't that big snap at the end - we'll see. 
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
                "stroke_width": 3,
                "include_tip": True,
                "include_ticks": False,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )

        self.add(axes)
        # self.wait()

        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.05)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(BLACK)

        self.wait()

        # Animate the points appearing
        # self.play(FadeIn(dots, lag_ratio=0.1), run_time=2)
        self.add(dots)
        self.wait()


        #Maybe first book pause here?
        # I think showing a few discret steps of the spiral 
        # diffusion process with tails could be cool? 
        # I think I do want the single particle path too. 




        # Ok, let's zoom in on one point, lower opacity on all other points, 
        # and send it on a random walk
        # I think we overlay the image stuff for p42 in illustrator/premier

        #Example_point_index
        i=75
        dot_to_move=dots[i].copy()
        

        self.wait()
        # self.play(dots.animate.set_opacity(0.1), 
        #          dot_to_move.animate.scale(1.25), #Make main dot a little bigger!
        #          self.frame.animate.reorient(0, 0, 0, (2.92, 1.65, 0.0), 4.19), 
        #          run_time=2.0)


        self.wait()

        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=BLACK, stroke_width=3, 
                                      opacity_range=(0.35, 0.35), fade_length=90)

        # self.wait()

        # traced_path.set_opacity(0.5)
        traced_path.set_fill(opacity=0)
        self.add(traced_path)
        self.add(dot_to_move)

        # self.remove(dot_to_move)


        np.random.seed(485) #485 is nice, 4 is maybe best so far, #52 is ok
        random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
        random_walk[0]=np.array([0.2, 0.12]) #make first step go up and to the right

        # Hmm how do I get that first step back?
        # like dis? Ok nice - but wait start point is a little off?
        # random_walk[1]=np.array([0.2, 0.12])

        # random_walk[-1]=np.array([0.08, -0.02])
        random_walk=np.cumsum(random_walk,axis=0) 
        # print(random_walk[0])

        random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        # print(random_walk_shifted[-1])

        #Hmm do I want an arrow or does the point just move with a tail?
        # self.wait()
        
        dot_history=VGroup()
        # dot_history.add(dot_to_move.copy().scale(0.22).set_color(YELLOW_FADE))
        # self.add(dot_history[-1])
        # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)


        self.wait()

        self.remove(dots[75])
        # self.remove(dot_to_move)
        # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=0.1, rate_func=linear)


        for j in range(100):
            if j>0: 
                dot_history.add(dot_to_move.copy().scale(0.75).set_color(BLACK))
                #self.add(dot_history[-1])
            # dot_to_move.move_to(axes.c2p(*random_walk_shifted[i]))
            self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[j])), run_time=0.1, rate_func=linear)

        

        dot_to_move.scale(1.25)

        self.wait()
        # Ok this is looking pretty guuud. #Book. 
        # Now, let me work on the progressive global diffusion a bit, then will go spend some time 
        # in adobe - Ok i think i want to break this apart though into two secense. let's do that. 

        self.wait(20)
        self.embed()

 



class p40_44v2b(InteractiveScene):
    def construct(self):
        '''
        Scene just for the book where we just do the global diffusion ish
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
                "stroke_width": 3,
                "include_tip": True,
                "include_ticks": False,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )

        self.add(axes)
        # self.wait()

        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.05)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(BLACK)

        self.wait()

        # Animate the points appearing
        # self.play(FadeIn(dots, lag_ratio=0.1), run_time=2)
        self.add(dots)
        # self.wait()


        #Maybe first book pause here?
        # I think showing a few discret steps of the spiral 
        # diffusion process with tails could be cool? 
        # I think I do want the single particle path too. 




        # Ok, let's zoom in on one point, lower opacity on all other points, 
        # and send it on a random walk
        # I think we overlay the image stuff for p42 in illustrator/premier

        #Example_point_index
        i=75
        dot_to_move=dots[i].copy()
        

        # self.wait()
        # # self.play(dots.animate.set_opacity(0.1), 
        # #          dot_to_move.animate.scale(1.25), #Make main dot a little bigger!
        # #          self.frame.animate.reorient(0, 0, 0, (2.92, 1.65, 0.0), 4.19), 
        # #          run_time=2.0)


        # self.wait()

        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=BLACK, stroke_width=3, 
                                      opacity_range=(0.35, 0.35), fade_length=90)

        # self.wait()

        # traced_path.set_opacity(0.5)
        traced_path.set_fill(opacity=0)
        # self.add(traced_path)
        # self.add(dot_to_move)

        # self.remove(dot_to_move)


        np.random.seed(485) #485 is nice, 4 is maybe best so far, #52 is ok
        random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
        random_walk[0]=np.array([0.2, 0.12]) #make first step go up and to the right

        # Hmm how do I get that first step back?
        # like dis? Ok nice - but wait start point is a little off?
        # random_walk[1]=np.array([0.2, 0.12])

        # random_walk[-1]=np.array([0.08, -0.02])
        random_walk=np.cumsum(random_walk,axis=0) 
        # print(random_walk[0])

        random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        # print(random_walk_shifted[-1])

        #Hmm do I want an arrow or does the point just move with a tail?
        # self.wait()
        
        dot_history=VGroup()
        # dot_history.add(dot_to_move.copy().scale(0.22).set_color(YELLOW_FADE))
        # self.add(dot_history[-1])
        # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)


        self.wait()

        # self.remove(dots[75])
        # self.remove(dot_to_move)
        # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=0.1, rate_func=linear)


        # for j in range(100):
        #     if j>0: 
        #         dot_history.add(dot_to_move.copy().scale(0.75).set_color(BLACK))
        #         #self.add(dot_history[-1])
        #     # dot_to_move.move_to(axes.c2p(*random_walk_shifted[i]))
        #     self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[j])), run_time=0.1, rate_func=linear)


        # dot_to_move.scale(1.25)

        self.wait()
        # Ok this is looking pretty guuud. #Book. 
        # Now, let me work on the progressive global diffusion a bit, then will go spend some time 
        # in adobe - Ok i think i want to break this apart though into two secense. let's do that. 



        #Ok so I want to fadd in rest of point and center my plot. Then diffuse everybody!!
        # self.play(self.frame.animate.reorient(0, 0, 0, (-0.07, 0.01, 0.0), 7.59), 
        #                 dots.animate.set_opacity(1.0), 
        #                 run_time=3.0)
        # self.wait()

        random_walks=[]
        np.random.seed(2)
        schedule2 = ScheduleLogLinear(N=100, sigma_min=0.02, sigma_max=0.09) 
        sigmas100=schedule2.sample_sigmas(99)
        sigmas100=(sigmas100.numpy()[::-1]).reshape(-1,1)
        for i in range(100):
            # rw=0.07*np.random.randn(100,2) #Uniform steps
            rw=sigmas100*np.random.randn(100,2) #Real noise schedule(scaled down)
            rw[0]=np.array([0,0]) #make be the starting point
            # rw[-1]=np.array([0.08, -0.02])
            rw=np.cumsum(rw,axis=0) 
            rw=np.hstack((rw, np.zeros((len(rw), 1))))
            rw_shifted=rw+np.array([batch[i][0], batch[i][1], 0])
            random_walks.append(rw_shifted)

        # maybe we actually totally lost the spiral for this animation? That might make for a more dramatic
        # bringing stuff back together phase

        traced_paths=VGroup()
        for idx, d in enumerate(dots): 
            # tp = TracedPath(d.get_center, stroke_color=YELLOW, stroke_width=2)
            # if idx != 75:  # Skip the already traced dot
            tp = CustomTracedPath(
                    d.get_center, 
                    stroke_color=CHILL_BROWN, 
                    stroke_width=2.0,
                    opacity_range=(0.3, 0.3),
                    fade_length=90
                )

            traced_path.set_fill(opacity=0)
            traced_paths.add(tp)
            # tp.set_opacity(0.2)
            # tp.set_fill(opacity=0)
            # traced_path.add(tp)
        self.add(traced_paths)

        remaining_indices=np.concatenate((np.arange(75), np.arange(76,len(batch))))    


        start_orientation=[0, 0, 0, (-0.07, 0.01, 0.0), 7.59]
        # end_orientation=[0, 0, 0, (0.23, -0.24, 0.0), 14.98]
        end_orientation=[0, 0, 0, (0.29, -0.21, 0.0), 12.94]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=100)

        self.wait()

        #A bit wide but captures everything
        self.frame.reorient(0, 0, 0, (-0.07, -0.49, 0.0), 12.54)

        for step in range(1, 100):
            self.play(*[dots[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in range(len(dots))], run_time=0.1, rate_func=linear)
            self.wait(0.2)

        self.wait()

        # remaining_indices=np.concatenate((np.arange(75), np.arange(76,len(batch))))
        # for step in range(100): #100
        #     self.play(*[dots[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in remaining_indices], 
        #              # self.frame.animate.reorient(*interp_orientations[step]), 
        #              run_time=0.1, rate_func=linear)

        #     #Kinda hacky but just try removing these after first step for now - that first path is distracting for big animation
        #     self.remove(dot_history)
        #     self.remove(dot_to_move)
        #     self.remove(traced_path)
        #     self.remove(dots[75])

        # self.wait()
        
        # for tp in traced_paths: tp.stop_tracing()



        # for tp in traced_paths: tp.remove_last_segment()

        # traced_paths[0].segments


        #Now play random walk backwards and zoom back in! Don't forget to remove traced paths as we go backwards
        #Reverse process works in interactive mode but not when rendering
        # I'll ask claude later or just play the other clip backwards. 
        # self.wait()
        # for j, step in enumerate(range(99, -1, -1)): #99
        #     self.play(*[dots[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in remaining_indices], 
        #              self.frame.animate.reorient(*interp_orientations[step]), 
        #              run_time=0.1, rate_func=linear)
        #     for tp in traced_paths:
        #         tp.remove_last_segment()
        #         if j==0: tp.remove_last_segment() #Bug patch
        #     # self.wait(0.01) #Hmm maybe adding a wait here will help???
        # self.add(dots[75])


        self.wait()



        # Ok ending cleanly on the simple spiral is probably nice here, I can start a new scene - this will help with cleanup etc.  


        self.wait(20)
        self.embed()





# class p48_51_deprecated_2(InteractiveScene):
#     def construct(self):
#         '''
#         Ok ok ok need to do a direct transition from p47b after fading out all the traces etc -> then bring
#         in the full vector field - I think this is going to be dope!
#         '''
#         batch_size=2130
#         dataset = Swissroll(np.pi/2, 5*np.pi, 100)
#         loader = DataLoader(dataset, batch_size=batch_size)
#         batch=next(iter(loader)).numpy()

#         axes = Axes(
#             x_range=[-1.2, 1.2, 0.5],
#             y_range=[-1.2, 1.2, 0.5],
#             height=7,
#             width=7,
#             axis_config={
#                 "color": CHILL_BROWN, 
#                 "stroke_width": 2,
#                 "include_tip": True,
#                 "include_ticks": True,
#                 "tick_size": 0.06,
#                 "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
#             }
#         )

#         # Create extended axes with SAME center point and proportional scaling
#         extended_axes = Axes(
#             x_range=[-2.0, 2.0, 0.5],    # Extended range
#             y_range=[-2.0, 2.0, 0.5],    # Extended range
#             height=7 * (4.0/2.4),        # Scale height proportionally: original_height * (new_range/old_range)
#             width=7 * (4.0/2.4),         # Scale width proportionally: original_width * (new_range/old_range)
#             axis_config={"stroke_width": 0}  # Make invisible
#         )

#         # Move extended axes to same position as original axes
#         extended_axes.move_to(axes.get_center())


#         dots = VGroup()
#         for point in batch:
#             # Map the point coordinates to the axes
#             screen_point = axes.c2p(point[0], point[1])
#             dot = Dot(screen_point, radius=0.04)
#             # dot.set_color(YELLOW)
#             dots.add(dot)
#         dots.set_color(YELLOW)
#         dots.set_opacity(0.3)

#         i=75
#         dot_to_move=dots[i].copy()
#         traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=3.5, 
#                                       opacity_range=(0.25, 0.9), fade_length=15)
#         # traced_path.set_opacity(0.5)
#         traced_path.set_fill(opacity=0)


#         np.random.seed(485) #485 is nice, 4 is maybe best so far, #52 is ok
#         random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
#         random_walk[0]=np.array([0.2, 0.12]) #make first step go up and to the right
#         # random_walk[-1]=np.array([0.15, -0.04])
#         random_walk[-1]=np.array([0.19, -0.05])
#         random_walk=np.cumsum(random_walk,axis=0) 

#         random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
#         random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        
#         dot_history=VGroup()
#         dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
#         # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)
#         # dot_to_move.move_to(axes.c2p(*random_walk_shifted[1]))
#         traced_path.update_path(0.1)

#         for j in range(100):
#             dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
#             dot_to_move.move_to(axes.c2p(*random_walk_shifted[j]))
#             traced_path.update_path(0.1)
#             # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)
#         traced_path.stop_tracing()

#         dot_to_move.set_opacity(1.0)

#         #Ok let me try to get all the big elements in here
#         x100=Tex('x_{100}', font_size=24).set_color(YELLOW)
#         x100.next_to(dot_to_move, 0.07*UP+0.001*RIGHT)

#         x0=Tex('x_{0}', font_size=24).set_color('#00FFFF')
#         x0.next_to(dots[i], 0.2*UP)
#         dots[i].set_color('#00FFFF').set_opacity(1.0)

#         arrow_x100_to_x0 = Arrow(
#             start=dot_to_move.get_center(),
#             end=dots[i].get_center(),
#             thickness=1,
#             tip_width_ratio=5, 
#             buff=0.025  # Small buffer so arrow doesn't overlap the dots
#         )
#         arrow_x100_to_x0.set_color('#00FFFF')
#         arrow_x100_to_x0.set_opacity(0.6)


#         arrow_x100_to_x99 = Arrow(
#             start=dot_to_move.get_center(),
#             end=[4.739921625933185, 2.8708813273028455, 0], #Just pul in from previous paragraph, kinda hacky but meh. ,
#             thickness=1.5,
#             tip_width_ratio=5, 
#             buff=0.04  # Small buffer so arrow doesn't overlap the dots
#         )
#         arrow_x100_to_x99.set_color(CHILL_BROWN)
#         # arrow_x100_to_x99.set_opacity(0.6)


#         self.frame.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69)
#         self.add(axes, dots, traced_path, dot_to_move)
#         self.add(x100,  x0, arrow_x100_to_x0, arrow_x100_to_x99)
#         self.wait()

#         # Ok so the continuity to think/worry about here is the brown arrow! Now I'm a bit worried about it's angle - hmm 
#         # Let's see how it shakes out. 
#         # I think first it's Fading everythig except that data and brown line (maybe scale of brown arrow changes)
#         # I might beg able to get away with some updates to the brown arrows angle on a zoom out as I add stuff, we'll see. 

#         self.play(FadeOut(traced_path), FadeOut(dot_to_move), FadeOut(x100), FadeOut(x0), FadeOut(arrow_x100_to_x0), 
#                  dots.animate.set_opacity(1.0).set_color(YELLOW), run_time=1.5)

#         # Ok ok ok so I now in need some vector fields. These come from trained models. Do I want to import the model 
#         # and sample from it here? Or do I want to exprot the vector fields? 
#         # I think it would be nice to fuck with the density etc in manim, so maybe we get a little aggressive and 
#         # try to import the full model? 
#         # Lets see here....
#         # Hmm kinda unclear if this is going to work on mac/CPU -> i guess it's worth a try? Pretty sure I can't train w/o cuda. 

#         model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_20_1.pt')

#         schedule = ScheduleLogLinear(N=256, sigma_min=0.01, sigma_max=10) #N=200
#         bound=2.0 #Need to match extended axes bro
#         num_heatmap_steps=30
#         grid=[]
#         for i, x in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
#             for j, y in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
#                 grid.append([x,y])
#         grid=torch.tensor(grid).float()

#         gam=1
#         mu=0.01 #0.5 is DDPM
#         cfg_scale=0.0
#         cond=None
#         sigmas=schedule.sample_sigmas(256)
#         xt_history=[]
#         heatmaps=[]
#         eps=None

#         with torch.no_grad():
#             model.eval();
#             xt=torch.randn((batch_size,) + model.input_dims)*sigmas[0] #Scaling by sigma here matters a lot - why is that???

#             for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
#                 eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
#                 # eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
#                 sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
#                 eta = (sig_prev**2 - sig_p**2).sqrt()
#                 xt = xt - (sig - sig_p) * eps + eta * model.rand_input(xt.shape[0]).to(xt)
#                 xt_history.append(xt.numpy())
#                 heatmaps.append(model.forward(grid, sig, cond=None))

#         xt_history=np.array(xt_history)
#         self.wait()

#         # Ok nice glad i tried this! Seems like I can sample right in manim - that's great. 
#         # Ok now let's draw some arrows, and then try to figure out how to bring thme in as a nice continuous
#         # extension of the single arrow I have. 
#         final_vectors = heatmaps[-1].detach().numpy()  # Shape should be (num_heatmap_steps^2, 2)

#         sigma_index=-1
#         def vector_function_direct(coords_array):
#             # print(coords_array.shape)
#             res=model.forward(torch.tensor(coords_array).float(), sigmas[sigma_index], cond=None)
#             return -res.detach().numpy()


#         #This is a great idea, should definitely try
#         def animate_vector_field_radially():
#             # Get brown arrow position in coordinate system
#             brown_pos = axes.p2c(arrow_x100_to_x99.get_start())
            
#             # Group vectors by distance from brown arrow
#             vector_groups = {}
#             for vector_mob in vector_field.submobjects:
#                 if hasattr(vector_mob, 'get_center'):
#                     vec_pos = axes.p2c(vector_mob.get_center())
#                     distance = np.linalg.norm(np.array(vec_pos[:2]) - np.array(brown_pos[:2]))
#                     dist_key = int(distance * 10)  # Group by distance intervals
#                     if dist_key not in vector_groups:
#                         vector_groups[dist_key] = []
#                     vector_groups[dist_key].append(vector_mob)
            
#             # Animate groups in order of distance
#             for dist_key in sorted(vector_groups.keys()):
#                 group = VGroup(*vector_groups[dist_key])
#                 self.play(FadeIn(group), run_time=0.2)
            
#             return vector_field



#         # Create the VectorField - note that it needs your axes as the coordinate_system
#         vector_field = VectorField(
#             func=vector_function_direct, #vector_function,  # or vector_function_interpolated for smoother results
#             coordinate_system=extended_axes,  # This is your existing axes object
#             density=3.0,  # Controls spacing between vectors (higher = more dense)
#             stroke_width=3,
#             stroke_opacity=0.7,
#             tip_width_ratio=4,
#             tip_len_to_width=0.01,
#             max_vect_len_to_step_size=0.7,  # Controls maximum vector length
#             # color_map_name="viridis",  # Color map for magnitude-based coloring
#             color=CHILL_BROWN
#         )

#         self.frame.reorient(0, 0, 0, (-0.06, 0.09, 0.0), 8.31)
#         self.play(FadeIn(vector_field), run_time=2.0)
#         self.wait()


#         # individual_arrows=extract_individual_arrows(vector_field)
#         # Ok so we still need to figure out a smooth transition between the single individual vector and the vector field
#         # If I can extract a single vector from the field I could do a replacement transform as I roll in rest of vectors and zoom 
#         # out. Before I do that though, let me make sure the time varying version of the vector field looks good. 
#         # I need to increment sigma_index from 0 to 255 and redraw the field each time. Claude?
#         # Ok so there's some cool time varying stuff I could do -> let me try the super simple loop appraoch first though 
#         # Just need to validate I get the fun/interesting temporal behavior when animating this way. 
#         # Hmm maybe not actually?

#         self.remove(vector_field)


#         time_tracker = ValueTracker(0.0)  # Start at time 0

#         def vector_function_with_tracker(coords_array):
#             """Vector function that uses the ValueTracker for time"""
#             current_time = time_tracker.get_value()
#             max_time = 8.0  # Map time 0-8 to sigma indices 0-255
#             sigma_idx = int(np.clip(current_time * 255 / max_time, 0, 255))
            
#             try:
#                 res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=None)
#                 return -res.detach().numpy()
#             except:
#                 return np.zeros((len(coords_array), 2))

#         # Create a custom VectorField that updates based on the tracker
#         class TrackerControlledVectorField(VectorField):
#             def __init__(self, time_tracker, **kwargs):
#                 self.time_tracker = time_tracker
#                 super().__init__(**kwargs)
                
#                 # Add updater that triggers when tracker changes
#                 self.add_updater(self.update_from_tracker)
            
#             def update_from_tracker(self, mob, dt):
#                 """Update vectors when tracker value changes"""
#                 # Only update if tracker value has changed significantly
#                 current_time = self.time_tracker.get_value()
#                 if not hasattr(self, '_last_time') or abs(current_time - self._last_time) > 0.01:
#                     self._last_time = current_time
#                     self.update_vectors()  # Redraw vectors with new time

#         # Create the tracker-controlled vector field
#         vector_field = TrackerControlledVectorField(
#             time_tracker=time_tracker,
#             func=vector_function_with_tracker,
#             coordinate_system=extended_axes,
#             density=3.0,
#             stroke_width=3,
#             stroke_opacity=0.7,
#             tip_width_ratio=4,
#             tip_len_to_width=0.01,
#             max_vect_len_to_step_size=0.7,
#             color=CHILL_BROWN
#         )

#         self.add(vector_field)

#         self.play(time_tracker.animate.set_value(8.0), run_time=6.0)
#         self.play(time_tracker.animate.set_value(0.0), run_time=4.0)

#         self.wait()




#         self.wait(20)
#         self.embed()


# class p48_51_deprecated(InteractiveScene):
#     def construct(self):
#         '''
#         Ok ok ok need to do a direct transition from p47b after fading out all the traces etc -> then bring
#         in the full vector field - I think this is going to be dope!
#         '''
#         batch_size=2130
#         dataset = Swissroll(np.pi/2, 5*np.pi, 100)
#         loader = DataLoader(dataset, batch_size=batch_size)
#         batch=next(iter(loader)).numpy()

#         axes = Axes(
#             x_range=[-1.2, 1.2, 0.5],
#             y_range=[-1.2, 1.2, 0.5],
#             height=7,
#             width=7,
#             axis_config={
#                 "color": CHILL_BROWN, 
#                 "stroke_width": 2,
#                 "include_tip": True,
#                 "include_ticks": True,
#                 "tick_size": 0.06,
#                 "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
#             }
#         )

#         dots = VGroup()
#         for point in batch:
#             # Map the point coordinates to the axes
#             screen_point = axes.c2p(point[0], point[1])
#             dot = Dot(screen_point, radius=0.04)
#             # dot.set_color(YELLOW)
#             dots.add(dot)
#         dots.set_color(YELLOW)
#         dots.set_opacity(0.3)

#         i=75
#         dot_to_move=dots[i].copy()
#         traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=3.5, 
#                                       opacity_range=(0.25, 0.9), fade_length=15)
#         # traced_path.set_opacity(0.5)
#         traced_path.set_fill(opacity=0)


#         np.random.seed(485) #485 is nice, 4 is maybe best so far, #52 is ok
#         random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
#         random_walk[0]=np.array([0.2, 0.12]) #make first step go up and to the right
#         # random_walk[-1]=np.array([0.15, -0.04])
#         random_walk[-1]=np.array([0.19, -0.05])
#         random_walk=np.cumsum(random_walk,axis=0) 

#         random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
#         random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        
#         dot_history=VGroup()
#         dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
#         # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)
#         # dot_to_move.move_to(axes.c2p(*random_walk_shifted[1]))
#         traced_path.update_path(0.1)

#         for j in range(100):
#             dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
#             dot_to_move.move_to(axes.c2p(*random_walk_shifted[j]))
#             traced_path.update_path(0.1)
#             # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)
#         traced_path.stop_tracing()

#         dot_to_move.set_opacity(1.0)

#         #Ok let me try to get all the big elements in here
#         x100=Tex('x_{100}', font_size=24).set_color(YELLOW)
#         x100.next_to(dot_to_move, 0.07*UP+0.001*RIGHT)

#         x0=Tex('x_{0}', font_size=24).set_color('#00FFFF')
#         x0.next_to(dots[i], 0.2*UP)
#         dots[i].set_color('#00FFFF').set_opacity(1.0)

#         arrow_x100_to_x0 = Arrow(
#             start=dot_to_move.get_center(),
#             end=dots[i].get_center(),
#             thickness=1,
#             tip_width_ratio=5, 
#             buff=0.025  # Small buffer so arrow doesn't overlap the dots
#         )
#         arrow_x100_to_x0.set_color('#00FFFF')
#         arrow_x100_to_x0.set_opacity(0.6)


#         arrow_x100_to_x99 = Arrow(
#             start=dot_to_move.get_center(),
#             end=[4.739921625933185, 2.8708813273028455, 0], #Just pul in from previous paragraph, kinda hacky but meh. ,
#             thickness=1.5,
#             tip_width_ratio=5, 
#             buff=0.04  # Small buffer so arrow doesn't overlap the dots
#         )
#         arrow_x100_to_x99.set_color(CHILL_BROWN)
#         # arrow_x100_to_x99.set_opacity(0.6)


#         self.frame.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69)
#         self.add(axes, dots, traced_path, dot_to_move)
#         self.add(x100,  x0, arrow_x100_to_x0, arrow_x100_to_x99)
#         self.wait()

#         # Ok so the continuity to think/worry about here is the brown arrow! Now I'm a bit worried about it's angle - hmm 
#         # Let's see how it shakes out. 
#         # I think first it's Fading everythig except that data and brown line (maybe scale of brown arrow changes)
#         # I might beg able to get away with some updates to the brown arrows angle on a zoom out as I add stuff, we'll see. 

#         self.play(FadeOut(traced_path), FadeOut(dot_to_move), FadeOut(x100), FadeOut(x0), FadeOut(arrow_x100_to_x0), 
#                  dots.animate.set_opacity(1.0).set_color(YELLOW), run_time=1.5)

#         # Ok ok ok so I now in need some vector fields. These come from trained models. Do I want to import the model 
#         # and sample from it here? Or do I want to exprot the vector fields? 
#         # I think it would be nice to fuck with the density etc in manim, so maybe we get a little aggressive and 
#         # try to import the full model? 
#         # Lets see here....
#         # Hmm kinda unclear if this is going to work on mac/CPU -> i guess it's worth a try? Pretty sure I can't train w/o cuda. 

#         model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_20_1.pt')

#         schedule = ScheduleLogLinear(N=256, sigma_min=0.01, sigma_max=10) #N=200
#         bound=1.5
#         num_heatmap_steps=30
#         grid=[]
#         for i, x in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
#             for j, y in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
#                 grid.append([x,y])
#         grid=torch.tensor(grid).float()

#         gam=1
#         mu=0.01 #0.5 is DDPM
#         cfg_scale=0.0
#         cond=None
#         sigmas=schedule.sample_sigmas(256)
#         xt_history=[]
#         heatmaps=[]
#         eps=None

#         with torch.no_grad():
#             model.eval();
#             xt=torch.randn((batch_size,) + model.input_dims)*sigmas[0] #Scaling by sigma here matters a lot - why is that???

#             for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
#                 eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
#                 # eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
#                 sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
#                 eta = (sig_prev**2 - sig_p**2).sqrt()
#                 xt = xt - (sig - sig_p) * eps + eta * model.rand_input(xt.shape[0]).to(xt)
#                 xt_history.append(xt.numpy())
#                 heatmaps.append(model.forward(grid, sig, cond=None))

#         xt_history=np.array(xt_history)
#         self.wait()

#         # Ok nice glad i tried this! Seems like I can sample right in manim - that's great. 
#         # Ok now let's draw some arrows, and then try to figure out how to bring thme in as a nice continuous
#         # extension of the single arrow I have. 
#         final_vectors = heatmaps[-1].detach().numpy()  # Shape should be (num_heatmap_steps^2, 2)

#         sigma_index=-1
#         def vector_function_direct(coords_array):
#             print(coords_array.shape)
#             res=model.forward(torch.tensor(coords_array).float(), sigmas[sigma_index], cond=None)
#             return -res.detach().numpy()


#         # Create interpolation function for the vector field
#         def vector_function(coords_array):
#             """
#             Function that takes an array of coordinates and returns corresponding vectors
#             coords_array: shape (N, 2) or (N, 3) - array of [x, y] or [x, y, z] coordinates
#             Returns: array of shape (N, 2) with [vx, vy] vectors (z component handled automatically)
#             """
#             result = np.zeros((len(coords_array), 2))
            
#             for i, coord in enumerate(coords_array):
#                 x, y = coord[0], coord[1]  # Take only x, y coordinates
                
#                 # Find the closest grid point to interpolate from
#                 distances = np.linalg.norm(grid.numpy() - np.array([x, y]), axis=1)
#                 closest_idx = np.argmin(distances)
                
#                 # Get the vector at the closest grid point
#                 vector = final_vectors[closest_idx]
#                 result[i] = vector
            
#             return -result #Reverse direction


#         #This is a great idea, should definitely try
#         def animate_vector_field_radially():
#             # Get brown arrow position in coordinate system
#             brown_pos = axes.p2c(arrow_x100_to_x99.get_start())
            
#             # Group vectors by distance from brown arrow
#             vector_groups = {}
#             for vector_mob in vector_field.submobjects:
#                 if hasattr(vector_mob, 'get_center'):
#                     vec_pos = axes.p2c(vector_mob.get_center())
#                     distance = np.linalg.norm(np.array(vec_pos[:2]) - np.array(brown_pos[:2]))
#                     dist_key = int(distance * 10)  # Group by distance intervals
#                     if dist_key not in vector_groups:
#                         vector_groups[dist_key] = []
#                     vector_groups[dist_key].append(vector_mob)
            
#             # Animate groups in order of distance
#             for dist_key in sorted(vector_groups.keys()):
#                 group = VGroup(*vector_groups[dist_key])
#                 self.play(FadeIn(group), run_time=0.2)
            
#             return vector_field



#         # Create the VectorField - note that it needs your axes as the coordinate_system
#         vector_field = VectorField(
#             func=vector_function_direct, #vector_function,  # or vector_function_interpolated for smoother results
#             coordinate_system=axes,  # This is your existing axes object
#             density=4.0,  # Controls spacing between vectors (higher = more dense)
#             stroke_width=3,
#             stroke_opacity=0.7,
#             tip_width_ratio=4,
#             tip_len_to_width=0.01,
#             max_vect_len_to_step_size=0.7,  # Controls maximum vector length
#             # color_map_name="viridis",  # Color map for magnitude-based coloring
#             color=CHILL_BROWN
#         )

#         self.frame.reorient(0, 0, 0, (-0.06, 0.09, 0.0), 8.31)
#         self.play(FadeIn(vector_field), run_time=2.0)
#         self.wait()


#         self.wait()




#         self.wait(20)
#         self.embed()



