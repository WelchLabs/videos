from manimlib import *
from functools import partial

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
WELCH_ASSET_PATH='/Users/stephen/manim/videos/welch_assets'

    # def copy_frame_positioning(self):
    #     frame = self.frame
    #     center = frame.get_center()
    #     height = frame.get_height()
    #     angles = frame.get_euler_angles()

    #     call = f"reorient("
    #     theta, phi, gamma = (angles / DEG).astype(int)
    #     call += f"{theta}, {phi}, {gamma}"
    #     if any(center != 0):
    #         call += f", {tuple(np.round(center, 2))}"
    #     if height != FRAME_HEIGHT:
    #         call += ", {:.2f}".format(height)
    #     call += ")"
    #     pyperclip.copy(call)


def generate_nice_ticks(min_val, max_val, min_ticks=3, max_ticks=16, ignore=[0]):
    """
    Generate a list of nice-looking tick values between min_val and max_val,
    and return extended range values for the full axis.
    
    Args:
        min_val (float): Minimum value for the data range
        max_val (float): Maximum value for the data range
        min_ticks (int): Minimum number of ticks desired
        max_ticks (int): Maximum number of ticks desired
        ignore (list): List of values to exclude from the ticks
        
    Returns:
        tuple: (tick_values, axis_min, axis_max)
            - tick_values (list): A list of tick values
            - axis_min (float): Suggested minimum value for the axis (one tick before min_val)
            - axis_max (float): Suggested maximum value for the axis (one tick after max_val)
    """
    # Ensure min_val < max_val
    if min_val > max_val:
        min_val, max_val = max_val, min_val
        
    # Handle case where min_val and max_val are equal or very close
    if abs(max_val - min_val) < 1e-10:
        # Create a small range around the value
        min_val = min_val - 1
        max_val = max_val + 1
    
    # Find the appropriate order of magnitude for the tick spacing
    range_val = max_val - min_val
    power = np.floor(np.log10(range_val))
    
    # Try different multiples of the base power of 10
    possible_step_sizes = [10**power, 5 * 10**(power-1), 2 * 10**(power-1), 10**(power-1)]
    
    # Find the first step size that gives us fewer than max_ticks
    chosen_step = possible_step_sizes[0]  # Default to the largest step
    
    for step in possible_step_sizes:
        # Calculate how many ticks we'd get with this step size
        first_tick = np.ceil(min_val / step) * step
        last_tick = np.floor(max_val / step) * step
        
        # Count ticks, excluding ignored values
        num_ticks = 0
        current = first_tick
        while current <= last_tick * (1 + 1e-10):
            if not any(abs(current - ignored_val) < 1e-10 for ignored_val in ignore):
                num_ticks += 1
            current += step
        
        if min_ticks <= num_ticks <= max_ticks:
            chosen_step = step
            break
        elif num_ticks > max_ticks:
            # If we have too many ticks, stop and use the previous step size
            break
    
    # Calculate the first tick at or below min_val
    first_tick = np.floor(min_val / chosen_step) * chosen_step
    
    # Calculate the last tick at or above max_val
    last_tick = np.ceil(max_val / chosen_step) * chosen_step
    
    # Calculate one tick before first_tick for axis_min
    axis_min = first_tick - chosen_step
    
    # Calculate one tick after last_tick for axis_max
    axis_max = last_tick + chosen_step
    
    # Generate the tick values that fall within the data range, excluding ignored values
    ticks = []
    current = np.ceil(min_val / chosen_step) * chosen_step
    
    while current <= max_val * (1 + 1e-10):  # Add a small epsilon to handle floating point errors
        # Only add the tick if it's not in the ignore list
        if not any(abs(current - ignored_val) < 1e-10 for ignored_val in ignore):
            ticks.append(float(current))  # Convert to float to avoid numpy types
        current += chosen_step
    
    # If we still have too few ticks, try the next smaller step size
    if len(ticks) < min_ticks and possible_step_sizes.index(chosen_step) < len(possible_step_sizes) - 1:
        return generate_nice_ticks(min_val, max_val, min_ticks, max_ticks, ignore)
    
    return ticks, float(axis_min), float(axis_max)


class WelchXAxis(VGroup):
    def __init__(
        self,
        x_min=0,
        x_max=6, 
        x_ticks=[1, 2, 3, 4, 5],  # Default tick values
        x_tick_height=0.15,        # Default tick height
        x_label_font_size=24,     # Default font size
        stroke_width=3,           # Default stroke width
        color=CHILL_BROWN,        # Default color (using predefined BROWN)
        arrow_tip_scale=0.1, 
        axis_length_on_canvas=5,
        include_tip=True,
        **kwargs
    ):
        
        VGroup.__init__(self, **kwargs)

        # Store parameters
        self.x_ticks = x_ticks
        self.x_tick_height = x_tick_height
        self.x_label_font_size = x_label_font_size
        self.stroke_width = stroke_width
        self.axis_color = color
        self.arrow_tip_scale=arrow_tip_scale
        self.x_min = x_min
        self.x_max = x_max
        self.axis_length_on_canvas=axis_length_on_canvas
        self.include_tip=include_tip

        self.axis_to_canvas_scale=(self.x_max-self.x_min)/axis_length_on_canvas
        self.x_ticks_scaled=(np.array(x_ticks)-self.x_min)/self.axis_to_canvas_scale

        # Create the basic components
        self._create_axis_line()
        self._create_ticks()
        self._create_labels()
        
    def _create_axis_line(self):
        
        # Create a line for the x-axis
        axis_line = Line(
            start=np.array([0, 0, 0]),
            end=np.array([self.axis_length_on_canvas, 0, 0]),
            color=self.axis_color,
            stroke_width=self.stroke_width
        )
        if self.include_tip:
            arrow_tip=SVGMobject(WELCH_ASSET_PATH+'/welch_arrow_tip_1.svg')
            arrow_tip.scale(self.arrow_tip_scale)
            arrow_tip.move_to([self.axis_length_on_canvas, 0, 0])
            axis_line = VGroup(axis_line, arrow_tip)

        self.add(axis_line)
        
    def _create_ticks(self):
        self.ticks = VGroup()
        
        for x_val in self.x_ticks_scaled:
            tick = Line(
                start=np.array([x_val, 0, 0]),
                end=np.array([x_val, -self.x_tick_height, 0]),  # Ticks extend downward
                color=self.axis_color,
                stroke_width=self.stroke_width
            )
            self.ticks.add(tick)
            
        self.add(self.ticks)
        
    def _create_labels(self):
        self.labels = VGroup()
        
        for x_val, x_val_label in zip(self.x_ticks_scaled, self.x_ticks):
            # In 3B1B's manim, use TexMobject instead of MathTex
            label = Tex(str(round(x_val_label, 4)))
            label.scale(self.x_label_font_size / 48)  # Approximate scaling
            label.set_color(self.axis_color)
            label.next_to(
                np.array([x_val, -self.x_tick_height, 0]),
                DOWN,
                buff=0.1
            )
            self.labels.add(label)
            
        self.add(self.labels)
    
    # Helper method to get the axis line
    def get_axis_line(self):
        return self.axis_line
    
    # Helper method to get ticks
    def get_ticks(self):
        return self.ticks
    
    # Helper method to get labels
    def get_labels(self):
        return self.labels

    def map_to_canvas(self, value, axis_start=0):
        value_scaled=(value-self.x_min)/(self.x_max-self.x_min)
        return (value_scaled+axis_start)*self.axis_length_on_canvas


class WelchYAxis(VGroup):
    def __init__(
        self,
        y_min=0,
        y_max=6, 
        y_ticks=[1, 2, 3, 4, 5],  # Default tick values
        y_tick_width=0.15,        # Default tick width
        y_label_font_size=24,     # Default font size
        stroke_width=3,           # Default stroke width
        color=CHILL_BROWN,        # Default color
        arrow_tip_scale=0.1,
        axis_length_on_canvas=5,
        include_tip=True,
        **kwargs
    ):
        VGroup.__init__(self, **kwargs)
        
        # Store parameters
        self.y_ticks = y_ticks
        self.y_tick_width = y_tick_width
        self.y_label_font_size = y_label_font_size
        self.stroke_width = stroke_width
        self.axis_color = color
        self.arrow_tip_scale = arrow_tip_scale
        self.y_min = y_min
        self.y_max = y_max
        self.axis_length_on_canvas = axis_length_on_canvas
        self.include_tip=include_tip
        
        self.axis_to_canvas_scale = (self.y_max - self.y_min) / axis_length_on_canvas
        self.y_ticks_scaled = (np.array(y_ticks)-self.y_min)/ self.axis_to_canvas_scale
        
        # Create the basic components
        self._create_axis_line()
        self._create_ticks()
        self._create_labels()
        
    def _create_axis_line(self):
        # Create a line for the y-axis
        axis_line = Line(
            start=np.array([0, 0, 0]),
            end=np.array([0, self.axis_length_on_canvas, 0]),
            color=self.axis_color,
            stroke_width=self.stroke_width
        )
        
        # Add SVG arrow tip at the end
        if self.include_tip:
            arrow_tip = SVGMobject(WELCH_ASSET_PATH+'/welch_arrow_tip_1.svg')
            arrow_tip.scale(self.arrow_tip_scale)
            arrow_tip.move_to([0, self.axis_length_on_canvas, 0])
            # Rotate the arrow tip to point upward
            arrow_tip.rotate(PI/2)  # Rotate 90 degrees to point up
            axis_line = VGroup(axis_line, arrow_tip)


        self.add(axis_line)
        
    def _create_ticks(self):
        self.ticks = VGroup()
        
        for y_val in self.y_ticks_scaled:
            tick = Line(
                start=np.array([0, y_val, 0]),
                end=np.array([-self.y_tick_width, y_val, 0]),  # Ticks extend to the left
                color=self.axis_color,
                stroke_width=self.stroke_width
            )
            self.ticks.add(tick)
            
        self.add(self.ticks)
        
    def _create_labels(self):
        self.labels = VGroup()
        
        for y_val, y_val_label in zip(self.y_ticks_scaled, self.y_ticks):
            # Use Tex for labels
            label = Tex(str(round(y_val_label,5)))
            label.scale(self.y_label_font_size / 48)  # Approximate scaling
            label.set_color(self.axis_color)
            label.next_to(
                np.array([-self.y_tick_width, y_val, 0]),
                LEFT,
                buff=0.1
            )
            self.labels.add(label)
            
        self.add(self.labels)
    
    # Helper method to get the axis line
    def get_axis_line(self):
        return self.axis_line
    
    # Helper method to get ticks
    def get_ticks(self):
        return self.ticks
    
    # Helper method to get labels
    def get_labels(self):

        return self.labels
    def map_to_canvas(self, value, axis_start=0):
        value_scaled=(value-self.y_min)/(self.y_max-self.y_min)
        return (value_scaled+axis_start)*self.axis_length_on_canvas









