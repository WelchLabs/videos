from welch_paths import welch_path
import manimgx as m
from welch_scene import WelchScene, image_at_height
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from functools import partial

 
CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'

graphics_dir=welch_path('backprop_3/graphics/')

def get_relu_joint(weight_1, weight_2, bias, extent=1):
    if m.np.abs(weight_2) < 1e-8:
        x_intercept = -bias / weight_1
        return [[x_intercept, -extent], [x_intercept, extent]] if -extent <= x_intercept <= extent else []
    elif m.np.abs(weight_1) < 1e-8:
        y_intercept = -bias / weight_2
        return [[-extent, y_intercept], [extent, y_intercept]] if -extent <= y_intercept <= extent else []
    else:
        points = []
        for x in [-extent, extent]:
            y = (-x * weight_1 - bias) / weight_2
            if -extent <= y <= extent: points.append([x, y])
        for y in [-extent, extent]:
            x = (-y * weight_2 - bias) / weight_1
            if -extent <= x <= extent: points.append([x, y])
        unique_points = []
        for p in points:
            is_duplicate = False
            for existing in unique_points:
                if abs(p[0] - existing[0]) < 1e-8 and abs(p[1] - existing[1]) < 1e-8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(p)
        return unique_points

def line_from_joint_points_1(joint_points, stroke_width=3):
    if joint_points:
        # Create 3D points for the joint line
        joint_3d_points = []
        for point in joint_points:
            x, y = point
            z = 0
            joint_3d_points.append([x, y, z])
        
        if len(joint_3d_points) >= 2:
            joint_line = m.DashedLine(
                start=[joint_points[0][0], joint_points[0][1], 0],
                end=[joint_points[1][0], joint_points[1][1], 0],
                color=m.WHITE,
                stroke_width=stroke_width,
                dash_length=0.05
            )
            return joint_line


def surface_func_general(u, v, w1, w2, b, viz_scale=0.5):
    linear_output = w1 * u + w2 * v + b
    relu_output = max(0, linear_output)
    z = relu_output * viz_scale 
    return m.np.array([u, v, z])


def surface_func_second_layer(u, v, w1, b1, w2, b2, neuron_idx=0, viz_scale=0.5):
    """
    Surface function for second layer neurons that combines first layer outputs.
    
    Args:
        u, v: Input coordinates (-1 to 1)
        w1: First layer weights (2x2 matrix)
        b1: First layer biases (2 element array)
        w2: Second layer weights (2x2 matrix) 
        b2: Second layer biases (2 element array)
        neuron_idx: Which second layer neuron (0 or 1)
        viz_scale: Scaling factor for visualization
    """
    
    # First layer neuron 1 output
    linear_output_1 = w1[0,0] * u + w1[0,1] * v + b1[0]
    relu_output_1 = max(0, linear_output_1)
    
    # First layer neuron 2 output  
    linear_output_2 = w1[1,0] * u + w1[1,1] * v + b1[1]
    relu_output_2 = max(0, linear_output_2)
    
    # Second layer neuron computation
    second_layer_input = w2[neuron_idx,0] * relu_output_1 + w2[neuron_idx,1] * relu_output_2 + b2[neuron_idx]
    second_layer_output = max(0, second_layer_input)
    
    # Use output as z-coordinate
    z = second_layer_output * viz_scale
    
    return m.np.array([u, v, z])


def get_second_layer_joints(w1, b1, w2, b2, neuron_idx=0, extent=1):
    """
    Calculate joint lines for second layer neurons.
    
    Returns a list of line segments representing the boundaries where
    the second layer neuron's output changes behavior.
    """
    
    joint_lines = []
    
    # First, get the first layer joint lines (these create discontinuities)
    joint_points_1 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent)
    joint_points_2 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent)
    
    # Add first layer joints as they create surface discontinuities
    if joint_points_1:
        joint_lines.append(joint_points_1)
    if joint_points_2:
        joint_lines.append(joint_points_2)
    
    # Now find where the second layer neuron itself switches on/off
    # This happens when: w2[neuron_idx,0] * relu1 + w2[neuron_idx,1] * relu2 + b2[neuron_idx] = 0
    
    # We need to check this condition in each region defined by the first layer ReLUs
    regions = [
        (False, False),  # Both first layer neurons off
        (True, False),   # Only first neuron on
        (False, True),   # Only second neuron on  
        (True, True)     # Both neurons on
    ]
    
    for relu1_active, relu2_active in regions:
        # Calculate the second layer boundary in this region
        w2_eff = w2[neuron_idx, :]
        b2_eff = b2[neuron_idx]
        
        if relu1_active and relu2_active:
            # Both active: w2[0]*(w1[0,0]*u + w1[0,1]*v + b1[0]) + w2[1]*(w1[1,0]*u + w1[1,1]*v + b1[1]) + b2 = 0
            # Simplifies to: (w2[0]*w1[0,0] + w2[1]*w1[1,0])*u + (w2[0]*w1[0,1] + w2[1]*w1[1,1])*v + (w2[0]*b1[0] + w2[1]*b1[1] + b2) = 0
            eff_w1 = w2_eff[0] * w1[0,0] + w2_eff[1] * w1[1,0]
            eff_w2 = w2_eff[0] * w1[0,1] + w2_eff[1] * w1[1,1]
            eff_b = w2_eff[0] * b1[0] + w2_eff[1] * b1[1] + b2_eff
            
        elif relu1_active and not relu2_active:
            # Only first active: w2[0]*(w1[0,0]*u + w1[0,1]*v + b1[0]) + b2 = 0
            eff_w1 = w2_eff[0] * w1[0,0]
            eff_w2 = w2_eff[0] * w1[0,1]
            eff_b = w2_eff[0] * b1[0] + b2_eff
            
        elif not relu1_active and relu2_active:
            # Only second active: w2[1]*(w1[1,0]*u + w1[1,1]*v + b1[1]) + b2 = 0
            eff_w1 = w2_eff[1] * w1[1,0]
            eff_w2 = w2_eff[1] * w1[1,1]
            eff_b = w2_eff[1] * b1[1] + b2_eff
            
        else:
            # Neither active: b2 = 0
            if abs(b2_eff) < 1e-8:
                # Always on the boundary (degenerate case)
                continue
            else:
                # Never on the boundary in this region
                continue
        
        # Get joint points for this effective linear function
        region_joints = get_relu_joint(eff_w1, eff_w2, eff_b, extent)
        
        if region_joints:
            # Need to clip these joints to the actual region where this applies
            clipped_joints = clip_joint_to_region(region_joints, relu1_active, relu2_active, w1, b1, extent)
            if clipped_joints:
                joint_lines.append(clipped_joints)
    
    return joint_lines

def clip_joint_to_region(joint_points, relu1_active, relu2_active, w1, b1, extent):
    """
    Clip joint line to the region where the specified ReLU conditions hold.
    """
    if not joint_points or len(joint_points) < 2:
        return []
    
    # Sample points along the joint line
    start, end = joint_points[0], joint_points[1]
    clipped_points = []
    
    # Check multiple points along the line
    for t in m.np.linspace(0, 1, 100):
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        
        # Check if this point satisfies the region conditions
        linear1 = w1[0,0] * x + w1[0,1] * y + b1[0]
        linear2 = w1[1,0] * x + w1[1,1] * y + b1[1]
        
        relu1_here = linear1 > 0
        relu2_here = linear2 > 0
        
        if relu1_here == relu1_active and relu2_here == relu2_active:
            clipped_points.append([x, y])
    
    # Return start and end of the clipped segment
    if len(clipped_points) >= 2:
        return [clipped_points[0], clipped_points[-1]]
    else:
        return []

def create_second_layer_joint_lines(w1, b1, w2, b2, neuron_idx=0, extent=1):
    """
    Create Manim objects for all second layer joint lines.
    """
    joint_lines = get_second_layer_joints(w1, b1, w2, b2, neuron_idx, extent)
    
    manim_lines = []
    colors = [m.RED, BLUE, GREEN, YELLOW, m.PURPLE]  # Different colors for different joints
    
    for i, joint_points in enumerate(joint_lines):
        if len(joint_points) >= 2:
            color = colors[i % len(colors)]
            joint_line = m.DashedLine(
                start=[joint_points[0][0], joint_points[0][1], 0],
                end=[joint_points[1][0], joint_points[1][1], 0],
                color=color,
                stroke_width=3,
                dash_length=0.05
            )
            manim_lines.append(joint_line)
    
    return manim_lines


class plane_folding_sketch_1(WelchScene):
    def construct(self):

        #nice trained 2 hidden layer model. 
        w1=m.np.array([[-0.02866297,  1.6250265 ],
             [-1.3056537 ,  0.46831134]], dtype=m.np.float32)
        b1=m.np.array([-0.4677289,  1.0067637], dtype=m.np.float32)
        w2=m.np.array([[ 1.3398709 ,  0.68694556],
                     [-0.29886743, -1.8411286 ]], dtype=m.np.float32)
        b2=m.np.array([-0.7817721 ,  0.90856946], dtype=m.np.float32)
        w3=m.np.array([[ 1.8897862,  3.0432484],
                     [-1.7220999, -2.2057745]], dtype=m.np.float32)
        b3=m.np.array([-1.0249746 ,  0.61326534], dtype=m.np.float32)
        
        #Not using this flat map right now - but will probably want it
        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.scale_to_fit_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.scale_to_fit_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(m.ORIGIN)

        self.set_view(0, 0, 0, (0.03, -0.02, 0.0), 3.27)
        # self.add(map_img)
        # self.wait()

        surface_func_11=partial(surface_func_general, w1=w1[0,0], w2=w1[0,1], b=b1[0], viz_scale=0.3)
        bent_surface_11 = parametric_surface(surface_func_11, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts11=textured_surface(bent_surface_11, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts11, 0,0,0)
        ts11.set_opacity(0.75)
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.8)
        group_11=m.Group(ts11, joint_line_11)

        surface_func_12=partial(surface_func_general, w1=w1[1,0], w2=w1[1,1], b=b1[1], viz_scale=0.3)
        bent_surface_12 = parametric_surface(surface_func_12, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts12=textured_surface(bent_surface_12, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts12, 0,0,0)
        ts12.set_opacity(0.75)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.8)
        group_12=m.Group(ts12, joint_line_12)


        self.add(group_11)
        self.add(group_12)    
        group_12.move_to([0, 0, 1.2])
        self.wait()         

        # Ok scale/flip/add!
        # I almost want liek 2 sets of copies now?
        # One for each combination to come together?
        # Let me start with the first one and then see what's up. 


        surface_func_21 = partial(
            surface_func_second_layer, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=1, viz_scale=0.3
        )


        self.wait()

        bent_surface_21 = parametric_surface(surface_func_21, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts21 = textured_surface(bent_surface_21, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts21, 0,0,0)
        ts21.set_opacity(0.75)

        joint_lines_21 = create_second_layer_joint_lines(w1, b1, w2, b2, neuron_idx=1, extent=1)
        group_21_lines = m.Group(*joint_lines_21)
        group_21=m.Group(ts21, group_21_lines)

        group_21.move_to([0, 0, 2.4])
        self.add(group_21)  


        # Submerging a surface into a very smooth fully opaque liquid...
        # That's not quite right though, becuase values below the surface get clipped to 0
        # Might be interesting/cool to see shaded regions of the map move around and like come together in 
        # different ways


        # Ok great, bending looks good. And i think oreintation is actually mayby right?
        # Contour lines or heatmaps could be nice -> I think for now just a fold line would be a good starting point

        # self.add(bent_surface)
        
        # Move camera to 3D view
        # self.set_camera_orientation(phi=70*DEGREES, theta=30*DEGREES)
        
        # Animate the transformation from flat to bent
        # self.play(
        #     Transform(map_img, bent_surface),
        #     run_time=3
        # )

        self.wait(20)
        pass











