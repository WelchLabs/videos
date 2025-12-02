from manimlib import *
from functools import partial

from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors

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

def viridis_hex(value, vmin, vmax):
    """
    Map a scalar `value` in [vmin, vmax] to a Viridis color (hex string).
    """
    # Normalize into [0,1]
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    # Get RGBA from viridis
    rgba = cm.viridis(norm(value))
    # Convert to hex
    return colors.to_hex(rgba)

def black_to_tan_hex(value, vmin, vmax=1):
    """
    Map a scalar `value` in [vmin, vmax] to a color from black to FRESH_TAN (#dfd0b9).
    """
    cmap = colors.LinearSegmentedColormap.from_list('black_tan', ['#000000', '#dfd0b9'])
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return colors.to_hex(cmap(norm(value)))

def softmax_with_temperature(logits, temperature=1.0, axis=-1):
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=axis, keepdims=True))
    return exp_scaled / np.sum(exp_scaled, axis=axis, keepdims=True)

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/grokking/graphics/to_manim')
data_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/grokking/from_linux/grok_1764602090')

class GrokkingHackingTwo(InteractiveScene):
    def construct(self):  


        svg_files=list(sorted(svg_dir.glob('*network_to_manim*')))

        with open(data_dir/'final_model_activations_sample.p', 'rb') as f:
            activations = pickle.load(f)

        all_svgs=Group()
        for svg_file in svg_files[1:16]: #Expand if I add more artboards
            svg_image=SVGMobject(str(svg_file))
            all_svgs.add(svg_image[1:]) #Thowout background

        all_svgs.scale(6.0) #Eh?

        self.add(all_svgs)

        ## Ok ok ok ok important question here -> can I just show activations
        ## by modifying these suckers?
        # self.remove(all_svgs[3][0])
        # self.add(all_svgs[3])
        # all_svgs[3][6].set_color(YELLOW) 

        # Ok if we do viridis, which I do think we try first, then I think I color the fill and 
        # borders. Indexing is going to be annoying with using the stuff from illustrator directly
        # but I don't think it will be that bad
        # Ok i kinda want to bring in real activations now and start coloring viridis instead of using dummies, let's do it. 

        

        #Borders and fills
        input_mapping_1a=[[0, 1], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]
        input_mapping_1b=[[19, 20], [21, 22]]
        input_mapping_2a=[[23, 24], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41]]
        input_mapping_2b=[[42, 43], [44, 45]]
        input_mapping_3a=[[46, 47], [51, 52], [53, 54], [55, 56], [57, 58], [59, 60], [61, 62], [63, 64]]
        input_mapping_3b=[[65, 66], [67, 68]]

        example_index=117

        #Color inputs
        for mapping, activations_index, offset in zip([input_mapping_1a, input_mapping_1b, input_mapping_2a, input_mapping_2b, input_mapping_3a, input_mapping_3b], 
                                              [0, 0, 1, 1, 2, 2], [0, 112, 0, 112, 0, 112]):
            for i, idx in enumerate(mapping):
                if i+offset == activations['x'][example_index][activations_index]:
                    all_svgs[2][idx[0]].set_color(FRESH_TAN)
                else:
                    all_svgs[2][idx[0]].set_color(BLACK)


        # 50/50 on viridis vs tan here, easy to change. 
        # Swaggin here a bit, these are upside down and not skipping correctly, I think
        # Easy to adjust later if needed for consistency

        embedding_fill_indices_1=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        embedding_fill_indices_2=[28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
        embedding_fill_indices_3=[53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73]
        for i, indices in enumerate([embedding_fill_indices_1, embedding_fill_indices_2, embedding_fill_indices_3]):
            vmin=np.min(activations['blocks.0.hook_resid_pre'][example_index][i])*1.5 #Scaling by column
            vmax=np.max(activations['blocks.0.hook_resid_pre'][example_index][i])*1.5
            for j, idx in enumerate(indices):
                # c=viridis_hex(activations['blocks.0.hook_resid_pre'][example_index, 0, i], vmin, vmax)
                c=black_to_tan_hex(activations['blocks.0.hook_resid_pre'][example_index, i, j], vmin, vmax)
                # print(activations['blocks.0.hook_resid_pre'][example_index, 0, j])
                all_svgs[4][idx].set_color(c)


        # self.wait()


        ## Attention - Values - mapping again is hacky here, technicall first layer should be same for all
        ## Keep moving for now.
        vmin=np.min(activations['blocks.0.attn.hook_v'][example_index])*0.25 #Scaling
        vmax=np.max(activations['blocks.0.attn.hook_v'][example_index])*0.25
        value_fill_indices_1=[0, 2, 4, 6, 8]
        value_fill_indices_2=[10, 12, 14, 16, 18]
        value_fill_indices_3=[20, 22, 24, 26, 28]
        value_fill_indices_4=[30, 32, 34, 36, 38]

        for head_id, indices in enumerate([value_fill_indices_1, value_fill_indices_2, value_fill_indices_3, value_fill_indices_4]):
            for j, idx in enumerate(indices):
                c=black_to_tan_hex(activations['blocks.0.attn.hook_v'][example_index, head_id, 1, j], vmin, vmax)
                all_svgs[12][idx].set_color(c)


        #Attention - attention patterns - need to be more precise with these!
        vmin=0
        vmax=0.8
        attn_fill_indices=[[0,0], [1,0], [1,1], [2,0], [2,1], [2,2]] #Indices to sample matrix at
        head_id=0
        for head_id, offset in enumerate([0, 6, 12, 18]):
            for j, idx in enumerate(attn_fill_indices):
                a=activations['blocks.0.attn.hook_attn'][example_index, head_id, idx[0], idx[1]]
                c=black_to_tan_hex(a, vmin, vmax)
                all_svgs[13][offset+j].set_color(c)

        self.remove(all_svgs[7]); self.add(all_svgs[7]) #Stacking for grid


        #MLP layer 1 (565, 3, 128) 
        vmin=np.min(activations['blocks.0.hook_resid_mid'][example_index])*0.85 #Scaling
        vmax=np.max(activations['blocks.0.hook_resid_mid'][example_index])*0.85
        mlp_indices_1=[0, 2, 4, 6, 8, 10, 12]
        for i, idx in enumerate(mlp_indices_1):
            c=black_to_tan_hex(activations['blocks.0.hook_resid_mid'][example_index, 2, i], vmin, vmax)
            all_svgs[9][idx].set_color(c)


        #MLP Layer 2 (565, 3, 512)
        vmin=np.min(activations['blocks.0.mlp.hook_pre'][example_index])*0.85 #Scaling
        vmax=np.max(activations['blocks.0.mlp.hook_pre'][example_index])*0.85
        mlp_indices_2=[14, 16, 18, 20, 22, 24, 26, 28, 30]
        for i, idx in enumerate(mlp_indices_2):
            c=black_to_tan_hex(activations['blocks.0.mlp.hook_pre'][example_index, 2, i], vmin, vmax)
            all_svgs[9][idx].set_color(c)

        # MLP Layer 3 (565, 3, 128) 
        vmin=np.min(activations['blocks.0.hook_mlp_out'][example_index])*0.85 #Scaling
        vmax=np.max(activations['blocks.0.hook_mlp_out'][example_index])*0.85
        mlp_indices_3=[32, 34, 36, 38, 40, 42, 44]
        for i, idx in enumerate(mlp_indices_3):
            c=black_to_tan_hex(activations['blocks.0.hook_mlp_out'][example_index, 2, i], vmin, vmax)
            all_svgs[9][idx].set_color(c)

        #Add back borders for occlusions
        # for i in range(1, 47, 2):
        #     self.remove(all_svgs[9][i]); self.add(all_svgs[9][i])


        #Logits or probs (565, 113)
        logit_indices_1=[3, 5, 7, 9, 11, 13, 15, 17]
        logit_indices_2=[19, 21]

        #Don't love probs or logits, how about temperature?
        probs_sortof=softmax_with_temperature(activations['logits'][example_index], temperature=25.0, axis=0)


        vmin=np.min(probs_sortof)*1.0 #Scaling
        vmax=np.max(probs_sortof)*1.0

        for i, idx in enumerate(logit_indices_1):
            c=black_to_tan_hex(probs_sortof[i], vmin, vmax)
            all_svgs[11][idx].set_color(c)

        for i, idx in enumerate(logit_indices_2):
            c=black_to_tan_hex(probs_sortof[i+111], vmin, vmax)
            all_svgs[11][idx].set_color(c)


        #MLP weights
        np.random.seed(5)
        R=np.random.uniform(0.3, 0.75, len(all_svgs[8]))
        for i in range(len(all_svgs[8])):
            all_svgs[8][i].set_opacity(R[i])
        # all_svgs[8][3].set_color(YELLOW)

        # self.remove(all_svgs[8])


        self.wait()

        #Ok 50/50 still on activations in tan vs viridis - easy to change as I actually get into paragraph 25
        #Alright building blocks are here
        # Tempted to wrap stuff up now, but probably makes sense to wait until I get more into the script!




        









        # all_svgs[9][44].set_color(YELLOW)


        # activations['blocks.0.attn.hook_v'][example_index, head_id, i, j]







        # all_svgs[4][57].set_color(YELLOW)


        # all_svgs[2][0].set_color(BLACK)

            # c=viridis_hex(activations['x'][i][0], 0, 1)
            # for j in idx:
            #     all_svgs[2][j].set_color(c)




        #Off inputs set to purple? Let's see how it feels
        #hmm ok need to hack some more here -> also 50/50 of purple or not, and how we handle borders
        #I have complete control here, just need to experiment a bit and decide. 


        # c= viridis_hex(, 0, 1)


        # all_svgs[2][28].set_color(YELLOW)

        # for a in input_mapping_3b: 
        #     for i in a:
        #         all_svgs[2][i].set_color(YELLOW)

        # self.wait()


        # for i in input_mapping[0]: all_svgs[2][i].set_color(RED)



        # activations['x'][example_index]






        self.wait()



        self.wait(20)
        self.embed()