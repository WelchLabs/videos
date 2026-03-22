from manimlib import *


IMG_DIR = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/vla/hackin/mar_17_2'
CHILL_BROWN = '#948979'


class P50(InteractiveScene):
    def construct(self):

        img_height = 4.0

        all_images = []
        for idx in range(100):
            path = f'{IMG_DIR}/step_{idx:03d}.png'
            img = ImageMobject(path)
            img.set_height(img_height)
            all_images.append(img)

        self.add(all_images[0])
        self.wait()

        label = Text("REALISTIC_VISION_V5.1", font_size=24).set_color(CHILL_BROWN)
        label.move_to(all_images[0].get_center() + DOWN * (img_height / 2 + 0.3))
        label.align_to(all_images[0], RIGHT)

        self.add(label)

        index_label = Text("1", font_size=24).set_color(CHILL_BROWN)
        index_label.move_to(all_images[0].get_center() + DOWN * (img_height / 2 + 0.3))
        index_label.align_to(all_images[0], LEFT)

        self.add(index_label)

        for i in range(1, 100):
            self.remove(all_images[i - 1])
            self.add(all_images[i])
            new_index_label = Text(str(i + 1), font_size=24).set_color(CHILL_BROWN)
            new_index_label.move_to(all_images[0].get_center() + DOWN * (img_height / 2 + 0.3))
            new_index_label.align_to(all_images[0], LEFT)
            self.remove(index_label)
            index_label = new_index_label
            self.add(index_label)
            self.wait(1/15)

        self.wait()

        expand_indices = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        expand_images = []
        for idx in expand_indices:
            path = f'{IMG_DIR}/step_{idx:03d}.png'
            img = ImageMobject(path)
            img.set_height(img_height)
            img.move_to(all_images[99].get_center())
            expand_images.append(img)

        # Target arrangement
        target_height = 1.8
        buff = 0.1

        targets = Group()
        for img in expand_images:
            t = img.copy()
            t.set_height(target_height)
            targets.add(t)
        targets.arrange(RIGHT, buff=buff)
        targets.move_to(ORIGIN)

        step_labels = []
        start_label_offset = DOWN * (img_height / 2 + 0.3)
        target_label_offset = DOWN * (target_height / 2 + 0.3 * (target_height / img_height))
        for i, idx in enumerate(expand_indices):
            sl = Text(str(idx + 1), font_size=16).set_color(CHILL_BROWN)
            sl.move_to(all_images[99].get_center() + start_label_offset)
            sl.align_to(all_images[99], LEFT)
            step_labels.append(sl)

        target_label_positions = []
        for i, idx in enumerate(expand_indices):
            tl = Text(str(idx + 1), font_size=16).set_color(CHILL_BROWN)
            tl.move_to(targets[i].get_center() + target_label_offset)
            tl.align_to(targets[i], LEFT)
            target_label_positions.append(tl)

        self.remove(all_images[99], label, index_label)
        for img in expand_images:
            self.add(img)
        for sl in step_labels:
            self.add(sl)

        strip_width = targets.get_width() + 1.0
        strip_height = targets.get_height() + 1.5
        frame = self.camera.frame
        target_frame_width = max(strip_width, strip_height * frame.get_width() / frame.get_height())

        self.play(
            *[
                img.animate.match_height(targets[i]).move_to(targets[i].get_center())
                for i, img in enumerate(expand_images)
            ],
            *[
                sl.animate.move_to(target_label_positions[i].get_center())
                for i, sl in enumerate(step_labels)
            ],
            frame.animate.set_width(target_frame_width).move_to(targets.get_center()),
            run_time=3,
        )
        self.wait()

        self.embed()
 
