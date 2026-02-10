# SHIFT+D | SHIFT D
# Copying code to recreate state
def copy_frame_positioning_precise(frame):
    center = frame.get_center()
    height = frame.get_height()
    angles = frame.get_euler_angles()

    call = f"reorient("
    theta, phi, gamma = (angles / DEG)
    call += f"{theta}, {phi}, {gamma}"
    if any(center != 0):
        call += f", {tuple(center)}"
    if height != FRAME_HEIGHT:
        call += ", {:.2f}".format(height)
    call += ")"
    print(call)
    pyperclip.copy(call)