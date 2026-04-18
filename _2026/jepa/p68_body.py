import cv2

VIDEO_PATH = '/Volumes/PG Work/Stephencwelch Dropbox/welch_labs/jepa/hacking/scratch_3.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
s = min(w, h)
cropped = frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
display = cv2.resize(cropped, (448, 448))

def onclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'QUERY_X = {x}\nQUERY_Y = {y}')

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', onclick)
print('Click your shoulder. Press q to quit.')
while True:
    cv2.imshow('frame', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
