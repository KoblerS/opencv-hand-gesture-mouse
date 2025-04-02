import cv2
import mediapipe as mp
import macmouse as mouse
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam with reduced resolution for better performance

max_cameras = 10
avaiable = []
for i in range(max_cameras):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    
    if not cap.read()[0]:
        print(f"Camera index {i:02d} not found...")
        continue
    
    avaiable.append(i)
    cap.release()
    
    print(f"Camera index {i:02d} OK!")

print(f"Cameras found: {avaiable}")

webcam = cv2.VideoCapture(1)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

click_start_time = None
last_distance = None

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
  while webcam.isOpened():
    success, frame = webcam.read()
    if not success:
      break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for Mediapipe processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get the coordinates of the index finger tip and thumb tip
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

        # Calculate the midpoint between the index finger tip and thumb tip
        size = pyautogui.size()
        x = int(((index_finger_tip.x + thumb_tip.x) / 2) * size.width)
        y = int(((index_finger_tip.y + thumb_tip.y) / 2) * (size.height * 1.3))

        mouse.move(x - 50, y - 50)
        # Check if the thumb and index finger are pressed together
        distance = ((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
        if distance < 0.038:  # Adjust the threshold as needed
          if click_start_time is None:
            click_start_time = time.time()
            last_distance = distance
            mouse.press(button='left')  # Start pressing the left button
          else:
            elapsed_time = time.time() - click_start_time
            if elapsed_time > 2:  # Right-click if held for more than 2 seconds
              mouse.release(button='left')  # Release the left button before right-click
              mouse.click(button='right')
              click_start_time = None  # Reset the timer
        else:
          if click_start_time is not None:
            mouse.release(button='left')  # Release the left button when fingers are apart
            click_start_time = None  # Reset the timer

    # Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

webcam.release()
cv2.destroyAllWindows()
