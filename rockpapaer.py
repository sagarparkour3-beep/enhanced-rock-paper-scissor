import cv2
import mediapipe as mp
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Score tracking
score_p1 = 0
score_p2 = 0

# Round control
round_scored = False
last_score_time = 0
cooldown = 3  # seconds
stable_count = 0
last_result = None
prev_gestures = {}

# Gesture classification
def classify_gesture(hand_landmarks):
    fingers = []
    for tip_id in [8, 12, 16, 20]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if sum(fingers) == 0:
        return "rock"
    elif sum(fingers) == 2 and fingers[0] == 1 and fingers[1] == 1:
        return "scissors"
    elif sum(fingers) == 4:
        return "paper"
    else:
        return "unknown"

# Determine winner
def get_winner(p1, p2):
    if p1 == p2:
        return "Draw"
    elif (p1 == "rock" and p2 == "scissors") or \
         (p1 == "scissors" and p2 == "paper") or \
         (p1 == "paper" and p2 == "rock"):
        return "Player 1 Wins"
    else:
        return "Player 2 Wins"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gestures = {}
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_info.classification[0].label  # 'Left' or 'Right'
            gesture = classify_gesture(hand_landmarks)
            gestures[label] = gesture

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(img, f"{label}: {gesture}", (10, 70 if label == "Left" else 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Compare gestures and update scores
    if "Left" in gestures and "Right" in gestures:
        current_result = get_winner(gestures["Left"], gestures["Right"])
        cv2.putText(img, current_result, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # Frame-based stability
        if current_result == last_result:
            stable_count += 1
        else:
            stable_count = 0
        last_result = current_result

        # Gesture change detection
        if gestures != prev_gestures and stable_count >= 10:
            # Time-based cooldown
            if time.time() - last_score_time > cooldown:
                if current_result == "Player 1 Wins":
                    score_p1 += 1
                elif current_result == "Player 2 Wins":
                    score_p2 += 1
                last_score_time = time.time()
                round_scored = True
                prev_gestures = gestures.copy()
    else:
        round_scored = False
        stable_count = 0
        last_result = None
        prev_gestures = {}

    # Display scores
    cv2.putText(img, f"Score - P1: {score_p1} | P2: {score_p2}", (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Rock Paper Scissors - Gesture Game", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()