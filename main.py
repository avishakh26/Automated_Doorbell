import cv2
import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load("doorbell.wav")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

ring_cooldown = 5
last_ring_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        current_time = time.time()
        if current_time - last_ring_time > ring_cooldown:
            print("Someone is at the door...")
            pygame.mixer.music.play()
            last_ring_time = current_time

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face detected automated Doorbell", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


