import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Kamerayı aç
cap = cv2.VideoCapture(0)

# El tespiti için Mediapipe Hands modelini yükle
with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
    
    # Hareketi kaydetmek için bir değişken oluştur
    move_x = 0
    move_y = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kamera açılamadı")
            break
        
        # Görüntüyü BGR den RGB ye dönüştür
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Yükseklik ve genişlik değerlerini al
        height, width, _ = image.shape
        # El tespiti yap
        results = hands.process(image)
        
        # Eğer el tespit edildiyse
        if results.multi_hand_landmarks:
            # Her bir el için
            for hand_landmarks in results.multi_hand_landmarks:
                # İşaret parmağı noktasını al
                index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                # İşaret parmağı noktasının konumunu piksel cinsinden hesapla
                x, y = int(index_finger_landmark.x * width), int(index_finger_landmark.y * height)
                # Yeşil bir daire çiz
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
                
                # Mouse hareketini hesapla
                move_x = (x - (width // 2)) // 10
                move_y = ((height // 2) - y) // 10
                
                # Mouse hareketini uygula
                pyautogui.moveRel(move_x, move_y, duration=0.1)
                
        # Görüntüyü göster
        cv2.imshow('Mediapipe Hands', image)
        
        # ESC tuşuna basılınca çık
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Hareketi geri al
    pyautogui.moveRel(-move_x, -move_y, duration=0.1)

cap.release()
cv2.destroyAllWindows()
