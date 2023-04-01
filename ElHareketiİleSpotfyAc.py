import cv2
import mediapipe as mp
import webbrowser

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Kamerayı aç
cap = cv2.VideoCapture(0)

# El tespiti için Mediapipe Hands modelini yükle
with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
    
    # Pinch hareketinin başlatıp başlatılmadığını takip etmek için bir değişken tanımla
    pinch_started = False
    
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
                # İşaret parmağı ve baş parmağı noktalarını al
                index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                # İşaret parmağı ve baş parmağı noktalarının konumunu piksel cinsinden hesapla
                x1, y1 = int(index_finger_landmark.x * width), int(index_finger_landmark.y * height)
                x2, y2 = int(thumb_landmark.x * width), int(thumb_landmark.y * height)
                
                # İşaret parmağı ve baş parmağı arasındaki mesafeyi hesapla
                distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                
                # Eğer mesafe 50 pikselden az ise pinch hareketi başlamış demektir
                if distance < 50:
                    pinch_started = True
                else:
                    pinch_started = False
                
                # Eğer pinch hareketi başladıysa
                if pinch_started:
                    # Spotify uygulamasını aç
                    webbrowser.open('spotify:')
                    # Pinch hareketinin bittiğinden emin olmak için ekranda bir mesaj göster
                    cv2.putText(image, "Pinch hareketi algılandı. Spotify açıldı.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    break
        
        # Görüntüyü göster
        cv2.imshow('Mediapipe Hands', image)
        
        # ESC tuşuna basılınca çık
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()