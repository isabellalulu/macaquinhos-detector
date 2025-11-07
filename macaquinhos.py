import math
import cv2
import mediapipe as mp
import os

# --- Carregando classificadores HaarCascade ---
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- Função para checar se apenas o dedo indicador está levantado ---
def is_index_pose(hand_landmarks):
    try:
        landmarks = hand_landmarks.landmark

        # Indicador levantado
        index_open = (
            landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
            landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        )

        # Outros dedos abaixados
        other_fingers_closed = (
            landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y >
            landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y >
            landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            landmarks[mp_hands.HandLandmark.PINKY_TIP].y >
            landmarks[mp_hands.HandLandmark.PINKY_PIP].y
        )

        return index_open and other_fingers_closed
    except Exception:
        return False

def is_thumb_pose (hand_landmarks):
    try:
        landmarks = hand_landmarks.landmark
  
        thumb_up = landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y
        
        other_fingers_closed = (
            landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y >
            landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y >
            landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            landmarks[mp_hands.HandLandmark.PINKY_TIP].y >
            landmarks[mp_hands.HandLandmark.PINKY_PIP].y and
            landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y >
            landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        )

        return thumb_up and other_fingers_closed
    except Exception:
        return False


# --- Função para medir distância entre dois pontos ---
def distancia(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Muda o diretório de trabalho para a pasta onde o script está
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Caminho atual:", os.getcwd())

# --- Carrega as imagens que serão mostradas ---
img_index = cv2.imread('macIndex.jpeg')
img_boca = cv2.imread('macBoca.jpg')
img_joia = cv2.imread('macJoia.jpg')

if img_index is None or img_boca is None or img_joia is None:
    print("Erro: verifique se as imagens 'macIndex.jpeg' e 'macBoca.jpg' estão na mesma pasta do código.")
    exit()

# Ajusta o tamanho das imagens
img_index = cv2.resize(img_index, (150, 100))
img_boca = cv2.resize(img_boca, (150, 100))
img_joia = cv2.resize(img_joia, (150, 100))

print(" Iniciando webcam... Pressione 'q' para sair.")

while True:
    ret, video = cap.read()
    if not ret:
        break

    video = cv2.flip(video, 1)
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    indicador_levantado = False
    mao_na_boca = False
    joia = False

    # --- Detecta o rosto ---
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    boca_pos = None  # posição estimada da boca

    for (x, y, w, h) in faces:
        # Marca o rosto
        cv2.rectangle(video, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # --- Estima posição da boca ---
        boca_x = x + w // 2
        boca_y = y + int(h * 0.75)
        boca_pos = (boca_x, boca_y)
        cv2.circle(video, boca_pos, 5, (0, 0, 255), -1)  # desenha ponto da boca

    # --- Detecta a mão e gestos ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(video, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordenadas do indicador
            h, w, _ = video.shape
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))

            if is_index_pose(hand_landmarks):
                indicador_levantado = True

            # Se há boca detectada, mede distância entre indicador e boca
            if boca_pos:
                dist = distancia(index_pos, boca_pos)
                if dist < 50:  # distância pequena = mão na boca
                    mao_na_boca = True

            #Joia 
            if is_thumb_pose (hand_landmarks):
                joia = True

    # --- Mostra imagem dependendo da ação ---
    img_h, img_w, _ = img_index.shape
    x_offset = video.shape[1] - img_w - 50  # margem da direita
    y_offset = video.shape[0] - img_h - 50  # margem de baixo

    if mao_na_boca:
        video[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = img_boca
    elif indicador_levantado:
        video[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = img_index
    elif  joia:
        video[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = img_joia

    # --- Mostra resultado ---
    cv2.imshow('Reconhecimento', video)

    # --- Encerra com 'q' ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
