# -*- coding: utf-8 -*-

import time
import requests
import cv2 
import mediapipe as mp
import threading
import numpy as np
#Librerias para preprocesado de data
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
#Librerias para creacion del modelo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import math

#%%MEDIAPIPE IMPORTS

# For webcam input:
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

#%%FUNCIONES

#------------------------------------------------------------------------------

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

#------------------------------------------------------------------------------

def extract_keypoints_hand(results_hands, shoulder_center_x, shoulder_center_y, magnitud):

    """

    FUNCION PARA OBTENER PUNTOS MEDIAPIPE DE LAS MANOS (DERECHA E IZQUIERDA).

    """

        # Obtener las coordenadas de ambas manos
    rh_aux = np.zeros((2, 21, 3))
    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            for j, landmark in enumerate(hand_landmarks.landmark):
                if i > 1 or j>21:
                    print("Error i o j")
                    # rh_aux = np.zeros((2, 21, 3))
                else:
                    
                    landmark.x = (landmark.x - shoulder_center_x) / magnitud
                    landmark.y = (landmark.y - shoulder_center_y) / magnitud

                    rh_aux[i][j] = [landmark.x, landmark.y, landmark.z]
                    
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
            #                         mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            #                         mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),)
                                      
    if len(rh_aux) == 1:
        rh_aux = np.append(rh_aux, np.full((1, 21, 3), 0), axis=0)


    rh_right = rh_aux[0].reshape(-1)
    rh_left = rh_aux[1].reshape(-1)

    rh = np.concatenate([rh_right, rh_left])
    
    #print(rh)

    return rh



def extract_keypoints_pose(results_pose, shoulder_center_x, shoulder_center_y, magnitud):
    
    """
    
    FUNCION PARA OBTENER PUNTOS MEDIAPIPE DE LOS BRAZOS (DERECHO E IZQUIERDO).
    
    """
    
    #Añadir para que cuando no se detecte pose rellene con ceros o algo
    ph = []
    # Puntos de interes del pose
    right_shoulder=12
    rigth_elbow=14
    rigth_wrist=16
    left_shoulder=11
    left_elbow=12
    left_wrist=15
    
    POINTS = [right_shoulder, rigth_elbow, rigth_wrist, left_shoulder, left_elbow, left_wrist]

    # pose = np.array([[results_pose.pose_landmarks.landmark[point].x, results_pose.pose_landmarks.landmark[point].y, results_pose.pose_landmarks.landmark[point].z] for point in POINTS]).flatten() if results_pose.pose_landmarks else np.zeros(3*3)

    for point in POINTS:

        if results_pose.pose_landmarks:  
          
            results_pose.pose_landmarks.landmark[point].x = (results_pose.pose_landmarks.landmark[point].x - shoulder_center_x) / magnitud
            results_pose.pose_landmarks.landmark[point].y = (results_pose.pose_landmarks.landmark[point].y - shoulder_center_y) / magnitud


            ph = np.append(ph, np.array((results_pose.pose_landmarks.landmark[point].x, results_pose.pose_landmarks.landmark[point].y, results_pose.pose_landmarks.landmark[point].z)).flatten())
            
        else: 
            ph = np.append(ph, np.array((0, 0, 0)).flatten())

    return ph

#------------------------------------------------------------------------------
#FUNCION PARA NORMALIZAR COORDENADAS

def normalizacionCoordenadas(results_pose):
    
    xl, yl = results_pose.pose_landmarks.landmark[11].x, results_pose.pose_landmarks.landmark[11].y
    xr, yr = results_pose.pose_landmarks.landmark[12].x, results_pose.pose_landmarks.landmark[12].y
    shoulder_center_x, shoulder_center_y = (xr + yr) / 2, (xl + yl) / 2
    
    magnitud = math.sqrt((xr-xl)**2 + (yr-yl)**2)
    
    return shoulder_center_x, shoulder_center_y, magnitud

#%%MAIN
with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    etiquetas = np.array(['Luces_Encender', 'Luces_Apagar', 'Color_Naranja',
                            'Color_Rojo',  'Color_Azul',
                            'Clima', 'MusicaRock','Hora',
                            'Interruptor', 'MusicaReggae','MusicaReggaeton',
                            'Play', 'Pausa', 'Siguiente', 'Anterior', 'Temporizador_1min',
                            'Temporizador_5min', 'Roomba',
                            'Piedra', 'Papel', 'Tijera', 'Juego', 'Color_Verde', 'AbrirPuerta'])  

    #Path para cargar el modelo
    model = load_model('ModeloGestoCNNSoloManosyPoseFlip_SinNorm.h5') 
    img = cv2.imread("GasviCentrado.png",-1)# Cargar la imagen que se va a superponer
    
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_int = int(fps)
    cont_frames = 0 #Siempre a 0 para contar los frames
    
    # #Si la camara va a más de 30 fps recortamos a 30 constantes
    # if (fps_int > 30):
    #     fps = 30
    #     fps_int = 30
        
    # print("A cámara vai a: ",fps,"fps")

    num_frames=60 #Frames que queramos que dure el video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    fase = "00" # Este string lo usaremos pa controlar la fase
    
    # Definir los rectángulos deseados a la altura de los hombros
    rect_width = 150
    rect_height = 150
    rect_x_der = int(((width - rect_width) // 2)//2.9)
    rect_y_der = int(((height - rect_height) // 2.1))
    rect_x_izq = int(((width - rect_width) // 1.2))
    rect_y_izq = int(((height - rect_height) // 2.1))
    # Definir rectángulo dedo click logo
    rect_width_logo = 82
    rect_height_logo = 82
    rect_x_logo = int(((width - rect_width) // 2)+rect_width_logo//2)
    rect_y_logo = int(((height - rect_height) // 12.5))
    
    right_shoulder=12
    left_shoulder=11
  
    while cap.isOpened():
        
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        if ret:

            # Procesar el cuadro con Mediapipe para obtener las coordenadas
            frame_cambiado = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results_pose = pose.process(frame_cambiado)

            if fase == "00":
                results_pose = pose.process(frame_cambiado)
                results_hands = hands.process(frame_cambiado)
                # Definimos var aux con los 0 frames
                coords = np.zeros((1, 60, 144))
                #coords = np.zeros((1, 60, 126))
                coords_aux = []
                coords_hands_aux = []
                coords_pose_aux = []
                
                cv2.line(frame, (rect_x_der, rect_y_der), (rect_x_der+rect_width, rect_y_der), (0,255,0), (5)) # Lin derecha
                cv2.line(frame, (rect_x_der, rect_y_der), (rect_x_der, rect_y_der+rect_height), (0,255,0), (5)) # Lin derecha
                cv2.line(frame, (rect_x_izq, rect_y_izq), (rect_x_izq+rect_width, rect_y_izq), (0,255,0), (5)) # Lin izquierda
                cv2.line(frame, (rect_x_izq+rect_width, rect_y_izq), (rect_x_izq+rect_width, rect_y_izq+rect_height), (0,255,0), (5)) # Lin izquierda
                # cv2.rectangle(frame, (rect_x_logo, rect_y_logo), (rect_x_logo+rect_width_logo, rect_y_logo+rect_height_logo), (0, 255, 0), 2) # Para el logo click 
                cv2.putText(frame, "Preme en GASVI", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0, 255), 2)
                img_resized = cv2.resize(img, (frame.shape[1], frame.shape[0]))
                frame = blend_transparent(frame, img_resized)
                
                #Si encuentra mano EN EL RECTÁNGULO se habilita cuenta atrás:
                if results_pose.pose_landmarks and results_hands.multi_hand_landmarks:

                    x_right_shldr = results_pose.pose_landmarks.landmark[12].x
                    y_right_shldr = results_pose.pose_landmarks.landmark[12].y
                    x_left_shldr = results_pose.pose_landmarks.landmark[11].x
                    y_left_shldr = results_pose.pose_landmarks.landmark[11].y
                    
                    x_right_finger = results_hands.multi_hand_landmarks[0].landmark[8].x
                    y_right_finger = results_hands.multi_hand_landmarks[0].landmark[8].y
                    
                    x_r_s=int(x_right_shldr * frame.shape[1]) #Para que non estea normalizado
                    y_r_s=int(y_right_shldr * frame.shape[0])
                    x_l_s=int(x_left_shldr * frame.shape[1])
                    y_l_s=int(y_left_shldr * frame.shape[0])

                    x_r_f=int(x_right_finger * frame.shape[1]) #Normalizamos man dentro do frame
                    y_r_f=int(y_right_finger * frame.shape[0]) 
                 

                    # Comprobar si los hombros y mano estás dentro de los rectángulo
                    # if (rect_x_der < x_r_s < rect_x_der+rect_width) and (rect_y_der < y_r_s < rect_y_der+rect_height) and (rect_x_izq < x_l_s < rect_x_izq+rect_width) and (rect_y_izq < y_l_s < rect_y_izq+rect_height):
                    if (rect_x_logo < x_r_f < rect_x_logo+rect_width_logo) and (rect_y_logo < y_r_f < rect_y_logo+rect_height):
                        cont_frames = cont_frames + 1
                        if cont_frames >= 3:
                            fase = "01"    
                            cont_frames = 0
                            # print("Ámbolos dous hombreiros e dedo dentro das marcas")
                            # out = cv2.VideoWriter('GestoDetectado.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
                    else: cont_frames = 0 # Reiniciamos    

            if fase == "01":
                # print("Fase 01")
                if cont_frames>=0 and cont_frames<40//3: 
                    cv2.putText(frame, "3", (height//2, width//2), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0, 255), 10)
                if cont_frames>=40//3 and cont_frames<40//2: 
                    cv2.putText(frame, "2", (height//2, width//2), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0, 255), 10)
                if cont_frames>=40//2 and cont_frames<40: 
                    cv2.putText(frame, "1", (height//2, width//2), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0, 255), 10)
                
                cont_frames = cont_frames + 1
                if cont_frames >= 40:
                    fase = "10"
                    out = cv2.VideoWriter('GestoDetectado.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
                    cont_frames = 0
           
            if fase == "10":
                # print("Fase 10")
                out.write(frame)
                
                # print("Num frames: ", cont_frames)
                if cont_frames == 59:
                    cv2.putText(frame, "Enviando datos ao servidor...", (35,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255, 255), 3)
    
                if cont_frames >= 60:
                    print("Ata aquí a gravación")
                    cont_frames = 0
                    out.release()
                    fase = "11" # Se va a leer el vídeo
                cont_frames = cont_frames + 1
            
            if fase == "11":
                # print("Fase 11")
                cap_vid = cv2.VideoCapture("GestoDetectado.mp4")
                
                
                while cap_vid.isOpened():
                    
                    ret_vid, frame_vid = cap_vid.read()
                    frame_vid = cv2.flip(frame_vid,1)
                    if ret_vid:
                        # print("Vaise ler o vídeo")
                        frame_vid_cambiado = cv2.cvtColor(frame_vid, cv2.COLOR_RGB2BGR)
                        results_hand_vid = hands.process(frame_vid_cambiado) 
                        results_pose_vid = pose.process(frame_vid_cambiado)
                        
                        # Nueva fase pa leer el video
                        # Unimos los puntos del MediaPipe pose y hand
                        shoulder_center_x, shoulder_center_y, magnitud = normalizacionCoordenadas(results_pose)
                        coords_hands_aux = extract_keypoints_hand(results_hand_vid, shoulder_center_x, shoulder_center_y, magnitud)
                        coords_pose_aux = extract_keypoints_pose(results_pose_vid, shoulder_center_x, shoulder_center_y, magnitud)
                        
                        coords_aux = np.concatenate([coords_hands_aux, coords_pose_aux])
                        #coords_aux = coords_hands_aux
                        coords[0][cont_frames] = coords_aux
                        cont_frames = cont_frames + 1
                        
                        if cont_frames >= 60:
                            fase = "12"
                            cont_frames = 0
                            break
                    else: 
                        print("GASVI TENME DANDO VOLTERETAS")
                
                cap_vid.release()
                 
            if fase == "12":
                # print("Fase 12")
                
                # cont_frames = cont_frames + 1
                # if cont_frames >= 30:
                    # print("Enviando datos ao servidor ...")
                    chamullar = []
                    
                    ###########################Enviar a la Raspi#####################################################
                    if np.sum(coords != 0):
                        
                        res = model.predict(coords)
                        print(res) #Impprimimos resultado
                        
                        """
                        chamullar = res.tolist()
                        # print(coords.tolist())
                        # res = requests.get('http://192.168.5.144:7000/', json= coords.tolist())
                        
                        indice = chamullar.index(max(chamullar))
                        # print("eL indice MAYOR ES: ", indice)  # etiquetas[np.where(res==res.max())]
                        print("Gesto: ",etiquetas[indice])"""
                        coords=np.zeros((1, 60, 144))
                    ###########################Enviar a la Raspi#####################################################
                    
                    fase = "00"
                    cont_frames = 0
                
            cv2.imshow('GASVI MainWindow', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        else: print("Error durante a gravación")
            
    cap.release()