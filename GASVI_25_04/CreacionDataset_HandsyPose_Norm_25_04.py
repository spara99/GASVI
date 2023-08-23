import cv2
import mediapipe as mp
#import pandas as pd
import os
import numpy as np
import math 
#Librerias para preprocesado de data
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
#Librerias para creacion del modelo
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import tensorflow as tf

# Configuración de Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

#%%FUNCIONES

#------------------------------------------------------------------------------
#FUNCION PARA SACAR DATA DE LA MANO 

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

    return rh

#------------------------------------------------------------------------------
#FUNCION PARA SACAR DATA DEL POSE

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
            
            
    # if results_pose.pose_landmarks:        
    #     zeta_r = results_pose.pose_landmarks.landmark[16].z
    #     zeta_l = results_pose.pose_landmarks.landmark[15].z
    # else:
    #     zeta_r=0
    #     zeta_l=0
        
    #print("Os datos do zeta: ",zeta_r," y izq ",zeta_l)
    
        
    return ph


#------------------------------------------------------------------------------
#FUNCION PARA NORMALIZAR COORDENADAS

def normalizacionCoordenadas(results_pose):
    
    xl, yl = results_pose.pose_landmarks.landmark[11].x, results_pose.pose_landmarks.landmark[11].y
    xr, yr = results_pose.pose_landmarks.landmark[12].x, results_pose.pose_landmarks.landmark[12].y
    shoulder_center_x, shoulder_center_y = (xr + yr) / 2, (xl + yl) / 2
    
    magnitud = math.sqrt((xr-xl)**2 + (yr-yl)**2)
    
    return shoulder_center_x, shoulder_center_y, magnitud
      
#------------------------------------------------------------------------------
#FUNCION PARA RECORTAR VIDEOS

def recorte_videos(actions, video):
    
    #Numero de frames a recortar
    # framesRecortar = 60
    
    # Abre el archivo de video
    cap = cv2.VideoCapture("C:/Users/Usuario/OneDrive/Documents/4oUNI/TFG/Media/Code/Lpro/SoloManosFlip/"+actions+"/"+video)

    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = 60
    
    return cap, length


   
#%%MAIN

if __name__=='__main__':
    
    #Aqui ponemos las etiquetas de los videos con los que vamos a entrenar el modelo
    
    etiquetas = np.array(['Luces_Encender', 'Luces_Apagar', 'Color_Naranja',
                            'Color_Rojo',  'Color_Azul',
                            'Clima', 'MusicaRock','Hora',
                            'Interruptor', 'MusicaReggae','MusicaReggaeton',
                            'Play', 'Pausa', 'Siguiente', 'Anterior', 'Temporizador_1min',
                            'Temporizador_5min', 'Roomba',
                            'Piedra', 'Papel', 'Tijera', 'Juego', 'Color_Verde', 'AbrirPuerta'])  

           
    
    #Abrimos procesos de MediaPipe
    with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
    
        #For para recorrer carpetas con etiquetas (ManoAbierta, ManoCerrada, etc)
        for actions in etiquetas:
            
            # Lista de videos de train en carpeta especifica
            video_list=os.listdir("C:/Users/Usuario/OneDrive/Documents/4oUNI/TFG/Media/Code/Lpro/SoloManosFlip/"+actions)
            
            #Iterar a través de cada video dentro de esas carpetas
            for video in range(len(video_list)):
                
                #Creamos las carpetas con las etiquetas de cada video para almacenar los .npy
                try: os.makedirs(os.path.join("C:/Users/Usuario/OneDrive/Documents/4oUNI/TFG/Media/Code/Lpro/SoloManosFlip/npys_hands_y_pose_sinNorm/", actions, str(video)))
                except: pass
                print("--------------------------------------------------------------------------------------------------------")
                #Abrimos video actual y llamamos funcion para recortarlo en tiempo y FPS
                cap, length = recorte_videos(actions, video_list[video])
                
                #Sacar factor de escala para normalizacion
                # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # horizontal_fov = math.radians(60)
                # f = (width/2) / math.tan(horizontal_fov/2)
                # distancia_camara = 1
                # SCALE_FACTOR = 1/f * distancia_camara

                print(str(video_list[video]) + "  procesandose ...")
            
                #Iterar a través de cada cuadro (Frames)
                for i in range(length):
                    
                    ret, frame = cap.read()
                    frame=cv2.flip(frame,1)
                    hand_coords = []
                    pose_coords = []
        
                    if ret:
                        # Procesar el cuadro con Mediapipe para obtener las coordenadas
                        frame_camb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        results_hands = hands.process(frame_camb)
                        results_pose = pose.process(frame_camb)
                        
                        #Llamamos funciones para sacar puntos de hand y pose desde MediaPipe
                        shoulder_center_x, shoulder_center_y, magnitud = normalizacionCoordenadas(results_pose)
                        pose_coords = extract_keypoints_pose(results_pose, shoulder_center_x, shoulder_center_y, magnitud)
                        hand_coords = extract_keypoints_hand(results_hands, shoulder_center_x, shoulder_center_y, magnitud)
                        # print("Hand cords: ", hand_coords)
                        #Sincroniamos punto z de hand y pose
                        # hand_coords = posicionZetaAbsoluta(hand_coords, zeta_r, zeta_l)
                        
                        #Unimos los puntos del MediaPipe pose y hand
                        #(AQUI SE DEBE SINCRONIZAR PUNTO DE MUÑECA DE POSE Y HAND ANTES DE CONCATENAR)
                        coords = np.concatenate([hand_coords, pose_coords])
                        #coords = hand_coords #Si sudamos del pose
                        #Se guardan en las carpetas anteriormente creadas los .npy con las coordenadas xyz de cada frame de cada video de cada etiqueta
                        npy_path = os.path.join("C:/Users/Usuario/OneDrive/Documents/4oUNI/TFG/Media/Code/Lpro/SoloManosFlip/npys_hands_y_pose_sinNorm/", actions, str(video), str(i))
                        np.save(npy_path,coords)
                        
    #--------------------- AQUI EMPIEZA CONSTRUCCION MODELO--------------------
                        
    label_map = {label:num for num, label in enumerate(etiquetas)}
    print(label_map)
    
    sequences, labels = [], []
    #For para recorrer carpetas con etiquetas (ManoAbierta, ManoCerrada, etc)
    for actions in etiquetas:
        
        # Lista de videos de train en carpeta especifica
        video_list=os.listdir("C:/Users/Usuario/OneDrive/Documents/4oUNI/TFG/Media/Code/Lpro/SoloManosFlip/"+actions)
            
        #Iterar a través de cada video dentro de esas carpetas
        for video in range(len(video_list)):
            window = []
                
            #Iterar a través de cada cuadro (Frames)
            for i in range(length):
                    
                res = np.load(os.path.join("C:/Users/Usuario/OneDrive/Documents/4oUNI/TFG/Media/Code/Lpro/SoloManosFlip/npys_hands_y_pose_sinNorm/", actions, str(video), "{}.npy".format(i)))
                window.append(res)
                
            sequences.append(window)
            labels.append(label_map[actions])
        
    #Ahora sacamos las X y la Y para entrenar el modelo:
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    #Dividimos data en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=(42))
    
  #------------------------MODELO CNN------------------------------------------
  
    num_classes = 24
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=X_train.shape[1:3]),
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    # tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="causal", activation="relu"),
    # tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="causal", activation="relu"),
    # tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(rate=0.2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(num_classes, activation='softmax')])
       
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
       
    #Train the Model
    model.fit(X_train, y_train, epochs=175, batch_size=142, validation_data=(X_test, y_test))
       
    model.save('C:/Users/Usuario/OneDrive/Documents/4oUNI/TFG/Media/Code/Lpro/SoloManosFlip/ModeloGestoCNNSoloManosyPoseFlip_SinNorm.h5') #Esto salva el modelo en un doc HDF5
    
    #---------------------------MATRIZ DE CONFUSION----------------------------
      
    yhat = model.predict(X_test)
      
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
      
    print(multilabel_confusion_matrix(ytrue, yhat))
      
    print(accuracy_score(ytrue, yhat))
    
    cap.release()                    

    
