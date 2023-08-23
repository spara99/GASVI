# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from flask import Flask
import multiprocessing

app = Flask(__name__)

@app.route("/fin_1min") #http://localhots:7000/fin_5min
def fin_1min():
    pipe_hijo.send(1)
	#PopUpWindow(popup='temporizador1')
    return "OK"

@app.route("/fin_5min") #http://localhots:7000/fin_5min
def fin_5min():
    pipe_hijo.send(5)
	#PopUpWindow(popup='temporizador5')
    return "OK"

def lanzar_server(pipe):
    global pipe_hijo
    pipe_hijo = pipe
    app.run(host='0.0.0.0', port=7000)
    
if __name__ == '__main__':
    
    multiprocessing.freeze_support()
    padre_pipe, hijo_pipe = multiprocessing.Pipe()
    p = multiprocessing.Process(target=lanzar_server, args=(hijo_pipe,))
    p.start()
    pipe_padre = padre_pipe
    
    # Import kivy dependencies 
    from kivymd.app import MDApp
    from kivy.core.window import Window
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.scatterlayout import ScatterLayout
    from kivy.uix.relativelayout import RelativeLayout
    from kivy.lang import Builder
    from kivy.uix.image import Image
    from kivy.uix.button import Button
    from kivy.properties import ObjectProperty
    from kivy.clock import Clock
    from kivy.graphics.texture import Texture
    from kivy.uix.screenmanager import ScreenManager
    from kivymd.uix.screen import MDScreen
    from kivy.uix.label import Label
    from kivy.uix.popup import Popup
    from kivy.uix.screenmanager import RiseInTransition, SlideTransition, FallOutTransition
    from kivy.uix.scrollview import ScrollView
    from kivy.graphics import RoundedRectangle, Color, Rectangle
    from kivy.animation import Animation
    import json

    # Hand recognition imports
    import time
    import requests
    import cv2 
    import mediapipe as mp
    import numpy as np
    import math

    #%%MEDIAPIPE IMPORTS

    # For webcam input:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    #%%FUNCIONES

    class PopUpWindow(Popup):
        def __init__(self, popup, **kwargs):
            super(PopUpWindow, self).__init__(**kwargs)
            self.size_hint = (None, None)
            self.title = " "
            self.separator_height = 0
            self.size = (450, 250)
            self.auto_dismiss = False
            Clock.schedule_once(self.dismiss_popup, 5.0)
            
            self.box = BoxLayout()
            self.box_image = BoxLayout(size_hint = (1,1))
            self.box_label = BoxLayout()
            print(popup)
            myimage = None
            mylabel = None
            if(popup == 'ganador_user'):
                self.title = ' '
                self.background = 'Imagenes/ganador.png'
            if(popup == 'ganador_gasvi'):
                self.title = ' '
                self.background = 'Imagenes/segundo.png'
            if(popup == 'temporizador1'):
                myimage = 'Imagenes/temporizador1.png'
                mylabel = '[size=15]Fin temporizador 1 min[/size]'
            if(popup == 'temporizador5'):
                myimage = 'Imagenes/temporizador5.png'
                mylabel = '[size=15]Fin temporizador 5 min[/size]'
            
            if(popup == 'temporizador5' or popup == 'temporizador1'):
                self.box_image.add_widget(Image(source = myimage, allow_stretch = True, width = 250, height = 250))
                self.box_label.add_widget(Label(text = mylabel, markup=True))
                self.box.add_widget(self.box_image)
                self.box.add_widget(self.box_label)
                self.add_widget(self.box)
    
        def dismiss_popup(self, dt):
            self.dismiss()
            
    class RoundedPanel(ScatterLayout):
        """Clase que define una Label con las esquinas redondeadas"""
    
        def __init__(self, data, **kwargs):
            super().__init__(**kwargs)
            #self.box = BoxLayout(size_hint=(1,1))
            self.background_color = (0, 0, 0, 0)  # Hacemos transparente el fondo de la Label
            #self.color = (196/255, 166/255, 254/255, 1)
            self.color = (88/255, 88/255, 88/255, 1)
            self.centro_fijo = self.center_x
            self.data = data
            self.size_hint=(0.85, None)
            self.do_translation = False
            
            with self.canvas.before:
                radius = min(self.height, self.width) / 4
                c = Color(*self.color)
                self.background = RoundedRectangle(size=self.size, color=c, pos_y=self.y, radius=[radius]*4)
    
            self.bind(size=self.update_background)
            self.y += 20
    
            self.process_json(self)
             
        def show_popup(self, popup):
                objpopup = PopUpWindow(popup)
                objpopup.open()
        
        def update_background(self, *args):
            """Actualiza el dibujo del fondo de la ScatterLayout"""
            self.background.size = self.size
            self.center_x = self.centro_fijo
    
        def process_json(self, popup, *args):
            self.box = BoxLayout()
            self.box_image = BoxLayout(size_hint = (0.3,1))
            self.box_label = BoxLayout()
            
            mylabel, myimage, popup = process_json(self.data)
            self.box_image.add_widget(Image(source = myimage, allow_stretch = True))
            self.box_label.add_widget(Label(text = mylabel, markup=True))
            self.box.add_widget(self.box_image)
            self.box.add_widget(self.box_label)
            self.add_widget(self.box)
    
            if(popup != None):
                self.show_popup(popup)
      
    class Pantalla(BoxLayout):
        """Clase que define la pantalla"""
        pass

    class GasviApp(MDApp):

        connection = None
        
        def __init__(self,  pipe_padre, **kwargs):
            super().__init__(**kwargs)     
            self.panels = []  # Lista que almacena los paneles añadidos
            self.max_panels = 100  # Número máximo de paneles a mostrar 
            
            # self.layout = BoxLayout(size_hint=(1, 1))
            self.scroll_layout = RelativeLayout(size_hint=(1,1))
            self.pipe_padre = pipe_padre
            Clock.schedule_interval(self.verificar_temporizadores, 10)
    
        def add_panel(self, *args):
            """Añade un nuevo panel a la parte derecha de la pantalla"""
            if len(self.panels) >= self.max_panels:
                return
        
            # Creamos un nuevo panel con un número consecutivo
            #micolor = random.choice(lista_colores)
            #data = {'funcion':'hora', 'hour': '1', 'min': '2','sec': '3','day': '4','month': '5','year': '2006','cancion': 'tamo en nota'}
            #data = {'funcion':'error_identificacion', 'error': False}
            
            #data = {'funcion': 'Piedra', 'partida_creada': True, 'ganador': None, 'eleccion_gasvi': 'Papel', 'puntos_user': '1', 'puntos_gasvi': '3'}data=json.dumps(self.data)
            panel = RoundedPanel(height=self.scroll_layout.height/7, data = self.data, width=self.scroll_layout.width)
        
            self.scroll_layout.add_widget(panel)
            self.panels.append(panel)
        
            # Animamos los paneles existentes para que se desplacen hacia arriba
            for i, panel in enumerate(self.panels[:-1]):
                animation = Animation(y= panel.y + panel.height + 20, duration=0.5)
                animation.start(panel)
           
        def verificar_temporizadores(self, *args):
            if(self.pipe_padre.poll()):
                data = self.pipe_padre.recv()
                if(data == 1):
                    objpopup = PopUpWindow(popup='temporizador1')
                    objpopup.open()
                if(data == 5):
                    objpopup = PopUpWindow(popup='temporizador5')
                    objpopup.open()
        
        def build(self):
            Window.size = (1300,800)
            Window.set_title('Gasvi')
            Window.set_icon("logo_gasvi.png")
            
            self.src = ["Gifs/Abrir_Puerta.gif", 
                        "Gifs/Anterior.gif", 
                        "Gifs/Apagar_Luces.gif", 
                        "Gifs/Clima.gif", 
                        "Gifs/Color_Azul.gif",
                        "Gifs/Color_Naranja.gif", 
                        "Gifs/Color_Rojo.gif", 
                        "Gifs/Color_Verde.gif", 
                        "Gifs/Encender_Luces.gif", 
                        "Gifs/Hora.gif", 
                        "Gifs/Interruptor.gif", 
                        "Gifs/Juego.gif", 
                        "Gifs/Musica_Reggaeton.gif", 
                        "Gifs/Musica_Rock.gif", 
                        "Gifs/Papel.gif", 
                        "Gifs/Pausa.gif", 
                        "Gifs/Piedra.gif", 
                        "Gifs/Play.gif", 
                        "Gifs/Roomba.gif", 
                        "Gifs/Siguiente.gif", 
                        "Gifs/Temporizador_1min.gif", 
                        "Gifs/Temporizador_5min.gif", 
                        "Gifs/Tijeras.gif"]
            
            
            #          ["Abrir_Puerta.gif", 
            #             "Anterior.gif", 
            #             "Apagar_Luces.gif", 
            #             "Clima.gif", 
            #             "Color_Azul.gif",
            #             "Color_Naranja.gif", 
            #             "Color_Rojo.gif", 
            #             "Color_Verde.gif", 
            #             "Encender_Luces.gif", 
            #             "Hora.gif", 
            #             "Interruptor.gif", 
            #             "Juego.gif", 
            #             "Musica_Reggaeton.gif", 
            #             "Musica_Rock.gif", 
            #             "Papel.gif", 
            #             "Pausa.gif", 
            #             "Piedra.gif", 
            #             "Play.gif", 
            #             "Roomba.gif", 
            #             "Siguiente.gif", 
            #             "Temporizador_1min.gif", 
            #             "Temporizador_5min.gif", 
            #             "Tijera.gif"]
            
            self.VB = BoxLayout(orientation ='vertical')
            self.VB_layout = ScrollView(size_hint = (1, 1), do_scroll_x = True, do_scroll_y = False, scroll_type = ['bars', 'content'], bar_width = 8, bar_inactive_color =  [0, 0, 0, 1], bar_color = [0, 0, 0, 1])
            # layout = GridLayout(cols = 23, size_hint_x = None, col_default_width = self.VB_layout.height*3, col_force_default = True)
            layout = GridLayout(cols = 23, size_hint_x = None, col_default_width = self.VB.height, col_force_default = True)
            layout.bind(minimum_width = layout.setter('width'))
            
            for i in range(23):
                image = Image(source = self.src[i], allow_stretch = True, anim_delay = 0.06, width = self.VB.height)
                layout.add_widget(image)
            self.VB.clear_widgets()
            self.VB_layout.clear_widgets()
            
            
            self.username = ""
            self.password = ""
            self.token = "123456789"
            self.error = ""
            
            self.screen_manager = ScreenManager()
            self.screen_manager.add_widget(Builder.load_file("pre-splash.kv"))
            self.screen_manager.add_widget(Builder.load_file("login.kv"))
            self.screen_manager.add_widget(Builder.load_file("log.kv"))
            self.screen_manager.add_widget(Builder.load_file("register.kv"))
            
            self.img = cv2.imread("GasviCentrado.png",-1) # Cargar la imagen que se va a superponer

            #self.url = "http://homeassistant:7000/" #192.168.26.67
            #self.url = "http://192.168.215.67:7000/" #
            self.url = "http://192.168.0.101:7000/"
            self.web_cam = Image(size_hint=(1,1))

            mainLayout = MDScreen(name="cam", md_bg_color=(196/255, 166/255, 254/255, 1))
            #mainLayout = MDScreen(name="cam", md_bg_color=(141/255, 109/255, 139/255, 1))

            self.superBox = BoxLayout(orientation ='vertical', size_hint_y = 0.65, size_hint_x = 1)
            #self.HB = BoxLayout(orientation ='vertical')
            
            logo = Image(source = 'logo_gasvi_letras.png')
            
            # To position widgets next to each other,
            # use a horizontal BoxLayout.
            self.HB = BoxLayout(orientation ='horizontal', size_hint_y = 0.2)
            
            btn1 = Button(text ="Tutorial", on_release = self.tutorial)
            btn2 = Button(text ="Gestos", on_release = self.gestos)
            btn3 = Button(text ="Sobre nosotros", on_release = self.sobre_nosotros)
            
            # HB represents the horizontal boxlayout orientation
            # declared above
            self.HB.add_widget(btn1)
            self.HB.add_widget(btn2)
            self.HB.add_widget(btn3)
            
            # To position widgets above/below each other,
            self.VB = BoxLayout(orientation ='vertical')
            self.VB_layout = ScrollView(size_hint = (1, 1), do_scroll_x = True, do_scroll_y = False, scroll_type = ['bars', 'content'], bar_width = 8, bar_inactive_color =  [0, 0, 0, 1], bar_color = [0, 0, 0, 1])
            # self.VB_layout = ScrollView(size_hint = (1, 1), do_scroll_x = True, do_scroll_y = False, scroll_type = ['bars', 'content'], bar_inactive_color =  [0, 0, 0, 1], bar_color = [0, 0, 0, 1])
                    # center_x = 0.5, center_y = 0.5,
            # VB represents the vertical boxlayout orientation
            # declared above
            self.VB.add_widget(logo)
            self.superBox.add_widget(self.HB)
            self.superBox.add_widget(self.VB)

            camlayout = BoxLayout()
            camlayout.add_widget(self.web_cam)
            
            menucamLayout = GridLayout(rows = 2)
            menucamLayout.add_widget(self.superBox)
            menucamLayout.add_widget(camlayout)
            
            topLayout = GridLayout(cols=2) 
            topLayout.add_widget(menucamLayout)
            #topLayout.add_widget(Button(text='World 1')) # A movida da dereita
            topLayout.add_widget(self.scroll_layout)
            
            mainLayout.add_widget(topLayout)
            self.screen_manager.add_widget(mainLayout)

            self.hands = mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            self.mark=1
            self.cont=0
            self.cont_frames = 0
            self.num_frames=60
            self.prim_fase=True
            self.print_atras=True

            return self.screen_manager

        #%%MAIN
        def on_start(self):
            Clock.schedule_once(self.main, 4)
            
        def main(self,*args):
            self.screen_manager.current = "main"
            
        def logger(self):
            self.username_introduced = self.screen_manager.get_screen("login").ids.user.text
            self.password_introduced = self.screen_manager.get_screen("login").ids.password.text
            if self.username_introduced == self.username and self.password_introduced == self.password:
                self.screen_manager.transition = SlideTransition()
                self.screen_manager.transition.direction = "left"
                self.screen_manager.current = "cam"
                self.cam()
                
            else:
                self.error = "Nombre de usuario o contraseña incorrectos"
                self.screen_manager.add_widget(Builder.load_file("error.kv"))
                print(self.screen_manager.current)
                self.screen_manager.transition = RiseInTransition()
                self.screen_manager.current = "error"
                
        def cam(self):
            self.cap=cv2.VideoCapture(0)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Number of frames to capture
            num_frames = 10

            # Start time
            start = time.time()

            # Grab a few frames
            for i in range(0, num_frames) :

                ret, frame = self.cap.read()

            # End time
            end = time.time()

            # Time elapsed
            seconds = end - start

            # Calculate frames per second
            self.fps  = num_frames / seconds
            
            self.num_frames=60 #Frames que queramos que dure el video
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            self.fase = "00" # Este string lo usaremos pa controlar la fase

            self.rect_width = 150
            self.rect_height = 150
            
            # Definir rectángulo dedo click logo
            self.rect_width_logo = 82
            self.rect_height_logo = 82
            self.rect_x_logo = int(((self.width - self.rect_width) // 2)+self.rect_width_logo//2)
            self.rect_y_logo = int(((self.height - self.rect_height) // 12.5))
            
            Clock.schedule_interval(self.update, 1/33)
            
        def register(self):
            self.register_username = self.screen_manager.get_screen("signup").ids.user.text
            self.register_password = self.screen_manager.get_screen("signup").ids.password.text
            self.token_introduced = self.screen_manager.get_screen("signup").ids.token.text
            if (self.token_introduced == self.token) and (self.register_username != "" or self.register_password != "") and (" " not in self.register_username or " " not in self.register_password):
                self.username = self.register_username
                self.password = self.register_password
                self.screen_manager.transition = SlideTransition()
                self.screen_manager.transition.direction = "right"
                self.screen_manager.current = "main"
            elif (self.register_username == "" or self.register_password == "") or (" " in self.register_username or " " in self.register_password):
                self.error = "El usuario o contraseña no pueden estar vacíos"
                self.screen_manager.add_widget(Builder.load_file("error.kv"))
                self.screen_manager.transition = RiseInTransition()
                self.screen_manager.current = "error"
            else: 
                self.error = "El token introducido no es correcto"
                self.screen_manager.add_widget(Builder.load_file("error.kv"))
                self.screen_manager.transition = RiseInTransition()
                self.screen_manager.current = "error"
            
        def clearlog(self):
            self.screen_manager.get_screen("login").ids.user.text = ""
            self.screen_manager.get_screen("login").ids.password.text = ""
            
        def clearsignup(self):
            self.screen_manager.get_screen("signup").ids.user.text = ""
            self.screen_manager.get_screen("signup").ids.password.text = ""
            self.screen_manager.get_screen("signup").ids.token.text = ""
            
        def entendido(self):
            self.screen_manager.transition = FallOutTransition()
            if self.error == "Nombre de usuario o contraseña incorrectos":
                self.screen_manager.current = "login"
            else:
                self.screen_manager.current = "signup"
            self.screen_manager.remove_widget(self.screen_manager.get_screen('error'))
            
        def salta_popup(self, instance):
            """Añade un label en el centro que dura 5 segundos"""
            # popupLayout = BoxLayout(orientation ='horizontal')
            # lbl1 = Label(text='Mensaxe: '+instance, font_size='20sp', color=(1,1,1,1))
            
            self.popup.open()

        def tutorial(self,instance):
            # centerx = self.VB.width*2
            # tut = BoxLayout(orientation ='horizontal', size_hint_x = 0.5)
            # tut.center_x = centerx
            self.VB.clear_widgets()
            # tutorial = Label(text = """
            #                  1.Toca el símbolo de Gasvi para comenzar la captura de vídeo.\n
            #                  2.Comenzará una cuenta atrás para que te prepares.\n
            #                  3.Realiza el gesto deseado.\n
            #                  4.Espera respuesta del servidor.\n
            #                  5.Comprueba los resultados en la parte derecha.\n
            #                  (En el apartado gestos encontrarás todos los gestos disponibles con su correspondiente demostración)""",
            #                     font_size = 18, color = [0,0,0,1], halign = 'center', valign="middle")
            # tut.add_widget(tutorial)
            auxlay = GridLayout(rows = 7)
            auxlay.add_widget(Label(text = "1.Toca el símbolo de Gasvi para comenzar la captura de vídeo.\n"
                                ,font_size = 18, color = [0,0,0,1], size_hint=(1, 1)))
            auxlay.add_widget(Label(text = "2.Comenzará una cuenta atrás para que te prepares.\n"
                                ,font_size = 18, color = [0,0,0,1], size_hint=(1, 1)))
            auxlay.add_widget(Label(text = "3.Realiza el gesto deseado.\n"
                                ,font_size = 18, color = [0,0,0,1], size_hint=(1, 1)))
            auxlay.add_widget(Label(text = "4.Espera respuesta del servidor.\n"
                                ,font_size = 18, color = [0,0,0,1], size_hint=(1, 1)))
            auxlay.add_widget(Label(text = " 5.Comprueba los resultados en la parte derecha.\n"
                                ,font_size = 18, color = [0,0,0,1], size_hint=(1, 1)))
            auxlay.add_widget(Label(text = "(En el apartado gestos encontrarás todos los gestos \n "
                                ,font_size = 18, color = [0,0,0,1], size_hint=(1, 0.8)))
            auxlay.add_widget(Label(text = " disponibles con su correspondiente demostración)"
                                ,font_size = 18, color = [0,0,0,1], size_hint=(1, 0.5)))
                             
            self.VB.add_widget(auxlay)
        
        def gestos(self, instance):
            self.VB_layout.clear_widgets()
            # layout = GridLayout(cols = 23, size_hint_x = None, col_default_width = self.VB_layout.height*3, col_force_default = True)
            layout = GridLayout(cols = 23, size_hint_x = None, col_default_width = self.VB.height, col_force_default = True)
            layout.bind(minimum_width = layout.setter('width'))
            
            for i in range(23):
                
                image = Image(source = self.src[i], allow_stretch = True, anim_delay = 0.07, width = self.VB.height)
                layout.add_widget(image)
            self.VB.clear_widgets()
            
            self.VB_layout.add_widget(layout)
            self.VB.add_widget(self.VB_layout)
            
        def sobre_nosotros(self,instance):
            sobre_nosotros = Label(text = "Gasvi® ha sido creado por: ", font_size = 40, color = [0,0,0,1])
            nombre1 = Label(text = "Christian", font_size = 40, color = [0,0,0,1])
            nombre2 = Label(text = "Iñaki", font_size = 40, color = [0,0,0,1])
            nombre3 = Label(text = "Pablo", font_size = 40, color = [0,0,0,1])
            nombre4 = Label(text = "Sixto", font_size = 40, color = [0,0,0,1])
            self.VB.clear_widgets()
            self.VB.add_widget(sobre_nosotros)
            self.VB.add_widget(nombre1)
            self.VB.add_widget(nombre2)
            self.VB.add_widget(nombre3)
            self.VB.add_widget(nombre4)
        
                    
        def extract_keypoints_hand(self, results_hands, shoulder_center_x, shoulder_center_y, magnitud):
        
            """
        
            FUNCION PARA OBTENER PUNTOS MEDIAPIPE DE LAS MANOS (DERECHA E IZQUIERDA).
        
            """
        
                # Obtener las coordenadas de ambas manos
            rh_aux = np.zeros((2, 21, 3))
            if results_hands.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    for j, landmark in enumerate(hand_landmarks.landmark):
                        if i > 1 or j>21:
                            #print("Error i o j")
                            casa = 1
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
        
        
        
        def extract_keypoints_pose(self, results_pose, shoulder_center_x, shoulder_center_y, magnitud):
            
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
        
        def normalizacionCoordenadas(self, results_pose):
            
            xl, yl = results_pose.pose_landmarks.landmark[11].x, results_pose.pose_landmarks.landmark[11].y
            xr, yr = results_pose.pose_landmarks.landmark[12].x, results_pose.pose_landmarks.landmark[12].y
            shoulder_center_x, shoulder_center_y = (xr + yr) / 2, (xl + yl) / 2
            
            magnitud = math.sqrt((xr-xl)**2 + (yr-yl)**2)
            
            return shoulder_center_x, shoulder_center_y, magnitud
   # ------------------------------------------------------------------------------
   
        def blend_transparent(self, face_img, overlay_t_img):
        
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


        def fase00(self):
            
            #print("Fase 00")
            # Definimos var aux con los 0 frames
            self.coords = np.zeros((1, 60, 144))
            coords_aux = []
            coords_hands_aux = []
            coords_pose_aux = []

            # cv2.line(self.frame, (self.rect_x_der, self.rect_y_der), (self.rect_x_der+self.rect_width, self.rect_y_der), (0,255,0), (5)) # Lin derecha
            # cv2.line(self.frame, (self.rect_x_der, self.rect_y_der), (self.rect_x_der, self.rect_y_der+self.rect_height), (0,255,0), (5)) # Lin derecha
            # cv2.line(self.frame, (self.rect_x_izq, self.rect_y_izq), (self.rect_x_izq+self.rect_width, self.rect_y_izq), (0,255,0), (5)) # Lin izquierda
            # cv2.line(self.frame, (self.rect_x_izq+self.rect_width, self.rect_y_izq), (self.rect_x_izq+self.rect_width, self.rect_y_izq+self.rect_height), (0,255,0), (5)) # Lin izquierda
            cv2.putText(self.frame, "Pulsa", (140, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0, 255), 8)
            cv2.putText(self.frame, "GASVI", (140+self.rect_x_logo-35, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0, 255), 8)
            cv2.putText(self.frame, "Pulsa", (140, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (254, 166, 195, 255), 3)
            cv2.putText(self.frame, "GASVI", (140+self.rect_x_logo-35, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (254, 166, 195, 255), 3)

            img_resized = cv2.resize(self.img, (self.frame.shape[1], self.frame.shape[0]))
            self.frame = self.blend_transparent(self.frame, img_resized)
            
            #Si encuentra mano EN EL RECTÁNGULO se habilita cuenta atrás: falta self.results_pose.pose_landmarks and 
            if self.results_hands.multi_hand_landmarks:

                # x_right_shldr = self.results_pose.pose_landmarks.landmark[12].x
                # y_right_shldr = self.results_pose.pose_landmarks.landmark[12].y
                # x_left_shldr = self.results_pose.pose_landmarks.landmark[11].x
                # y_left_shldr = self.results_pose.pose_landmarks.landmark[11].y

                x_right_finger = self.results_hands.multi_hand_landmarks[0].landmark[8].x
                y_right_finger = self.results_hands.multi_hand_landmarks[0].landmark[8].y            

                # x_r_s=int(x_right_shldr * self.frame.shape[1]) #Para que non estea normalizado
                # y_r_s=int(y_right_shldr * self.frame.shape[0])
                # x_l_s=int(x_left_shldr * self.frame.shape[1])
                # y_l_s=int(y_left_shldr * self.frame.shape[0])

                x_r_f=int(x_right_finger * self.frame.shape[1]) #Normalizamos man dentro do frame
                y_r_f=int(y_right_finger * self.frame.shape[0]) 
                
                # Comprobar si la muñeca está dentro del rectángulo
                # if (self.rect_x_der < x_r_s < self.rect_x_der+self.rect_width) and (self.rect_y_der < y_r_s < self.rect_y_der+self.rect_height) and (self.rect_x_izq < x_l_s < self.rect_x_izq+self.rect_width) and (self.rect_y_izq < y_l_s < self.rect_y_izq+self.rect_height):
                if (self.rect_x_logo < x_r_f < self.rect_x_logo+self.rect_width_logo) and (self.rect_y_logo < y_r_f < self.rect_y_logo+self.rect_height):
                    self.cont_frames = self.cont_frames + 1
                    if self.cont_frames >= 3:
                        self.fase="01"    
                        self.cont_frames = 0
                        # print("Ámbolos dous hombreiros e dedo dentro das marcas")
                        # out = cv2.VideoWriter('GestoDetectado.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
                else: self.cont_frames = 0 # Reiniciamos  

    
    #------------------------------------------------------------------------------
        
        def fase01(self):

            #print("Fase 01")
            if self.cont_frames>=0 and self.cont_frames<40//3: 
                cv2.putText(self.frame, "3", (self.height//2, self.width//2), cv2.FONT_HERSHEY_SIMPLEX, 8, (0,0,0, 255), 14)
                cv2.putText(self.frame, "3", (self.height//2, self.width//2), cv2.FONT_HERSHEY_SIMPLEX, 8, (254,166,195, 255), 10)

            if self.cont_frames>=40//3 and self.cont_frames<40//2: 
                cv2.putText(self.frame, "2", (self.height//2, self.width//2), cv2.FONT_HERSHEY_SIMPLEX, 8, (0,0,0, 255), 14)
                cv2.putText(self.frame, "2", (self.height//2, self.width//2), cv2.FONT_HERSHEY_SIMPLEX, 8, (254,166,195, 255), 10)

            if self.cont_frames>=40//2 and self.cont_frames<40: 
                cv2.putText(self.frame, "1", (self.height//2, self.width//2), cv2.FONT_HERSHEY_SIMPLEX, 8, (0,0,0, 255), 14)
                cv2.putText(self.frame, "1", (self.height//2, self.width//2), cv2.FONT_HERSHEY_SIMPLEX, 8, (254,166,195, 255), 10)

            self.cont_frames = self.cont_frames + 1

            if self.cont_frames >= 40:
                
                self.fase = "10"
                #print("Frames por segundo: ", self.fps)
                self.out = cv2.VideoWriter('GestoDetectado.mp4', self.fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.cont_frames = 0

    #------------------------------------------------------------------------------

        def fase10(self):

            #print("Fase 10")
            self.out.write(self.frame)
            self.cont_frames = self.cont_frames + 1
            #print("Num frames: ", self.cont_frames)
            
            if self.cont_frames == 59:
                cv2.putText(self.frame, "Enviando datos al servidor...", (35,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0, 255), 7)
                cv2.putText(self.frame, "Enviando datos al servidor...", (35,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (254,166,195, 255), 3)


            if self.cont_frames >= 60:
                
                #print("Ata aquí a gravación")
                self.cont_frames = 0
                self.out.release()
                self.fase = "11" # Se va a leer el vídeo

    #------------------------------------------------------------------------------

        def fase11(self):

            #print("Fase 11")
            cap_vid = cv2.VideoCapture("GestoDetectado.mp4")
            
            #Sacar factor de escala para normalizacion
            # width = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            # horizontal_fov = math.radians(60)
            # f = (width/2) / math.tan(horizontal_fov/2)
            # distancia_camara = 1
            # SCALE_FACTOR = 1/f * distancia_camara

            while cap_vid.isOpened():

                ret_vid, frame_vid = cap_vid.read()
                frame_vid = cv2.flip(frame_vid,1)

                if ret_vid:

                    #print("Vaise ler o vídeo")
                    frame_vid_cambiado = cv2.cvtColor(frame_vid, cv2.COLOR_RGB2BGR)
                    results_hand_vid = self.hands.process(frame_vid_cambiado) 
                    results_pose_vid = self.pose.process(frame_vid_cambiado)

                    # Nueva fase pa leer el video
                    # Unimos los puntos del MediaPipe pose y hand
                    shoulder_center_x, shoulder_center_y, magnitud = self.normalizacionCoordenadas(results_pose_vid)
                    coords_hands_aux = self.extract_keypoints_hand(results_hand_vid, shoulder_center_x, shoulder_center_y, magnitud)
                    coords_pose_aux = self.extract_keypoints_pose(results_pose_vid, shoulder_center_x, shoulder_center_y, magnitud)
                    
                    coords_aux = np.concatenate([coords_hands_aux, coords_pose_aux])
                    # coords_aux = coords_hands_aux
     
                    self.coords[0][self.cont_frames] = coords_aux
                    self.cont_frames = self.cont_frames + 1

                    if self.cont_frames >= 60:

                        self.fase = "12"
                        self.cont_frames = 0
                        break

                else: 
                    print("GASVI TENME DANDO VOLTERETAS")

    #------------------------------------------------------------------------------

        def fase12(self):

            #print("Fase 12")
            # self.cont_frames = self.cont_frames + 1
            # if self.cont_frames >= 30:

            #print("Enviando datos ao servidor ...")
            # self.model = load_model('SoloManosFlip/ModeloGestoCNNSoloManosyPoseFlip_SinNorm.h5') 

            ###########################Enviar a la Raspi#####################################################
            if np.sum(self.coords != 0):
                # res = self.model.predict(self.coords)
                # print(res)
                # print(self.coords.tobytes())
                # msg = str(self.coords)
                msg = self.coords
                if msg.any() and self.connection:
                    self.connection.write(msg.encode('utf-8'))
                res = requests.get(self.url, json = msg.tolist())
                print(res.text) #Impprimimos resultado
                self.data = res.text
                self.add_panel(self)
                self.coords=np.zeros((1, 60, 144))
            ###########################Enviar a la Raspi#####################################################

            self.fase = "00"
            self.cont_frames = 0

    #------------------------------------------------------------------------------

        def update(self,*args):

            self.ret, self.frame = self.cap.read()
            self.frame = cv2.flip(self.frame,1)

            #Path para cargar el modelo
            #model = load_model('VideoTrainProgramaSinFlip/ModeloGestoCNN.h5') 

            # cap = cv2.VideoCapture(2)
            # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # print(width)
            # print(height)

            #Siempre a 0 para contar los frames

            # #Si la camara va a más de 30 fps recortamos a 30 constantes
            if (self.fps > 30):

                self.fps = 30

            # print("A cámara vai a: ",fps,"fps")
            # img = cv2.imread("silueta.png",-1) # Cargar la imagen que se va a superponer        

            if self.ret:

                # Procesar el cuadro con Mediapipe para obtener las coordenadas
                self.frame_cambiado = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                

                # img_resized = cv2.resize(img, (frame.shape[1], frame.shape[0])) ###0
                # # resultado = cv2.addWeighted(frame, 1, img_resized, 0, 0)# Superponer la imagen en el frame
                # frame = blend_transparent(frame, img_resized) ###O

                if self.fase == "00":
                    # self.results_pose = self.pose.process(self.frame_cambiado)
                    self.results_hands = self.hands.process(self.frame_cambiado)
                    self.fase00()

                if self.fase == "01":
                    self.fase01()

                if self.fase == "10":
                    self.fase10()

                if self.fase == "11":
                    self.fase11()

                if self.fase == "12":
                    self.fase12()                      


            else: print("Error durante a gravación")

            
            buf = cv2.flip(self.frame, 0).tobytes()
            img_texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture

    def process_json(data):
        text_label = ""
        text = 'error detección de respuesta'
        url_image  = 'Imagenes/error.png'
        popup = None
    
        size = '23'
        end_size = '[/size]'
        
        action = json.loads(data)
    
        if (action['funcion'] == 'encender_luz'):
            if(action['error'] == False):
                url_image = 'Imagenes/bombilla_encendida.png'
                text = 'Has encendio la bombilla'
            else:
                url_image = 'Imagenes/error.png'
                text = 'La bombilla ya está encendida'
    
        elif (action['funcion'] == 'apagar_luz'):
            if(action['error'] == False):
                url_image = 'Imagenes/bombilla_apagada.png'
                text = 'Has apagado la bombilla'
            else:
                url_image = 'Imagenes/error.png'
                text = 'La bombilla ya está apagada'
    
        elif (action['funcion'] == 'luz_roja'):
            if(action['error'] == False):
                url_image = 'Imagenes/bombilla_roja.png'
                text = 'La bombilla ha cambiado a color Rojo'
            else:
                url_image = 'Imagenes/error.png'
                text = 'Para color Rojo tiene que estar encendida'
                size = '18'
    
        elif (action['funcion'] == 'luz_azul'):
            if(action['error'] == False):
                url_image = 'Imagenes/bombilla_azul.png'
                text = 'La bombilla ha cambiado a color Azul'
            else:
                url_image = 'Imagenes/error.png'
                text = 'Para color Azul tiene que estar encendida'
                size = '18'
    
        elif (action['funcion'] == 'luz_verde'):
            if(action['error'] == False):
                url_image = 'Imagenes/bombilla_verde.png'
                text = 'La bombilla ha cambiado a color Verde'   
            else:
                url_image = 'Imagenes/error.png'
                text = 'Para color Verde tiene que estar encendida'
                size = '18'
    
        elif (action['funcion'] == 'luz_naranja'):
            if(action['error'] == False):
                url_image = 'Imagenes/bombilla_naranja.png'
                text = 'La bombilla ha cambiado a color Naranja'   
            else:
                url_image = 'Imagenes/error.png'
                text = 'Para color Naranja tiene que estar encendida'
                size = '18'
    
        elif (action['funcion'] == 'clima'):
            if(action['estado'] == 'cloudy' or action['estado'] == 'fog'):
                url_image = 'Imagenes/nube.png'
                text = 'El clima actual es Nublado'
            elif(action['estado'] == 'rainy' or action['estado'] == 'pouring'):
                url_image = 'Imagenes/lluvia.png'
                text = 'El clima actual es Lluvia'
            elif(action['estado'] == 'sunny'):
                url_image = 'Imagenes/sol.png'
                text = 'El clima actual es Soleado'
            elif(action['estado'] == 'partlycloudy'):
                url_image = 'Imagenes/nubesol.png'
                text = 'El clima actual es Parcialmente nublado'
            elif(action['estado'] == 'lightning' or action['estado'] == 'lightning-rainy'):
                url_image = 'Imagenes/tormenta.png'
                text = 'El clima actual es Tormenta'
            else:
                url_image = 'Imagenes/error.png'
                text = 'No tengo este clima implementado'
    
        elif(action['funcion'] == 'play'):
            if(action['genero'] == 'regueton'):
                url_image = 'Imagenes/regueton.png'
                text = 'Playlist de regueton: '+action['cancion']
            if(action['genero'] == 'reguee'):
                url_image = 'Imagenes/reguee.png'
                text = 'Playlist de reguee: '+action['cancion']
            if(action['genero'] == 'rock'):
                url_image = 'Imagenes/rock.png'
                text = 'Playlist de rock: '+action['cancion']
    
        elif(action['funcion'] == 'pause'):
            if(action['error'] == False):
                url_image = 'Imagenes/pause.png'
                text = 'Pausando la reproducción'
            else:
                url_image = 'Imagenes/error.png'
                text = 'No puedes pausar sin música activa'
    
        elif(action['funcion'] == 'resume'):
            if(action['error'] == False):
                url_image = 'Imagenes/play.png'
                text = 'Reanudando la reproducción'
            else:
                url_image = 'Imagenes/error.png'
                text = 'No puedes reanudar sin música pausada'
    
        elif(action['funcion'] == 'next'):
            if(action['error'] == False):
                url_image = 'Imagenes/next.png'
                text = 'Saltando a siguiente canción: '+action['cancion']
                size = '18'
            else:
                url_image = 'Imagenes/error.png'
                text = 'No puedes saltar sin música activa'
    
        elif(action['funcion'] == 'previus'):
            if(action['error'] == False):
                url_image = 'Imagenes/previus.png'
                text = 'Volviendo a anterior canción: '+action['cancion']
                size = '18'
            else:
                url_image = 'Imagenes/error.png'
                text = 'No puedes retroceder sin música activa'
    
        elif(action['funcion'] == 'alternar_puerta'):
            if(action['estado'] == 'abierta'):
                url_image = 'Imagenes/puerta_abierta.png'
                text = 'Abriendo la puerta'  
            if(action['estado'] == 'cerrada'):
                url_image = 'Imagenes/puerta_cerrada.png'
                text = 'Cerrando la puerta'  
    
        elif(action['funcion'] == 'time'):
            url_image = 'Imagenes/reloj.png'
            text = 'En este momento es '+action['hour']+':'+action['min']+':'+action['sec']+' del '+action['day']+'/'+action['month']+'/'+action['year']
    
        elif(action['funcion'] == 'enchufe'):
            if(action['estado'] == 'on'):
                text = 'Encendiendo enchufes'
                url_image = 'Imagenes/enchufe_encendido.png'
            if(action['estado'] == 'off'):
                text = 'Apagando enchufes'
                url_image = 'Imagenes/enchufe_apagado.png'
    
        elif(action['funcion'] == 'roomba'):
            if(action['estado'] == False):
                text = 'Desactivando Roomba'
                url_image = 'Imagenes/roomba.png'
            if(action['estado'] == True):
                text = 'Activando Roomba'
                url_image = 'Imagenes/roomba.png'
    
        elif(action['funcion'] == 'temporizador'):
            if(action['duration'] == "1"):
                text = 'Inicio del temporizador de 1 minuto'
                url_image = 'Imagenes/temporizador1.png'
            if(action['duration'] == "5"):
                text = 'Inicio del temporizador de 5 minuto'
                url_image = 'Imagenes/temporizador5.png'
    
        elif(action['funcion'] == 'juego'):
            if(action['partida_reseteada'] == False):
                text = 'Comenzando Piedra Papel Tijeras'
                url_image = 'Imagenes/juego.png'     
            if(action['partida_reseteada'] == True):
                text = 'Reseteando la partida'
                url_image = 'Imagenes/juego.png'
    
        elif(action['funcion'] == 'Piedra' or action['funcion'] == 'Tijera' or action['funcion'] == 'Papel'):
           if(action['partida_creada'] == False):
               text = 'Tienes que iniciar la partida'
               url_image = 'Imagenes/error.png' 
           if(action['partida_creada'] == True):  
               if(action['ganador'] == None):
                   text = "Has sacado "+ action['funcion'] +" y Gasvi "+action['eleccion_gasvi']+ ' Tu:'+action['puntos_user']+' Gasvi:'+action['puntos_gasvi']    
                   size = '18'
               if(action['ganador'] == 'user'):
                   popup = 'ganador_user'
                   text = "Has ganado con "+ action['funcion'] +" y Gasvi "+action['eleccion_gasvi']+ ' Tu:'+action['puntos_user']+' Gasvi:'+action['puntos_gasvi']    
                   size = '18'
               if(action['ganador'] == 'gasvi'):
                   text = "Has perdido con "+ action['funcion'] +" y Gasvi "+action['eleccion_gasvi']+ ' Tu:'+action['puntos_user']+' Gasvi:'+action['puntos_gasvi']    
                   popup = 'ganador_gasvi'  
                   size = '18'
               if(action['funcion'] == 'Piedra'):
                   url_image = 'Imagenes/piedra.png'
               if(action['funcion'] == 'Papel'):
                   url_image = 'Imagenes/papel.png'
               if(action['funcion'] == 'Tijera'):
                   url_image = 'Imagenes/tijera.png'    
                   
        elif(action['funcion'] == 'error_identificacion'):
           text= 'No se ha podido detectar tu gesto'
           url_image = 'Imagenes/interrogacion.png'
    
        else:
           text = 'Error al detectar la funcion'
           url_image = 'Imagenes/error.png'
        
    
        add_size = '[size='+size+']'
    
        text_label = add_size +text+ end_size
        return text_label, url_image, popup
        #return text_label, url_image
                            

    app = GasviApp(pipe_padre=padre_pipe)
    app.run()