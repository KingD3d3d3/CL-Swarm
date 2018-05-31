from __future__ import division
# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import copy

# Importing the pytorch packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '768')

Config.set('modules', 'monitor', '')


goUpdate = False #True then update main Game loop 

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AIs, which we call "brain", and that contains our neural network that represents our Q-function
numAgents = 1
action2rotation = [0,20,-20]
brain = []
last_reward = []
scores = []
last_distance = []

# Config for multiple agents
for i in range(numAgents):   
    brain.append(Dqn(5,3,0.9))
    last_reward.append(0)
    scores.append([])
    last_distance.append(0)  # last distance of car to goal
    
# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((largeur,hauteur))
    goal_x = [20] * numAgents
    goal_y = [hauteur - 20] * numAgents
    first_update = False

# Creating the car class

class Car(Widget):
    
    carId = NumericProperty(0)
    debugCircle = 50
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>largeur-10 or self.sensor1_x<10 or self.sensor1_y>hauteur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>largeur-10 or self.sensor2_x<10 or self.sensor2_y>hauteur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>largeur-10 or self.sensor3_x<10 or self.sensor3_y>hauteur-10 or self.sensor3_y<10:
            self.signal3 = 1.
                
class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):
    
    car = []
    ball1 = []
    ball2 = []
    ball3 = []
    
    def serve_car(self):
        
        for i in range(numAgents):
            c = Car()
            b1 = Ball1()
            b2 = Ball2()
            b3 = Ball3()
            self.car.append(c)
            self.ball1.append(b1)
            self.ball2.append(b2)
            self.ball3.append(b3)
            self.add_widget(c)
            self.add_widget(b1)
            self.add_widget(b2)
            self.add_widget(b3)
        
        for i in range(numAgents):
            self.car[i].center =  (randint(1,1023),randint(1,767)) # self.center
            self.car[i].carId = i
            self.car[i].velocity = Vector(6, 0)
            
    
    def update(self, dt):
        global goal_x
        global goal_y
        global largeur
        global hauteur
        
        global brain
        global last_reward
        global scores
        global last_distance

        largeur = self.width
        hauteur = self.height
        if first_update:
            # Draw goals
            with self.canvas:
                Color(1,0,0)
                self.ellipse1 = Ellipse(pos=(20, hauteur - 40), size=(30.,30.))
                self.ellipse2 = Ellipse(pos=(largeur - 40, 20), size=(30.,30.))
            init()
        
        if(not goUpdate):
            return
        
        for i in range(numAgents):
            xx = goal_x[i] - self.car[i].x
            yy = goal_y[i] - self.car[i].y
            orientation = Vector(*self.car[i].velocity).angle((xx,yy))/180.
            #print('orientation', orientation)
            last_signal = [self.car[i].signal1, self.car[i].signal2, self.car[i].signal3, orientation, -orientation]
            action = brain[i].update(last_reward[i], last_signal)
            scores[i].append(brain[i].score())
            rotation = action2rotation[action]
            self.car[i].move(rotation)
            distance = np.sqrt((self.car[i].x - goal_x[i])**2 + (self.car[i].y - goal_y[i])**2)
            self.ball1[i].pos = self.car[i].sensor1
            self.ball2[i].pos = self.car[i].sensor2
            self.ball3[i].pos = self.car[i].sensor3
    
            # sand
            if sand[int(self.car[i].x),int(self.car[i].y)] > 0:
                self.car[i].velocity = Vector(1, 0).rotate(self.car[i].angle)
                last_reward[i] = -1
            else: # otherwise
                self.car[i].velocity = Vector(6, 0).rotate(self.car[i].angle)
                last_reward[i] = -0.5 # -0.5
                if distance < last_distance[i]: # getting closer
                    last_reward[i] = 0.1 # 0.1
            # Out of map
            if self.car[i].x < 10:
                self.car[i].x = 10
                last_reward[i] = -1
            if self.car[i].x > self.width - 10:
                self.car[i].x = self.width - 10
                last_reward[i] = -1
            if self.car[i].y < 10:
                self.car[i].y = 10
                last_reward[i] = -1
            if self.car[i].y > self.height - 10:
                self.car[i].y = self.height - 10
                last_reward[i] = -1
    
            if distance < 100:
                goal_x[i] = self.width-goal_x[i]
                goal_y[i] = self.height-goal_y[i]
                last_reward[i] = 1
            last_distance[i] = distance
            
            # Do only for car num 0 now
            #if i==0:
            #    self.checkDistanceToOthers(self.car[i])
        
            self.checkDistanceToOthers(self.car[i])
            
    def checkDistanceToOthers(self, c):
        # c is myself
        for i in range(numAgents):
            if i == c.carId:
                continue
            if Vector(c.pos).distance(self.car[i].pos) < self.car[i].debugCircle:
                
                #print("score car " + str(c.carId) + " is : " + str(brain[c.carId].score()))
                #print("score car " + str(i) + " is : " + str(brain[i].score()))
                #print("***")
                # Exchange, compare and update brains 
                if (brain[c.carId].score() > brain[i].score()):
                    
                    # Copy Model to others
                    brain[i].model = copy.deepcopy(brain[c.carId].model)
                    brain[i].optimizer = copy.deepcopy(brain[c.carId].optimizer)
                    
                    # Share Experiences
                    #if brain[c.carId].memory.tree.dataCount >100:
                    #   brain[i].memory.receive(brain[c.carId].memory.sample_event(100))
                
                    pass 
        
        
# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class Car2App(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'Go', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((largeur,hauteur))

    def save(self, obj):
        global goUpdate
        goUpdate = not goUpdate
        #print("saving brain...")
        #brain.save()
        #globalScore = [sum(x)/numAgents for x in zip(*scores)]
        #plt.plot(globalScore)
        #plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain[0].load()

# Running the whole thing
if __name__ == '__main__':
    Car2App().run()
