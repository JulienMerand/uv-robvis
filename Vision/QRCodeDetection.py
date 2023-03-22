import cv2
import os
import numpy as np 
from cv2 import aruco


# #Initialize the dictionary
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# for i in range (1, 5):
#     size = 700
#     img = aruco.generateImageMarker(aruco_dict, i, size)
#     cv2.imwrite('./exemple_qrcode/image_'+str(i)+".jpg",img)    
#     cv2.imshow('artag',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows

def order_coordinates(pts, var):
    coordinates = np.zeros((4,2),dtype="int")

    if(var):
        #Parameters sort model 1 
        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]
        coordinates[3] = pts[np.argmax(s)] 

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]
        coordinates[2] = pts[np.argmax(diff)]
    
    else:
        #Parameters sort model 2 
        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]
        coordinates[2] = pts[np.argmax(s)] 

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]
        coordinates[3] = pts[np.argmax(diff)]
    
    return coordinates


image = cv2.imread(r"RobVis_image_2\exemple1.jpg")
h,w = image.shape[:2]

image = cv2.resize(image,(int(w*0.7), int(h*0.7)))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Init du dictionnaire aruco et lancement de la détection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters()

# Détection des coins et des id
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Affichage des markers
frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

# Initialise une liste vide pour garder les coordonnées du centre de
# chaque QRcode
params = []

for i in range(len(ids)):

    # Récupère les coins de chaque tag
    c = corners[i][0]

    # Dessin un cercle en chaque centre
    cv2.circle(image,(int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255,255,0), -1)
    
    # Sauvegarde les coordonnées de chaque centre dans params
    params.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

# Transfome la list en array
params = np.array(params)

# Cette étape est indispensable pour ordonner les centres
# car le détecteur les trouve mais dans un ordre quelconque
if(len(params)>=4):
    #Sort model 1 
    params = order_coordinates(params,False)
    
    #Sort Model 2
    params_2 = order_coordinates(params,True)

# Nous créons un mask de l'image initiale dans laquelle les pixels
# appartenant à la zone comprise entre les QRcodes sont à 255 et les autres à 0
mask = np.zeros([int(h*0.7), int(w*0.7),3], dtype=np.uint8)
cv2.fillConvexPoly(mask, np.int32([params]), (255, 255, 255), cv2.LINE_AA)

# combinaison entre le mask et l'image initiale
substraction = cv2.subtract(image,mask)

cv2.imshow('markers',frame_markers)
cv2.imshow('black mask',mask)
cv2.imshow('substraction',substraction)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Connexion à la camera en cours...")
cap = cv2.VideoCapture(0)
print("Connecté !")

while True:
    ret, frame = cap.read()
    paint = frame

    # Nous chargeons l'image que nous souhaitons insérer entre les QRcodes
    # paint = cv2.imread(r"RobVis_image_2\earth.jpg")
    height, width = paint.shape[:2]

    # Nous définissons les coordonnées de la région d'intérêt que nous
    # souhaitons afficher i.e toute l'image
    coordinates = np.array([[0,0],[width,0],[0,height],[width,height]])

    # Comme vous l'avez fait auparavant pour le feature matching nous 
    # calculons la transformation homographique entre l'image de la terre et
    # les coordonnées de la zone sur laquelle nous voulons insérer l'image en
    # question
    hom, status = cv2.findHomography(coordinates, params_2)
    
    # Ensuite nous la transformation selon cette homographie
    warped_image = cv2.warpPerspective(paint, hom, (int(w*0.7), int(h*0.7)))

    # insertion de l'image warpé dans la zone blanche. 
    addition = cv2.add(warped_image,substraction)

    
    cv2.imshow('detection',addition)
    

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()