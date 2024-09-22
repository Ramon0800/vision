import cv2
from ultralytics import YOLO
import numpy as np
import json

#Obteniendo los datos requeridos
model=YOLO('yolov8s.pt')
img_cat='OIP.jpg'
img_dog='dog2.jpg'

#Definiendo funcion para detectar gatos y perros
def detection(img_path):
    
#Pasando imgen a instancia del modelo    
    img=cv2.imread(img_path)
    results=model(img)
    
#Obteniendo cajas delimitadoras
    for result in results:
        boxes=result.boxes 
        for i in range(len(boxes.cls)):
            if boxes is not None and boxes.cls[i]==16 or boxes.cls[i]==15:
                point=boxes.conf[i]
                cls=boxes.cls[i]
                x1,y1,x2,y2=map(int,boxes.xyxy[i])
                
#Representando en imagen                
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                if cls==15:
                    cv2.putText(img,f'gato:{point:.2f}',(x1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                else:
                    cv2.putText(img,f'perro:{point:.2f}',(x1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                    
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return x1,y1,x2,y2
   
#Definiendo funcion que remplaza gato por perro
def dog_cat(img_cat):
    
#Obteniendo poligono
    with open('poligono2.json') as f:
        data=json.load(f) 
    
    for box in data['boxes']:
        poligono=np.array(box['points'],dtype=np.int32)
        
#Obteniendo punto de inicio de caja delimitadora del gato   
    img_final=cv2.imread(img_cat)
    x1,y1,x2,y2=detection(img_cat)
    x1=x1-60
    y1=y1-13

#Obteniendo coordenadas de caja delimitadora del perro y creando mascaras
    img_o=cv2.imread(img_dog)
    mask=np.zeros(img_o.shape[:2],dtype=np.uint8)
    cv2.fillPoly(mask,[poligono],255)
    mask_INV=cv2.bitwise_not(mask)
    x1_p,y1_p,x2_p,y2_p=detection(img_dog)
    h_destino=y2_p-y1_p
    w_destino=x2_p-x1_p

#Recortando las imagenes necesarias para sumar las mascaras   
    img_o=img_o[y1_p:y2_p, x1_p:x2_p]
    mask=mask[y1_p:y2_p, x1_p:x2_p]
    mask_INV=mask_INV[y1_p:y2_p, x1_p:x2_p]
    
#Asegurando q todos tengan el mismo tamaño    
    img=img_final[y1:y1+h_destino, x1:x1+w_destino]
    mask_INV=cv2.resize(mask_INV,(img.shape[1],img.shape[0]))
    mask=cv2.resize(mask,(img.shape[1],img.shape[0]))
    img_o=cv2.resize(img_o,(img.shape[1],img.shape[0]))

#Sumando mascaras y obteniendo la imagen final
    img_objeto=cv2.bitwise_and(img_o,img_o,mask=mask)
    img_fondo=cv2.bitwise_and(img,img,mask=mask_INV)
    resultado=cv2.add(img_fondo,img_objeto)
    resultado=cv2.resize(resultado,(img.shape[1],img.shape[0]))
    h_results,w_results=resultado.shape[:2]
    img_final[y1:y1+h_results,x1:x1+w_results]=resultado
    
    cv2.imshow('img_final',img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
dog_cat(img_cat)



