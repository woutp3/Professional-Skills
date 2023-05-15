import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import threading

alpha = 2 #contrast op de afbeelding vergroten met 2
beta = -127 #helderheid van de afbeelding verkleinen met 127

cap = cv2.VideoCapture(0) #video "cap" is afkomstig van de PC camera

#We stellen de grote van de afbeelding in 
cap.set(3, 640) #De afbeelding is in de x-as 640 pixels groot
cap.set(4, 480) #De afbeelding is in de y-as 480 pixels groot

#We laden de AI met object herkenning. Deze wordt gebruikt om de letters op de nummerplaat te lezen
reader = easyocr.Reader(['en'])

#We starten de loop waarin de afbeeldingen worden geprocessed
while True:

    number_plate = "None" 
    #We halen de afbeelding op van de video
    success, img = cap.read() #We slaan een frame van de video, afkomstig van onze camera, op als een variabele (img)

    #We gaan de afbeelding aanpassen
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta) #We verhogen het contrast van de afbeelding met de eerder gegeven waarden en verlagen de helderheid met de eerder gedefinieerde waarden
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #We zetten de aangepaste afbeelding om in een zwart-wit afbeelding
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #We gaan de noise (ruis) van de afbeelding verlagen en de randen van objecten op de afbeelding verscherpen

    #We zoeken rechthoeken in de aangepaste afbeelding
    edged = cv2.Canny(bfilter, 30, 200) #We gaan randen van objecten in de afbeelding detecteren
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #We gaan de hoeken van deze randen detecteren en de omtrek van de aangesloten punten opslaan
    contours = imutils.grab_contours(keypoints) #We gaan de gedetecteerde randomtrekken van deze objecten opslaan. Hierbij worden de hoeken van deze omtrek als punten in 2 dimensionele ruimte, bijvoorbeeld [210,180] voor het punt dat zich bevindt op x-locatie 210 en y-locatie 180
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #We gaan de 10 grootste omtrekken opslaan
    location = None

    #We gaan zien of de gedecteerde omtrekken vier hoeken hebben en dus een vierhoek zijn. Als dit het geval is gaan we deze omtrek opslaan als 4 punten (hoeken) in 2 dimensionale ruimte
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    #We creëren een masker ter grote van de originele afbeelding
    mask = np.zeros(gray.shape, np.uint8)

    #We proberen op de gedetecteerde vierhoek letters en cijfers te lezen. Als dit mislukt wordt enkel de opgeslagen frame die in het begin was opgeslagen getoond
    try:
        new_image = cv2.drawContours(mask, [location], 0,255, -1) #We gaan een omtrek tekenen op de punten die we eerder hadden opgeslagen. De omtrek komt overeeen met de gedetecteerde rechthoek
        new_image = cv2.bitwise_and(img, img, mask=mask) #We vergelijken de originele afbeelding die we hadden opgeslagen met het masker op bit-niveau
        (x,y) = np.where(mask==255) #We definiëren de x en de y coordinaten waar het masker zich bevindt
        (x1, y1) = (np.min(x), np.min(y)) #We definiëren de minimum x- en y-coördinaten van de locatie van het masker
        (x2, y2) = (np.max(x), np.max(y)) #We definiëren de maximum x- en y-coördinaten van de locatie van het masker
        cropped_image = gray[x1:x2+1, y1:y2+1] #We creëeren een nieuwe afbeelding die even groot is als het masker dat we hebben gecreëerd

        result = reader.readtext(cropped_image) #Op de verkleinde afbeelding laten we de tekst lezen door middel van een andere AI

        # Als er tekst is gedetecteerd dan wordt deze weergegeven en opgeslagen
        if result[0][1]:
            print(result[0][1])
            number_plate = (result[0][1])
            text = result[0][-2]
            font = cv2.FONT_HERSHEY_SIMPLEX

            #We gaan de tekst en een omtrek op het gedetecteerde object plaatsen op de originele afbeelding
            cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Als er geen nummerplaat is herkend wordt de originele afbeelding weergegeven
    except:
        img = img
    #We tonen de afbeelding op het computerscherm
    cv2.imshow('numberPlate', img)

    #Als we op de 'q' knop drukken sluit het programma zich af
    if cv2.waitKey(1) and 0xFF == ord('q'):
            break

#We sluiten het programma af
cap.release()
cv2.destroyAllWindows()