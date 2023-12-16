# RODOR algorithm for coin counting, modules
# Based on paper RODOR algorithm for the identification of irregular surfaces with circular 
# contours applied to the recognition and counting of coins
# Paper and code by Msc Evelyn Orellana and Msc Edgar Rodriguez. 2023
# Paper on https://www.revistatoolbar.com/
# Email :    eorellana@revistatoolbar.com,     erodriguez@revistatoolbar.com
#
# This project is licensed under the terms of the MIT License

import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from matplotlib import pyplot as plt
import csv
import time
import pickle

pxWhite         = 255                   # Pixel white
pxBlack         = 0                     # Pixel black
InputDir        = 'Input/'              # Input files directory
OutputDir       = 'Output/'             # Output files directory
OutputLogDir    = 'Output/ImageLog/'    # Output log image files
FileExtension   = '.jpg'                # File image extension acepted
Accuracy        = 95                    # Minimum accuracy of evaluation
AccuracyDiameter= 95                    # Minimum accuracy of relation x, y diameters
LogImage        = True                  # If true generate runtime log files image

def fnReadImage( Path, ImgFileName ):
    # Read Image Color and Gray
    ImageColor = cv.imread( Path + ImgFileName + FileExtension, cv.IMREAD_COLOR ) 
    ImageGray  = cv.cvtColor(ImageColor, cv.COLOR_BGR2GRAY)
    
    return ImageColor, ImageGray
   
def fnSaveImagen( Path, ImgFileName, ImgPostfix, ImgFile):
    # Save image
    cv.imwrite( Path + ImgFileName + ImgPostfix + FileExtension, ImgFile)
    
def fnShowImage( ImgVector, Description, ImgSize, Position):
    # Display images on screen
    Canva = plt.figure(figsize = ImgSize) 

    Container = [] # Image control conteiner vector
    for ax in range(0, 3 ):
        if( Description[ax] != '' ):
            PosImg = Position[ax]
            Container.append( Canva.add_subplot(PosImg))
            Container[ax].title.set_text(Description[ax]) 
            Container[ax].imshow(ImgVector[ax], cmap='gray') 
    
    plt.show()

def fnImageBorder(ImageGray):
    # Generate border image from binarized
    ImageThreshold = np.array(Image.fromarray(~ImageGray < 10 )).astype(np.uint8) * pxWhite 
    ImageMedianBlur = cv.medianBlur(ImageThreshold, 5)
    ImageBinarized = fnBinarizedImg(ImageMedianBlur)
    ImgBorder = ImageBinarized.copy()
    
    return ImgBorder

def fnBinarizedImg(ImgSource):
    #Binarize image
    ImgBin = ImgSource.copy()
    
    for y in range(0, len(ImgBin) ):
        for x in range(0, len(ImgBin[y]) ):
            if( ImgBin[y][x] == pxWhite ):
                ImgBin[y][x] = pxBlack
            else:
                ImgBin[y][x] = pxWhite
    return ImgBin

def fnHammingDistance(Img1, Img2):
    # Hamming Distance
    HamDist = 0
    for y in range(0, len(Img1) ):
        for x in range(0, len(Img1[y]) ):
            if( Img1[y][x] == pxWhite ):
                if( Img1[y][x] == Img2[y][x] ):
                    HamDist += 1
    return HamDist

def fnHDsunken(ImgCanny, ImgBorder):
    # Sunken Edge
    ImgCannyLoc = ImgCanny.copy()
    NegBorder = ImgBorder.copy()
    
    kernel = np.ones((2, 2), np.uint8)
    ImgDilat1   = ~cv.dilate(ImgCannyLoc, kernel, iterations=1) 
    ImgDilation = ~cv.add(~NegBorder,~ImgDilat1)
    HDSunken = fnHammingDistance(ImgDilation, np.ones(ImgDilation.shape, np.uint8) * pxWhite)
    return HDSunken, ImgDilation
            
def fnAccuracy( Val1, Val2 ):
    pAccuracy = 0
    if( Val1 > Val2 ):
        pAccuracy = Val2 / Val1 * 100
    else :
        pAccuracy = Val1 / Val2 * 100
    return pAccuracy

def fnDiameter(Img1, y, x):
    # Y-axis diameters
    Diameter = 0
    if( Img1[y][x] == pxWhite ):
        if( Img1[y-1][x] == pxBlack ):
            for y1 in range(y, len(Img1) ):
                if( Img1[y1][x] == pxWhite ):
                    Diameter += 1
                else :
                    break;
    return Diameter
           
def fnCircle(Img1, y, x):
    # Find X-axis Y-axis diameter
    x1 = 0
    x2 = 0
    Diameter = 0
    Circle = False

    DiameterY = fnDiameter(Img1, y, x) 
    if ( DiameterY > 0 ):
        Radio = int( (DiameterY / 2) + 0.5 )
        
        for xm in reversed(range(x - DiameterY, x)):
            if( Img1[y + Radio][xm] != pxWhite ):
                x1 = xm + 1
                break;
        
        for xp in range(x, x + DiameterY):
            if( Img1[y + Radio][xp] != pxWhite ):
                x2 = xp
                break;
                
        DiameterX = x2 - x1
        if( fnAccuracy((DiameterX), DiameterY) > AccuracyDiameter ):
            Diameter = int( ((DiameterY + DiameterX) / 2) + 0.5 )
            Circle = True

    return Circle, Diameter, x1, y   # x, y in Left top corner ( LTC ) of image

def fnFindCoins(ImgFile):
    # Find coins circles
    CoinsCircles = []
    ImageColor, ImageGray = fnReadImage( InputDir + 'TestImg/', ImgFile )
    
    ImgBorder = fnImageBorder(ImageGray)

    Height, Width  = ImgBorder.shape
    for line in range(0, Height ):
        for column in range(0, Width ):
            Circle, Diameter, xLTC, yLTC = fnCircle(ImgBorder, line, column)
            
            if( Circle == True ):                
                HCircles=cv.HoughCircles(ImgBorder[yLTC-2:yLTC+Diameter+2, xLTC-2:xLTC+Diameter+2], cv.HOUGH_GRADIENT_ALT, 1.5, 50, 
                                         param1=100, param2=.8,minRadius=0,maxRadius=0)
                if( type(HCircles) is np.ndarray ):
                    DiameterC = int(HCircles[0,:2][0][2] + 0.5) * 2  # [0,:2]=First Circle [0]=First Array [2]=Radio ([0]=x [1]=y)
                    RadioC    = int(DiameterC / 2)
                    
                    cv.circle(ImgBorder,(xLTC+RadioC,yLTC+RadioC),RadioC+2,(150,150,150),-1) #Draw Circle
                    CoinsCircles.append([Diameter,xLTC,yLTC])  # x, y in Left top corner ( LTC ) of image
                
    if( LogImage == True ):
        fnSaveImagen( OutputLogDir, 'FindCoins-BorderCircle-', ImgFile, ImgBorder)
    return CoinsCircles, ImageColor, ImageGray

def fnTagPosXY(Radio, x, y, Pos):
    # Tag coin image
    if( Pos == 'Top' ):
        y = y + Radio - int(Radio * .80)
        
    if( Pos == 'Middle' ):
        x = x + Radio - int(Radio * .30)
        y = y + Radio - int(Radio * .35)
        
    if( Pos == 'BottonLeft' ):
        x = x + Radio - int(Radio * .85)
        y = y + Radio + int(Radio * .40)

    if( Pos == 'BottonRight' ):
        x = x + Radio - int(Radio * .60)
        y = y + Radio + int(Radio * .80)        
    return x, y

def fnImageProcess(ImageColor, ImageGray, Type, ImageName):
    # Calculates diameter, radius and segments of relief and sunken relief
    Diameter = 0
    Radio = 0
    hdSegmentEdge = 0
    hdSegmentSunken = 0
    
    #Image Processing
    ImageGrayBorder = cv.copyMakeBorder(ImageGray,2,2,2,2,cv.BORDER_CONSTANT,value=(pxWhite,pxWhite,pxWhite))
    ImgBorder = fnImageBorder(ImageGrayBorder)
    ImgCircles= cv.HoughCircles(ImgBorder,cv.HOUGH_GRADIENT_ALT,1.5,50,param1=100,param2=.8,minRadius=0,maxRadius=0) # Houg segments

    #Data image
    Diameter = int(ImgCircles[0,:2][0][2] + 0.5) * 2  # [0,:2]=First Circle [0]=First Array [2]=Radio ([0]=x [1]=y)
    Radio    = int(Diameter / 2)

    ImgCanny = cv.Canny(ImageGrayBorder,100,200)
    hdSegmentEdge = fnHammingDistance(ImgCanny, np.ones(ImgCanny.shape, np.uint8)*255) # Canny segments count
    
    hdSegmentSunken, ImgDilation1 =   fnHDsunken(ImgCanny, ImgBorder)
    
    if( LogImage == True ):
        fnSaveImagen( OutputLogDir, Type+'-GrayOri-', ImageName, ImageGrayBorder)
        fnSaveImagen( OutputLogDir, Type+'-BorderHou-', ImageName, ImgBorder)
        fnSaveImagen( OutputLogDir, Type+'-CannyEdge-', ImageName, ImgCanny)
        fnSaveImagen( OutputLogDir, Type+'-DilationSunken-', ImageName, ImgDilation1)
        
    return Diameter, Radio, hdSegmentEdge, hdSegmentSunken
