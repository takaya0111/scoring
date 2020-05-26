import PyPDF2
import natsort
import cv2
import numpy as np
import pdf2image
from pdf2image import convert_from_path
import os.path
import img2pdf
from PIL import Image

#赤丸の描画,半径22
def red_circle(im,x,y):
    cv2.circle(im, (x, y), 22, color=(0, 0, 245), thickness=2)

#ばつ印
def batu(im,x,y):
    x1 = x-18
    x2 = x+18
    y1 = y-18
    y2 = y+18
    cv2.line(im, (x1-2, y1-2), (x2+2, y2+2), color=(223, 223, 255), thickness=3)
    cv2.line(im, (x1-2, y2+2), (x2+2, y1-2), color=(223, 223, 255), thickness=3)
    cv2.line(im, (x1, y1), (x2, y2), color=(120, 120, 255), thickness=2)
    cv2.line(im, (x1, y2), (x2, y1), color=(120, 120, 255), thickness=2)

#座標変換

def identity1(im,f1,f2):
    h, w = im.shape[:2]
    src = f2.astype(np.float32)
    dest = f1.astype(np.float32)
    M = cv2.getPerspectiveTransform( src,dest)
    return cv2.warpPerspective(im, M, (w, h), borderValue=(255, 255, 255))

def EstimatHelmart(srcPoint, dstPoint):
    hsX1 =0.0
    hsY1 =0.0
    hsX2 =0.0
    hsY2 =0.0
    hsn  =0.0
    hsM1 =0.0
    hsM2 =0.0
    hsM3 =0.0

    for i in range(len(srcPoint[:])):
        hX1=srcPoint[i][0]
        hY1=srcPoint[i][1]
        hX2=dstPoint[i][0]
        hY2=dstPoint[i][1]

        hsX1 += hX1
        hsY1 += hY1
        hsX2 += hX2
        hsY2 += hY2

        hsn  += 1
        hsM1 += hX1*hX2 + hY1*hY2
        hsM2 += hY1*hX2 - hX1*hY2
        hsM3 += hX1*hX1 + hY1*hY1

    if hsn < 0 :
        # 計算できない場合は変換無しで返す
        htA = 1
        htB = 0
        htC = 0
        htD = 0
    else:
        htA= (hsX1*hsX2+hsY1*hsY2-hsn*hsM1) / (hsX1*hsX1+hsY1*hsY1-hsn*hsM3)
        htB= (hsY1*hsX2-hsX1*hsY2-hsn*hsM2) / (hsX1*hsX1+hsY1*hsY1-hsn*hsM3)
        htC= (hsX2-htA*hsX1 - htB*hsY1) / hsn   
        htD= (hsY2-htA*hsY1 + htB*hsX1) / hsn

    return htA, htB, htC, -htB, htA, htD

#アフィン変換の座標を取得
def GetAffinePos( Pt, AfPrm):
    return Pt[0] * AfPrm[0] + Pt[1] * AfPrm[1] + AfPrm[2], Pt[0] * AfPrm[3] + Pt[1] * AfPrm[4] + AfPrm[5]

#２つの座標の距離
def DistancePoints( Pt1, Pt2):
    return ((Pt1[0]-Pt2[0])**2 + (Pt1[1]-Pt2[1])**2)**0.5

#サイズ・チャンネル数の取得
def GetShape( srcImg):
    if len( srcImg.shape) >= 3:
        height, width, channel = srcImg.shape[:3]
    else:
        height, width = srcImg.shape[:2]
        channel = 1
    return height, width, channel

#特徴点の探索
def DetectKeyPoint( detector, srcImg, maxsize=None):
    
    height, width, channel = GetShape( srcImg)

    if channel == 1:
        img = srcImg
    else:
        img = cv2.cvtColor( srcImg, cv2.COLOR_BGR2GRAY) 

    if maxsize is None:
        scale = 1
        imgSz = img
    elif max( height, width) <= maxsize:
        scale = 1
        imgSz = img
    else:
        scale = maxsize / (height if height > width else width)
        imgSz = cv2.resize( img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    kp, des = detector.detectAndCompute( imgSz, None)

    return kp, des, scale, imgSz

#マッチングのフィルタ　（distance による）
def MatchFilterDist(srcMatches, exceptMatches=None, distThreshhold=None):
    if exceptMatches is None:
        badmatch = []
    else:
        badmatch = exceptMatches

    if distThreshhold is None:
        goodmatch = srcMatches
    else:
        goodmatch = []
        
        for m in srcMatches:
            if m.distance <= distThreshhold:
                goodmatch.append(m)
            else:
                badmatch.append(m)

    return goodmatch, badmatch

#座標の取得
def GetMatchPoints( kp1, scale1, kp2, scale2, match):
    srcPt = []
    dstPt = []
    for m in match:
        srcPt.append( (kp1[m.queryIdx].pt[0] / scale1, kp1[m.queryIdx].pt[1] / scale1) )
        dstPt.append( (kp2[m.trainIdx].pt[0] / scale2, kp2[m.trainIdx].pt[1] / scale2) )
    return srcPt, dstPt

#マッチングのフィルタ(ヘルマート変換の推定からの距離による)
def MatchFilterHelmart(kp1, scale1, kp2, scale2, srcMatches, exceptMatches=None, distThreshhold=3):
    if exceptMatches is None:
        badmatch = []
    else:
        badmatch = exceptMatches

    #ヘルマート変換のパラメータを推定する
    srcPt, dstPt  = GetMatchPoints( kp1, scale1, kp2, scale2, srcMatches)
    AffineParams = EstimatHelmart( srcPt, dstPt)
   
    dist=[]
    for i in range(len(srcPt)):
        dstPtEst = GetAffinePos( srcPt[i], AffineParams)
        dist.append( DistancePoints( dstPt[i], dstPtEst ))

    goodmatch = []
    avg = np.average( dist)
    std = np.std( dist)
    for i in range(len(srcMatches)):
        if  abs( dist[i] - avg) <= distThreshhold * std:
            goodmatch.append( srcMatches[i])
        else:
            badmatch.append( srcMatches[i])

    return goodmatch, badmatch

def check_support(im,x,y,ox,oy,point=0):
    global score
    img_1=im.copy()
    img_1=img_1[x-15:x+20,y-12:y+12]
    if img_1.sum()/((img_1.shape)[0]*(img_1.shape[1]*3)) <190:
        red_circle(im,oy,ox)    
        score += point
    else:
        batu(im,oy,ox)

def check2013(im):
    img = im.copy()
    scores=[]
    global score
    score=0
    
    check_support(img,630,351,630,210,2)
    check_support(img,685,405,685,210,2)
    check_support(img,740,295,745,210,2)
    check_support(img,800,463,800,210,2)
    check_support(img,855,379,855,210,2)
    
    scores.append(score)
    score=0
    
    check_support(img,915,295,915,210,2)
    check_support(img,970,295,970,210,2)
    check_support(img,1030,323,1030,210,2)
    check_support(img,1085,295,1085,210,2)
    check_support(img,1140,351,1140,210,2)
    check_support(img,1195,323,1195,210,2)
    check_support(img,1250,323,1250,210,2)
    check_support(img,1305,323,1305,210,2)
    check_support(img,1365,263,1365,210,2)
    check_support(img,1420,323,1420,210,2)
    
    check_support(img,630,792,630,715,1)
    check_support(img,685,850,685,715,1)
    check_support(img,740,792,740,715,1)
    check_support(img,798,768,798,714,1)
    check_support(img,855,768,855,714,1)
    
    scores.append(score)
    score=0
    score_support=0
    check_support(img,910,822,910,714,1)
    check_support(img,970,768,970,713,1)
    if score==2:
        score_support +=3
    score=0
    check_support(img,1030,792,1030,713,1)
    check_support(img,1085,905,1085,713,1)
    if score==2:
        score_support +=3
    score=0
    check_support(img,1140,850,1140,712,1)
    check_support(img,1200,768,1200,712,1)
    if score==2:
        score_support +=3
    scores.append(score_support)
    score=0
    
    check_support(img,1255,820,1255,712,3)
    check_support(img,1310,958,1310,711,1)
    check_support(img,1370,820,1370,711,1)
    check_support(img,1430,764,1430,710,1)
    
    check_support(img,630,1352,630,1215,3)
    check_support(img,690,1270,690,1215,3)
    check_support(img,745,1352,745,1215,3)
    check_support(img,802,1270,802,1215,3)
    
    scores.append(score)
    
    img[:30,:]=255
    img[1600:,:]=255
    img[:,:30]=255
    img[:,2300:]=255
    #     print(scores)
    return img

image_standard_s = convert_from_path("/Users/hasegawatakashikana/Downloads/download/standard.pdf")
image_standard = image_standard_s[0]
image_standard = np.array(image_standard)
#採点するpdfが入ったフォルダへのpath
before_file_path="/Users/hasegawatakashikana/Desktop/before"
#採点後のpdfを出力するファイルへのpath
after_path= "/Users/hasegawatakashikana/Desktop/after"

for a in os.listdir(before_file_path):
    if ".pdf" in a:
        a_path = before_file_path+"/" +a
        before = convert_from_path(a_path)
        before_0 = before[0]
        before_0 = np.array(before_0)
        image_test = np.array(before_0)
        if image_test[90:190,400].mean()>250:
            W,H = image_test.shape[:2]
            image_test = image_test[80:W-80,80:H-80]
            image_test=cv2.resize(image_test,(H,W))
        img1 = image_standard.copy()
        # オリジナル答案のマッチングに不要な情報削除→かなり白紙にする
        img1[300:1300,110:1600] = 255
        img1[1000:,370:1700]=255
        img2 = image_test.copy()
        #　採点する答案の端っこの余分な情報削除
        img2[:,2000:] = 255
        img2[1000:,500:1600] = 255

        
        #特徴点の検索
        detecter = cv2.AKAZE_create() #AKAZE
        
        kp1, des1, scale1, imgSz1 = DetectKeyPoint( detecter, img1, 1200)
        kp2, des2, scale2, imgSz2 = DetectKeyPoint( detecter, img2, 1200)
        
        #マッチング
        bf= cv2.BFMatcher( cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match( des1, des2)
        
        #マッチングの選別
        goodmatch, badmatch1 = MatchFilterDist( matches, None, 50) #distance が75以下をgoodmatch
        goodmatch, badmatch2 =MatchFilterHelmart( kp1,scale1, kp2, scale2, goodmatch, None,3)
        
        img1_pt = [list(map(int, kp1[m.queryIdx].pt)) for m in goodmatch]
        img2_pt = [list(map(int, kp2[m.trainIdx].pt)) for m in goodmatch]

        img1_pt = np.array(img1_pt)
        img2_pt = np.array(img2_pt)
        img1_pt = img1_pt*(image_standard.shape[1]/1200)
        img2_pt = img2_pt*(image_standard.shape[1]/1200)
        
        # やるべきこと
        # １。特徴量３つ選ぶ
        # ２。アフィン変換で調整する
        # ３。もとの画像の大きさに戻す
        
        ax1,ax2=int(1323-15),int(1323+15) #だめだったらここの振れ幅調整してみよう
        ay1,ay2=int(83-50),int(83+50)
        bx1,bx2=int(140-50),int(140+50)
        by1,by2=int(120-50),int(120+50)
        cx1,cx2=int(1920-160),int(1920+160)
        cy1,cy2=int(1510-15),int(1510+15)
        dx1,dx2=int(185-55),int(185+55)
        dy1,dy2=int(1485-35),int(1485+35)
        ex1,ex2=int(80-50),int(80+50)
        ey1,ey2=int(1500-60),int(1500+60)

        feature = []
        feature_stock=[]
        for i,j in enumerate(img1_pt):
            if (j[0]>ax1)&(j[0]<ax2):
                if (j[1]>ay1)&(j[1]<ay2):
                    feature.append(i)
                    feature_stock.append(i)
                    break
        for i,j in enumerate(img1_pt):
            if (j[0]>bx1)&(j[0]<bx2):
                if (j[1]>by1)&(j[1]<by2):
                    feature.append(i)
                    feature_stock.append(i)
                    break
        for i,j in enumerate(img1_pt):
            if (j[0]>cx1)&(j[0]<cx2):
                if (j[1]>cy1)&(j[1]<cy2):
                    feature.append(i)
                    feature_stock.append(i)
                    break
        for i,j in enumerate(img1_pt):
            if (j[0]>dx1)&(j[0]<dx2):
                if (j[1]>dy1)&(j[1]<dy2):
                    feature.append(i)
                    feature_stock.append(i)
                    break
        if len(feature)<4:
            for i,j in enumerate(img1_pt):
                if (j[0]>ex1)&(j[0]<ex2):
                    if (j[1]>ey1)&(j[1]<ey2):
                        feature.append(i)
                        feature_stock.append(i)
                        break


        feature1 = []
        for i in feature:
            feature1.append(img1_pt[i])
            feature2 = []
        for i in feature:
            feature2.append(img2_pt[i])
        feature1 = np.array(feature1)
        feature2 = np.array(feature2)
        image = image_test.copy()
        converted = identity1(image,feature1,feature2)
        image=check2013(converted)
        image=identity1(image,feature2,feature1)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(np.uint8(image))
        before[0]=image

        
            #画像を読み込み、pdfファイルに変換
        for j,i in enumerate(before):
                i = i.resize((2338,1654))
                #pdfファイルの保存名を指定
                pdf_name = after_path + "/" +  "PDF_" + str(j) + ".pdf"
                i.save(pdf_name,resolution=200)
        merge = PyPDF2.PdfFileMerger()
        for j in sorted(os.listdir(after_path)):
            if "PDF_" in j :
                merge.append(after_path + "/" + j)
                os.remove(after_path + "/" + j)
        merge.write(after_path + "/" + a)
        merge.close()
