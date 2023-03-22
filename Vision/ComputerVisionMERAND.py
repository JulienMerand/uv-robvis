from __future__ import print_function
from __future__ import division
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import math


def DisplayImage(path):
    '''
    Charge une image dans une variabe définie comme un tableau NumPy multi-dimensionnel 
    donc le shape est nombre rows (height) x nombre columns (width) x nombre channels (depth)

    Afficher l'image sur l'écran. Attention avec cv2.waitKey(0) vous devrez cliquer dans la fenêtre
    d'affichage et appuyer sur une touche (echap par exemple) pour poursuivre le reste du scirpt
    (ou fermer le script dans le cas présent)
    '''
    
    image = cv2.imread(path)
    (h, w, d) = image.shape
    print("\n")
    print("width={}, height={}, depth={}".format(w, h, d))

    cv2.imshow("Image", image)
    cv2.waitKey(0)


def Analyse_Couleur(path):
    '''
    Affiche 3 images en appliquant respectivement un masque Bleu, Vert et Rouge
    '''
    image = cv2.imread(path)

    # cettte image est resizée afin de réduire la quantité de pixels à traiter
    image = cv2.resize(image,(300,300))

    # changement d'espace colorimétrique
    # ici BGR vers HSV : COLOR_BGR2HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # défintion des contraintes min et max sur les composantes ici seule la teinte
    # est contrainte
    min_green = np.array([50,220,220])
    max_green = np.array([60,255,255])

    min_red = np.array([170,220,220])
    max_red = np.array([180,255,255])

    min_blue = np.array([110,220,220])
    max_blue = np.array([120,255,255])


    # création des masques à partir des limites précédentes
    mask_g = cv2.inRange(hsv, min_green, max_green)
    mask_r = cv2.inRange(hsv, min_red, max_red)
    mask_b = cv2.inRange(hsv, min_blue, max_blue)

    # application des masques sur l'image RGB afin de ne garder que les parties qui
    #nous intéressent.
    res_b = cv2.bitwise_and(image, image, mask= mask_b)
    res_g = cv2.bitwise_and(image,image, mask= mask_g)
    res_r = cv2.bitwise_and(image,image, mask= mask_r)

    # affichage de l'image après sélection de la partie "verte" de l'image
    cv2.imshow('Blue',res_b)
    cv2.imshow('Green',res_g)
    cv2.imshow('Red',res_r)
    cv2.waitKey(0)


def Binarisation(path):
    img_c = cv2.imread(path)
    img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

    # Otsu's thresholding
    ret, thresh8 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret, thresh9 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Par filtrage Adaptatif Gaussien
    img = cv2.medianBlur(img, 5)

    thresh6 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    thresh7 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    # Affichage
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV','ADAPTIVE MEAN THRESHOLDING', 'ADAPTIVE GAUSSIAN THRESHOLDING', "OTSU'S THRESHOLDING", "OTU'S THRESHOLDING AFTER GAUSSIAN FILTERING"]
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7, thresh8, thresh9]
    for i in range(len(images)):
        plt.subplot(3,4,i+1),
        plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


def Crop():
    '''
    Crop un frame de la webcam selon la région selectionne avec la souris
    '''
    # initialisation de la webcam
    cap=cv2.VideoCapture(0)
    
    # capture d'une image
    ret, frame=cap.read()
     
    # sélection d'une régions d'intérêt (ROI) à la souris
    r = cv2.selectROI(frame)
    
    # print les informations la région sélectionnée
    print("coin (x,y) = (",r[1],",",r[0],") - taille (dx,dy) = (",r[2],",",r[3],")")
     
    # image croppée (création de la sous-image sélectionnée)
    imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # affichage de l'image croppée
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)


def Filtrage_HSV():
    
    def souris(event, x, y, flags, param):
        nonlocal lo, hi, color, hsv_px
        
        if event == cv2.EVENT_MOUSEMOVE:
            # Conversion des trois couleurs RGB sous la souris en HSV
            px = frame[y,x]
            px_array = np.uint8([[px]])
            hsv_px = cv2.cvtColor(px_array,cv2.COLOR_BGR2HSV)
        
        if event==cv2.EVENT_LBUTTONDBLCLK:
            color=image[y, x][0]

        if event==cv2.EVENT_LBUTTONDOWN:
            if color>5:
                color-=1

        if event==cv2.EVENT_MBUTTONDOWN:
            if color<250:
                color+=1
                
        lo[0]=color-5
        hi[0]=color+5

    color=100

    lo=np.array([color-5, 100, 50])
    hi=np.array([color+5, 255,255])

    color_info=(0, 0, 255)

    cap=cv2.VideoCapture(0)
    cv2.namedWindow('Camera')
    cv2.setMouseCallback('Camera', souris)
    hsv_px = [0,0,0]

    while True:
        ret, frame=cap.read()
        
        image=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        image = cv2.GaussianBlur(image, (11,11), 0)
        mask=cv2.inRange(image, lo, hi)

        kernel = np.ones((3,3), np.uint8)
        mask=cv2.erode(mask, kernel, iterations=4)
        mask=cv2.dilate(mask, kernel, iterations=4)
        # mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        image2=cv2.bitwise_and(frame, frame, mask= mask)
        cv2.putText(frame, "Couleur: {:d}".format(color), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
        
        # Affichage des composantes HSV sous la souris sur l'image
        pixel_hsv = " ".join(str(values) for values in hsv_px)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "px HSV: "+pixel_hsv, (10, 260),
                font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        mask_cntr = np.zeros(frame.shape, np.uint8)
        
        elements, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(elements) > 0:
            c = sorted(elements, key=cv2.contourArea, reverse=True)
            cnt = c[0]
            ((x_circle,y_circle), rayon) = cv2.minEnclosingCircle(cnt)

            # Rapport d'aspect
            x_aspect, y_aspect, w, h = cv2.boundingRect(c[0])
            aspect_ratio = float(w/h)
            # Valeur min, max et leur pos dans l'image
            # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image, mask=mask)
            # Orientation du grand axe de l'ellipse
            (x_ellipse,y_ellipse),(MA,ma),angle = cv2.fitEllipse(c[0])
            # Couleur et intensité moyenne
            mean_val = cv2.mean(image, mask=mask)

            # print(f"Moyenne : {mean_val} | ratio : {aspect_ratio} | angle : {angle}",)

            cv2.drawContours(frame, [cnt], 0, (0,255,0), 3)
            cv2.drawContours(mask_cntr, [cnt], 0, (0,255,0), 3)
            cv2.circle(frame, (int(x_circle), int(y_circle)), int(rayon), [0,0,255], 2)
            cv2.putText(frame, "Objet !!!", (int(x_circle), int(y_circle)-50), cv2.FONT_HERSHEY_DUPLEX, 1, [0,255,0], 1, cv2.LINE_AA)
                
        cv2.imshow('Camera', frame)
        cv2.imshow('Masque', mask_cntr)
        # cv2.imshow('image2', image2)
        # cv2.imshow('Mask', mask)
        
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def Morph(path):

    img = cv2.imread(path,0)
    img = cv2.resize(img,(450,450))

    # Défintion du kernel de l'érosion taille 5x5
    kernel = np.ones((5,5),np.uint8)

    erosion = cv2.erode(img,kernel,iterations = 1)

    # Défintion du kernel de la dilation taille 3x3
    kernel = np.ones((3,3),np.uint8)

    dilatation = cv2.dilate(img,kernel,iterations = 1)

    # Défintion du kernel de l'ouverture taille 7x7
    kernel = np.ones((7,7),np.uint8)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Défintion du kernel de la fermeture taille 7x7
    kernel = np.ones((7,7),np.uint8)

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    titles = ['Original', 'Erosion', 'Dilatation', 'Closing', 'Opening']
    images = [img, erosion, dilatation, closing, opening]
    for i in range(len(images)):
        plt.subplot(2,3,i+1),
        plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


def Calc_Hist(path):
    img = cv2.imread(path)
    hist = cv2.calcHist([img], [2], None, [256], [0,256])

    # create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    hist_masked = cv2.calcHist([masked_img], [2], None, [256], [10,256])

    cv2.imshow("mask", masked_img)
    cv2.waitKey(0)

    plt.subplot(1,2,1)
    plt.plot(hist, color='b')
    plt.subplot(1,2,2)
    plt.plot(hist_masked, color='b')
    plt.show()


def Reco_Hist():

    src_base = cv2.imread("Intro Vision\RobVis_images_1\main1.jpg")
    src_test1 = cv2.imread("Intro Vision\RobVis_images_1\main2.jpg")
    src_test2 = cv2.imread("Intro Vision\RobVis_images_1\main3.jpg")

    hsv_base = cv2.cvtColor(src_base, cv2.COLOR_BGR2HSV)
    hsv_test1 = cv2.cvtColor(src_test1, cv2.COLOR_BGR2HSV)
    hsv_test2 = cv2.cvtColor(src_test2, cv2.COLOR_BGR2HSV)

    hsv_half_down = hsv_base[hsv_base.shape[0]//2:,:]
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]

    # Hue varie de 0 à 179, la saturation varie de 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges # concat lists

    # Utilise les canaux 0 et 1 pour calculer l'histogramme (H et S)
    channels = [0, 1]

    hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges)
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_half_down = cv2.calcHist([hsv_half_down], channels, None, histSize, ranges)
    cv2.normalize(hist_half_down, hist_half_down, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_test1 = cv2.calcHist([hsv_test1], channels, None, histSize, ranges)
    cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_test2 = cv2.calcHist([hsv_test2], channels, None, histSize, ranges)
    cv2.normalize(hist_test2, hist_test2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    for compare_method in range(4):
        base_base = cv2.compareHist(hist_base, hist_base, compare_method)
        base_half = cv2.compareHist(hist_base, hist_half_down, compare_method)
        base_test1 = cv2.compareHist(hist_base, hist_test1, compare_method)
        base_test2 = cv2.compareHist(hist_base, hist_test2, compare_method)
        
        # affiche les résultats de l'opérateur de comparaison
        print('Method:', compare_method, 'Perfect, Base-Half, Base-Test(1), Base-Test(2) :',\
            base_base, '/', base_half, '/', base_test1, '/', base_test2)


def Reco_TemplateMatching(path, temp):
    
    # chargement d'une image
    img_rgb = cv2.imread(path)

    # conversio en niveau de gris (un seul canal)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # chargement de l'image template à rechercher
    template = cv2.imread(temp,0)
    w, h = template.shape[::-1]

    # seuil de décision qui valide ou non le matching
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)

    # affiche tous les matchs validés sur l'image originale
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
    
    cv2.imshow("Template", cv2.imread(temp))
    cv2.imshow('Detected',img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Reco_Synthese():

    # cap = cv2.VideoCapture(r'Intro Vision\RobVis_images_1\chris2.mp4')
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    
    for i in range(100):
        ret, frame = cap.read()
    if ret:
        # frame = cv2.resize(frame, (0,0), fx=2, fy=2)
        r = cv2.selectROI(frame)
        template_init = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        template = cv2.cvtColor(template_init, cv2.COLOR_BGR2GRAY)
        w, h = template.shape[::-1]
        print("Template enregistré !")
    else :
        print("Erreur")
    cap.release()
    cv2.destroyAllWindows()

    cv2.imshow("Template", template_init)

    # cap = cv2.VideoCapture(r'Intro Vision\RobVis_images_1\chris2.mp4')
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
            
            threshold = 0.7
            loc = np.where( res >= threshold)

            for pt in zip(*loc[::-1]):
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

            cv2.imshow("Video", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()


def Detect_Pixel_Connexes(path):

    img = cv2.imread(path)
    img = cv2.resize(img, (600,600))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    # Si thresh est l'image binarisée
    # extraction des régions et des propriétés des régions
    label_img = label(thresh)
    regions = regionprops(label_img)
    # print(regions)

    # affichage des régions et des boites englobantes
    fig, ax = plt.subplots()
    ax.imshow(thresh, cmap=plt.cm.gray)

    for props in regions:
        xy = props.centroid
        x0, y0 = xy[0], xy[1]
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    ax.axis((0, 600, 600, 0))
    plt.show()

    cv2.waitKey(0)


def FeatureMatching_FAST(path):

    image1 = cv2.imread(path)
    image1 = cv2.resize(image1, (int(image1.shape[1]/2),int(image1.shape[0]/2)))

    image2 = cv2.imread(path)
    image2 = cv2.resize(image2, (int(image2.shape[1]/2),int(image2.shape[0]/2)))

    # l'image doit être tranformée en niveau de gris
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # instanciation de la classe FAST
    fast = cv2.FastFeatureDetector_create() 

    # extraction des keypoints avec non Max Supression
    Keypoints_1 = fast.detect(gray1, None)

    # non Max Suppression desactive
    fast.setNonmaxSuppression(False)

    # Keypoints without non Max Suppression
    Keypoints_2 = fast.detect(gray2, None)

    # création d'une copie de l'image initiale 
    image_with_nonmax = np.copy(image1)
    image_without_nonmax = np.copy(image2)

    # Dessiner les keypoints sur l'image initiale 
    cv2.drawKeypoints(image1, Keypoints_1, image_with_nonmax, color=(0,35,250))
    cv2.drawKeypoints(image2, Keypoints_2, image_without_nonmax, color=(0,35,250))

    cv2.imshow('Non max supression',image_with_nonmax)
    cv2.imshow('Max supression',image_without_nonmax)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def FeatureMatching_BRIEF(path):

    image = cv2.imread(path)
    image = cv2.resize(image, (int(image.shape[1]/2),int(image.shape[0]/2)))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fast = cv2.FastFeatureDetector_create() 

    # instanciation de la classe Brief
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    keypoints = fast.detect(gray, None)    
    # Extraction des descripteurs en chaque keypoint
    brief_keypoints, descriptor = brief.compute(gray, keypoints)

    brief = np.copy(image)
    non_brief = np.copy(image)

    # Draw keypoints on top of the input image
    cv2.drawKeypoints(image, brief_keypoints, brief, color=(0,35,250))
    cv2.drawKeypoints(image, keypoints, non_brief, color=(0,35,250))

    cv2.imshow('Fast corner detection',non_brief)
    cv2.imshow('BRIEF descriptors',brief)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def FeatureMatching_ORB(path):
    
    image = cv2.imread(path)
    image = cv2.resize(image, (int(image.shape[1]/2),int(image.shape[0]/2)))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Instanciation de la classe ORB
    orb = cv2.ORB_create(nfeatures = 1000)

    preview = np.copy(image)
    dots = np.copy(image)

    # Extraction des keypoints et des descripteurs
    keypoints, descriptor = orb.detectAndCompute(gray, None)

    # Dessine les keypoints
    cv2.drawKeypoints(image, keypoints, preview, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(image, keypoints, dots, flags=2)

    cv2.imshow('Points',preview)
    cv2.imshow('Matches',dots)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def FeatureMatching_ORBxBrutForce(train, test, resize=True):
    
    image_1 = cv2.imread(train)
    image_2 = cv2.imread(test) 

    if resize:
        image_1 = cv2.resize(image_1, (int(image_1.shape[1]/2),int(image_1.shape[0]/2)))
        image_2 = cv2.resize(image_2, (int(image_2.shape[1]/2),int(image_2.shape[0]/2)))

    gray_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
    gray_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures = 1000)

    # Copie de l'image originale pour afficher les keypoints
    preview_1 = np.copy(image_1)
    preview_2 = np.copy(image_2)

    # copy d'image_1 pour afficher les points uniquement
    dots = np.copy(image_1)

    train_keypoints, train_descriptor = orb.detectAndCompute(gray_1, None)
    test_keypoints, test_descriptor = orb.detectAndCompute(gray_2, None)

    #Draw the found Keypoints of the main image
    cv2.drawKeypoints(image_1, train_keypoints, preview_1, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(image_2, train_keypoints, dots, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    #############################################
    ########## Mise en correspondance ###########
    #############################################

    # Instanciation the BruteForce Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Lancement du match à partir des deux images
    matches = bf.match(train_descriptor, test_descriptor)

    # Nous trions les meilleurs matchs pour finalement ne garder que les
    # meilleurs matchs (valeur les plus faibles). Ici nous gardons les 100
    # premiers pour notre exemple.

    nb_keypoints = 1000

    matches = sorted(matches, key = lambda x : x.distance)
    good_matches = matches[:nb_keypoints]

    # Récupère les keypoints des good_matches
    train_points = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    test_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Il faut trouver la transformation homographique qui permet de passer
    # de l'ensemble des keypoints de l'image de training à l'ensemble
    # des keypoints de l'image de test afin de dessiner la boite englobant
    # l'objet dans l'image test

    # En utilisant un RANSAC
    M, mask = cv2.findHomography(train_points, test_points, cv2.RANSAC,5.0)

    h,w = gray_1.shape[:2]

    # Création de la matrice à partir du résultat du ransac
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    # Dessine mes point matchés
    dots = cv2.drawMatches(dots,train_keypoints,image_2,test_keypoints,good_matches, None,flags=2)

    # Dessine la boite englobante
    result = cv2.polylines(image_2, [np.int32(dst)], True, (50,0,255),3, cv2.LINE_AA)

    cv2.imshow('Points',preview_1)
    cv2.imshow('Matches',dots)
    cv2.imshow('Detection',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    # DisplayImage(r"Intro Vision\RobVis_images_1\test_img.png")
    # Analyse_Couleur(r"Intro Vision\RobVis_images_1\filtering.png")
    # Binarisation(r"Intro Vision\RobVis_images_1\shape_noise.png")
    # Crop()
    # Filtrage_HSV()
    # Morph(r"Intro Vision\RobVis_images_1\world.png")
    # Calc_Hist(r"Intro Vision\RobVis_images_1\test_img.png")
    # Reco_Hist()
    # Reco_TemplateMatching(r"Intro Vision\RobVis_images_1\roadsign.png", r"Intro Vision\RobVis_images_1\sign_stop.png")
    # Reco_Synthese()
    # Detect_Pixel_Connexes(r"RobVis_images_1\world.png")
    # FeatureMatching_FAST(r"RobVis_image_2\train.jpg")
    # FeatureMatching_BRIEF(r"RobVis_image_2\train.jpg")
    # FeatureMatching_ORB(r"RobVis_image_2\train.jpg")
    FeatureMatching_ORBxBrutForce(r"RobVis_image_2\train.jpg", r"RobVis_image_2\test.jpg")
    # FeatureMatching_ORBxBrutForce(r"RobVis_image_2\sign_stop.png", r"RobVis_image_2\roadsign.png", resize=False)

























































