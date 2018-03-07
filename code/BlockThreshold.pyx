'''
Created on 19 Apr 2016

@author: haifa
'''
import numpy
from skimage.filter import threshold_otsu

def QudtreeThreshold (im,t,posi,Depth):
    
    Bounds1 = numpy.array([posi[0][0],posi[0][1], posi[4][0] , posi[4][1]])
    #print Bounds1
    imc1 = numpy.array(im.crop(Bounds1.astype(numpy.int_)))
    B1_mean = imc1.mean()
    B1_mean = B1_mean.mean()
#    B1_maxmean = B1_mean.max()
#    B1_minmean = B1_mean.min()
    B1_std = numpy. std(imc1)
    
    Bounds2 = numpy.array([posi[1][0] , posi[1][1], posi[5][0] , posi[5][1]])
    #print Bounds2
    imc2 = numpy.array(im.crop(Bounds2.astype(numpy.int_)))
    B2_mean = imc2.mean()
    B2_mean = B2_mean.mean()
#    B2_maxmean = B2_mean.max()
#    B2_minmean = B2_mean.min()
    B2_std = numpy. std(imc2)
    
    Bounds3 = numpy.array([posi[3][0] , posi[3][1], posi[7][0] , posi[7][1]])
    #print Bounds3
    imc3 = numpy.array(im.crop(Bounds3.astype(numpy.int_)))
    B3_mean = imc3.mean()
    B3_mean = B3_mean.mean()
#    B3_maxmean = B3_mean.max()
#    B3_minmean = B3_mean.min()
    B3_std = numpy. std(imc3)
    
    Bounds4 = numpy.array([posi[4][0] , posi[4][1], posi[8][0] , posi[8][1]])
    #print Bounds4
    imc4 = numpy.array(im.crop(Bounds4.astype(numpy.int_)))
    B4_mean = imc4.mean()
    B4_mean = B4_mean.mean()
#    B4_maxmean = B4_mean.max()
#    B4_minmean = B4_mean.min()
    B4_std = numpy. std(imc4)
    
    if (t == 1 ): # Otsu's thresholding method

        try:   
            B1_thresh = (threshold_otsu(imc1))
        except ValueError:
            B1_thresh = 0
        try:   
            B2_thresh = (threshold_otsu(imc2))
        except ValueError:
            B2_thresh = 0   
        try:   
            B3_thresh = (threshold_otsu(imc3))
        except ValueError:
            B3_thresh = 0
        try:   
            B4_thresh = (threshold_otsu(imc4))
        except ValueError:
            B4_thresh = 0
              
#     elif (t == 2):    # THRESH_TRIANGLE
#         ret,thresh1 = cv2.threshold(imc1,0,255,cv2.THRESH_TRIANGLE) 
#         B1_thresh= thresh1.mean()
#         ret,thresh1 = cv2.threshold(imc2,0,255,cv2.THRESH_TRIANGLE) 
#         B2_thresh= thresh1.mean()
#         ret,thresh1 = cv2.threshold(imc3,0,255,cv2.THRESH_TRIANGLE) 
#         B3_thresh= thresh1.mean()
#         ret,thresh1 = cv2.threshold(imc4,0,255,cv2.THRESH_TRIANGLE) 
#         B4_thresh= thresh1.mean()
# #     elif (t == 3):    # THRESH_yen
# #                     
# #         B1_thresh = threshold_yen(imc1)
# #         B2_thresh = threshold_yen(imc2)
# #         B3_thresh = threshold_yen(imc3)
# #         B4_thresh = threshold_yen(imc4)
# #         
#     elif (t == 4):    # THRESH_BINARY
#         ret,thresh1 = cv2.threshold(imc1,0,255,cv2.THRESH_BINARY) 
#         B1_thresh= thresh1.mean()
#         ret,thresh1 = cv2.threshold(imc2,0,255,cv2.THRESH_BINARY) 
#         B2_thresh= thresh1.mean()
#         ret,thresh1 = cv2.threshold(imc3,0,255,cv2.THRESH_BINARY) 
#         B3_thresh= thresh1.mean()
#         ret,thresh1 = cv2.threshold(imc4,0,255,cv2.THRESH_BINARY) 
#         B4_thresh= thresh1.mean()
        
    elif (t == 2):    # default Sauvola's thresholding method
                     
        B1_thresh = B1_mean * ( 1 + 0.2 * ( B1_std / 128 - 1 ) )
        B2_thresh = B2_mean * ( 1 + 0.2 * ( B2_std / 128 - 1 ) )
        B3_thresh = B3_mean * ( 1 + 0.2 * ( B3_std / 128 - 1 ) )
        B4_thresh = B4_mean * ( 1 + 0.2 * ( B4_std / 128 - 1 ) )
                
    elif (t == 3):    # Niblack thresholding method using (mean)
        
        B1_thresh = B1_mean + 0.5 * B1_std 
        B2_thresh = B2_mean + 0.5 * B2_std
        B3_thresh = B3_mean + 0.5 * B3_std 
        B4_thresh = B4_mean + 0.5 * B4_std 
        
#     elif (t == 7 ): # MidGrey thresholding method using (mean)
#         B_thre1 =((B1_maxmean + B1_minmean ) / 2 )
#         B_thre2 =((B2_maxmean + B2_minmean ) / 2 )
#         B_thre3 =((B3_maxmean + B3_minmean ) / 2 ) 
#         B_thre4 =((B4_maxmean + B4_minmean ) / 2 ) 
    
#     elif (t == 6): # variance
#         B_thre1 = (imc1 - imc1.mean())**2 
#         B_thre1 = B_thre1.mean()
#         B_thre2 = (imc2 - imc2.mean())**2 
#         B_thre2 = B_thre2.mean()
#         B_thre3 = (imc3 - imc3.mean())**2 
#         B_thre3 = B_thre3.mean()
#         B_thre4 = (imc4 - imc4.mean())**2 
#         B_thre4 = B_thre4.mean()
#     elif ( t == 7): # Phansalkar thresholding method
#         B_thre1 = B1_mean * (1 + 2 * math.exp(-10 * B1_mean) + 1 * ((B1_std / 2) - 1))
#         B_thre2 = B2_mean * (1 + 2 * math.exp(-10 * B2_mean) + 1 * ((B2_std / 2 ) - 1)) 
#         B_thre3 = B3_mean * (1 + 2 * math.exp(-10 * B3_mean) + 1 * ((B3_std / 2 ) - 1)) 
#         B_thre4 = B4_mean * (1 + 2 * math.exp(-10 * B4_mean) + 1 * ((B4_std / 2 ) - 1))   
    #maxThreshold = max([B_thre1 , B_thre2 ,B_thre3 , B_thre4])
    l=min([B1_thresh , B2_thresh , B3_thresh , B4_thresh])
    
    return l#reduce(lambda x, y: x + y, l) / len(l)