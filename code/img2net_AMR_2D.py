######################################################################## Imports
import numpy 
import matplotlib.pyplot
#from pathos.multiprocessing import ProcessingPool
#from itertools import izip
import os
#import scipy
#from scipy import ndimage
import math
#import sys
import random
#import cv2
import time
#import pp
#from skimage.filter import threshold_li
from skimage.filters import threshold_otsu
#from skimage.filter import threshold_yen
import networkx as nx
#import igraph
#from skimage import measure
#import Image
from nested_dict import nested_dict
#from collections import defaultdict
#from scipy import stats
#from itertools import izip
#from numpy import ones
#from FindPositions import FindPositions
#from BlockThreshold import QudtreeThreshold
#from EdgeConv import Generate_Edges_Convs
#from plotting import plotting
import img2net_help
reload (img2net_help)
#######################################################################  Functions
 
def QudtreeThreshold (im,t,posi,Depth):
      
    Bounds1 = numpy.array([posi[0][0],posi[0][1], posi[4][0] , posi[4][1]])
    #print Bounds1
    imc1 = numpy.array(im.crop(Bounds1.astype(numpy.int_)))
    B1_mean = imc1.mean()
    B1_mean = B1_mean.mean()
    B1_std = numpy. std(imc1)
      
    Bounds2 = numpy.array([posi[1][0] , posi[1][1], posi[5][0] , posi[5][1]])
    #print Bounds2
    imc2 = numpy.array(im.crop(Bounds2.astype(numpy.int_)))
    B2_mean = imc2.mean()
    B2_mean = B2_mean.mean()
    B2_std = numpy. std(imc2)
      
    Bounds3 = numpy.array([posi[3][0] , posi[3][1], posi[7][0] , posi[7][1]])
    #print Bounds3
    imc3 = numpy.array(im.crop(Bounds3.astype(numpy.int_)))
    B3_mean = imc3.mean()
    B3_mean = B3_mean.mean()
    B3_std = numpy. std(imc3)
      
    Bounds4 = numpy.array([posi[4][0] , posi[4][1], posi[8][0] , posi[8][1]])
    #print Bounds4
    imc4 = numpy.array(im.crop(Bounds4.astype(numpy.int_)))
    B4_mean = imc4.mean()
    B4_mean = B4_mean.mean()
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

    #maxThreshold = max([B_thre1 , B_thre2 ,B_thre3 , B_thre4])
    l=min([B1_thresh , B2_thresh , B3_thresh , B4_thresh])
      
    return l#reduce(lambda x, y: x + y, l) / len(l)
def most_common (lst):
    return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]
 
def Generate_Edges_Convs(Depth, cellCoords,im,DisValue,imWidth,imHeight,MinSize,Post,dir_conv):
    dx = float(imWidth) / 2**Depth
    dy = float(imHeight) / 2**Depth
    dz = float(1)/2**Depth # imDepth =1 in 2D images
    # deleting the redundant positions
    #Post=list(set(Pos[Depth-1].values()))
    # TODO: Here we can replace any point if it is found in lower depth.
    # sorting the positions
    Post=sorted(Post,key=lambda x: (float(x[1]), float(x[0])))
     
    print  "# of pos is %d" % len(Post)
    keys=list(xrange(len(Post)))
    ###################################################################
    #testing the accurcey of reordering the points
#     P1= [item[0] for item in Post]
#     P2= [item[1] for item in Post]
#     matplotlib.pyplot.imshow(im, alpha=.3)
#           
#     matplotlib.pyplot.scatter(P1,P2)
#     matplotlib.pyplot.show()
    ######################################################################
    # Generating the Edges
    Post = {k: v for k, v in zip(keys, Post)}
    print len(Post)
    ######################################################################
    # Generating the Edges
    Edges = []
    for key1 in range(len(Post)):
        for key2 in range(key1):
            Dx= numpy.abs(Post[key1][0]-Post[key2][0])
            Dy= numpy.abs(Post[key1][1]-Post[key2][1])
            Dz= numpy.abs(Post[key1][2]-Post[key2][2])
            if((float(Dx)/dx)**2+(float(Dy)/dy)**2+(float(Dz)/dz)**2  < 1.1):
                Edges.append((key1,key2,1))
    print "# of Edges is %d" % len(Edges)
    #####################################################################
    # Generating the Edges
#     Edges=[[(key1,key2,1) if ((float(numpy.abs(Post[key1][0]-Post[key2][0]))/dx)**2+(float(numpy.abs(Post[key1][1]-Post[key2][1]))/dy)**2+(float(numpy.abs(Post[key1][2]-Post[key2][2]))/dz)**2  < 1.1)
#             for key1 in xrange(len(Post))]
#          for key2 in xrange(key1)]
    #Edges = [(key1, [key2 for key2 in xrange(key1) if (float(numpy.abs(Post[key1][0]-Post[key2][0]))/dx)**2+(float(numpy.abs(Post[key1][1]-Post[key2][1]))/dy)**2+(float(numpy.abs(Post[key1][2]-Post[key2][2]))/dz)**2  < 1.1],1) for key1 in xrange(len(Post))]
    #print "# of Edges is %d" % len(Edges)
     
     
     
     
     
     
     
    ########################### Finding the main grid #######################
    connectedNodes=[]
    DEdges=[]
    z1 = most_common(Edges)
    queue=[]
    queue.append(z1[0])
    visited=[]
    n =0
    while (len(queue)>0):
         
        node= queue.pop(0)
        if node not in visited:
            for i in range (len(Edges)):
                     
                if ((node == Edges[i][0])or (node == Edges[i][1])):
                    if (Edges[i][0] not in connectedNodes):
                        connectedNodes.append(Edges[i][0])
                    if (Edges[i][1] not in connectedNodes):
                        connectedNodes.append(Edges[i][1])
                    if (Edges[i][0] not in queue):
                        queue.append(Edges[i][0])
                    if (Edges[i][1] not in queue):
                        queue.append(Edges[i][1])
                         
            visited.append(node)
         
    FinalEdges=[]
    print len(connectedNodes)
#     for i in range(len(Edges)):
#         if ((Edges[i][0] in connectedNodes)or (Edges[i][1] in connectedNodes)):
#             FinalEdges.append(Edges[i])
     
    FinalEdges =[Edges[i] for i in xrange(len(Edges))if ((Edges[i][0] in connectedNodes)or (Edges[i][1] in connectedNodes))]
    print len(FinalEdges)
    ########################### Saving the positions #####################
    #numpy.save(dir_posi,Post) 
    ######################################################################
    E=len(FinalEdges)    
    pos2D,pos3D = img2net_help.grid_pos2D(1,Post)
    sh = numpy.shape(im)
    Width,Height = sh[1],sh[0]
    convs=[]
    for e in range(E):
        n=Edges[e][0]
        m=Edges[e][1]
        r=Edges[e][2]
        x1=pos3D[n][0]
        y1=pos3D[n][1]
        x2=pos3D[m][0]
        y2=pos3D[m][1]
        conv=img2net_help.help_edgekernel(Width,Height,DisValue[Depth-1],x1,y1,x2,y2,0,0)
        numpy.save(dir_conv+'_L='+str(e).zfill(4),conv)
        #if(E<1000):
        convs.append(conv)
    return FinalEdges,dir_conv,Post

def FindPositions(x1,y1,dx,dy,crd,Depth,B_thresh1):
    n=0
    posi={}
    ThrePos={}
    h=numpy.sqrt(3.0)/2.0
    for j in range(3):  
        for i in range(3):
                  
            ThrePos[n] =(int(x1+dx*i),int(y1+dy*j))
            if (crd=='rectangular'): 
                posi[n]=(x1+dx*i,y1+dy*j,1,Depth,B_thresh1)                                
            n += 1
#     if(crd=='triangular'):
        nx=int(dx/2)
        m = 0
        for j in range(int(3/h)):  
            for i in range(3):
                posi[m]=((x1+0.5*numpy.mod(h,2))+dx*i,y1+(h+0.5)+dy*h*j,1,Depth,B_thresh1)
                m += 1
#         for i in range(1,3,2):
#             posi[m]=(x1+nx*i,y1,1,Depth)
#             m += 1
#         for j in range(0,3):
#             posi[m]=(x1+dx*i,y1+dy,1,Depth)
#             m += 1        
#         for i in range(1,3,2):
#             posi[m]=(x1+nx*i,y1+2*dy,1,Depth)
#             m += 1
        print "number of positions in mesh shape:%d"%m
      
    if(crd=='hexagonal'):
        nx=int(dx/2)
        m = 0
            
        for i in range(1,4,2):
            posi[m]=(x1+nx*i,y1,1,Depth,B_thresh1)
            m += 1
        for j in range(0,3,3):
            posi[m]=(x1+dx*i,y1+dy,1,Depth,B_thresh1)
            m += 1        
        for i in range(1,4,2):
            posi[m]=(x1+nx*i,y1+2*dy,1,Depth,B_thresh1)
            m += 1
        print "number of positions in mesh shape:%d"%m
          
      
    print "number of positions in Quads:%d"%n
      
    return posi,ThrePos

def Divide_Decision(Depth,k,cellCoords,v,smin,dmax,im,imWidth,imHeight,crd,D,compute_convs,T):
    ''' Divide_Decision is a function to assign the Bin bounds and check some condition '''    
    # assumes cellCoords is [x, y] with origin (0,0) in top-left
    cellWidth = float(imWidth) / 2**Depth
    cellHeight = float(imHeight) / 2**Depth
    
    dx = float(cellWidth/2)
    dy = float(cellHeight/2) 
    
    x1 = (cellWidth*cellCoords[0])+D#+0.1
    y1 = (cellHeight*cellCoords[1])+D
    x2 = (cellWidth*(cellCoords[0]+1))+D
    y2 = (cellHeight*(cellCoords[1]+1))+D
    
    thisBounds = numpy.array([x1, y1, x2 , y2])
    mind = min(dx,dy)
    minEdgeSize = min(cellWidth,cellHeight)
    maxEdgeSize = max(cellWidth,cellHeight)
    imc = numpy.array(im.crop(thisBounds.astype(numpy.int_)))
   
    if  (minEdgeSize >= smin) : # comparing the boundaries with minimum size of the face
            
            
            if (k==1):      # THRESH_Otsu
                try:
                    B_thresh1 = threshold_otsu(imc)
                except ValueError:
                    B_thresh1 = 0
                        
                        
#             elif (k==2):    # THRESH_TRIANGLE
#                 ret,thresh1 = cv2.threshold(imc,0,255,cv2.THRESH_TRIANGLE) 
#                 B_thresh1= thresh1.mean()
# #             elif (k==3):    # THRESH_yen
# #                 
# #                 B_thresh = threshold_yen(imc)
#             elif (k==4):    # THRESH_BINARY
#                 ret,thresh1 = cv2.threshold(imc,0,255,cv2.THRESH_BINARY) 
#                 B_thresh1= thresh1.mean()
            elif (k==2):    # default Sauvola's thresholding method
                B_thresh = imc.mean()
                B_std= numpy.std(imc)
                B_thresh1 = B_thresh * ( 1 + 0.2 * ( B_std / 128 - 1 ) )
            elif (k==3):    # Niblack thresholding method using (mean)
                B_thresh = imc.mean()
                B_std= numpy.std(imc)
                B_thresh1 = B_thresh + 0.5 * B_std 
                
#             elif (k == 7 ): # MidGrey thresholding method using (mean)
#                 B1_maxmean = max(B_thresh)
#                 B1_minmean = min(B_thresh)
#                 B_thresh1 =((B1_maxmean + B1_minmean ) / 2 )
                
            #########################################################
#             print "computing a Bin's positions in depth %d is\n "% Depth
#             print " cellwidth = %f , cellhight =%f" %(cellWidth,cellHeight)
#             print " coordinates : x = %d , y= %d" % (x1,y1)
#             print " coordinates : x2 = %d , y2= %d" % (x2,y2)
#             print " dx=%d , dy=%d"%(dx,dy)
                        
            # compute all positions at each Bin
            ####################################    
            posi,ThrePos =  (x1,y1,dx,dy,crd,Depth,B_thresh1)
            Pos[Depth][tuple(cellCoords)]= posi 
                             
            if (int(mind) >= smin):
                avgDthreshold = QudtreeThreshold (im,k,ThrePos,Depth)
                    
                AddedValue= ThresholdD[Depth]+ avgDthreshold
                ThresholdD[Depth]= AddedValue
                    
                print "the sub-quads threshoding= %f"%(avgDthreshold)
                return 0
            else:
                if (int(mind) >=2):
                    avgDthreshold = QudtreeThreshold (im,k,ThrePos,Depth)
                    
                    AddedValue= ThresholdD[Depth]+ avgDthreshold
                    ThresholdD[Depth]= AddedValue
                    
                return 2 
                
    else :
        print " This bin is smaller than than the Edge minimum size "
        return 1

def checkCell(Depth,k, cellCoords, v,smin,im,imWidth,imHeight,crd,D,dmax,T):
    ''' checkCell is a recursive function to divide bins and store its info '''
    
    if (Depth > dmax+1):
        print "the grid reaches the maximum depth %s" % Depth
        return
    
    # Dividing the current depth for 4 parts :
    ####################################
    minThreshold = Divide_Decision(Depth,k,cellCoords,v,smin,dmax,im,imWidth,imHeight,crd,D,1,T)
    if (minThreshold  != 1):
        
        cellCoords = numpy.array(cellCoords)    # turning cellCoords from a python list into a numpy array, so we can do element-wise multiplication and addition
        
        Quadtree[Depth].append(tuple(cellCoords))
        if (minThreshold  != 2):
            checkCell(Depth+1,k, cellCoords*2 + numpy.array([0,0]), v,smin,im,imWidth,imHeight,crd,D,dmax,T)
            checkCell(Depth+1,k, cellCoords*2 + numpy.array([0,1]), v,smin,im,imWidth,imHeight,crd,D,dmax,T) 
            checkCell(Depth+1,k, cellCoords*2 + numpy.array([1,0]), v,smin,im,imWidth,imHeight,crd,D,dmax,T)
            checkCell(Depth+1,k, cellCoords*2 + numpy.array([1,1]), v,smin,im,imWidth,imHeight,crd,D,dmax,T) 

    return 

def plotting(im,Depth,pos,Edges,dir_output,imWidth,imHeight):
    matplotlib.pyplot.clf()
    matplotlib.pyplot.imshow(im, alpha=.3)
    G =len(Edges)
     
    for g in range(G):#loop over n# of edges
        u = Edges[g][0]
        v = Edges[g][1]
     
        if(pos[u]!=pos[v]):
            alp=1.0
        else:
            alp=0.4
        colors = matplotlib.pyplot.get_cmap('jet')
        matplotlib.pyplot.plot([pos[u][0],pos[v][0]],[pos[u][1],pos[v][1]],color=colors(0),alpha=alp,lw=2) #(pos[v][3])*100
                 
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.xlim(0,imWidth)
    matplotlib.pyplot.ylim(0,imHeight)
    mng = matplotlib.pyplot.get_current_fig_manager()
    mng.resize(*mng.window.maxsize()) 
    matplotlib.pyplot.savefig(os.path.join(dir_output,'plot_grid','plot_grid.pdf'), bbox_inches=0,dpi=1000)
    matplotlib.pyplot.show()
    #im.show()
    return None


###################################################################################### Main 



def Adaptive_grid(imWidth,imHeight,crd,im,dir_conv,dir_Edges,dir_QuadTree,dir_output):
    #job_server = pp.Server()

    temp=time.time()
    #print "Starting pp with", job_server.get_ncpus(), "workers"
    
    print "image width = %d, image height = %d\n" % (imWidth, imHeight)
    k = 2
    
    smin = 2.2 # thresholding works on 2px and more.
    if (smin<2):
        print " The minimum block size can not be less than 2 pixels"
        return
    D = int(math.ceil(numpy.log(float(min(imWidth, imHeight))/smin)+float(3/2)))+1
    # because depth start from 0 in Python
    W = imWidth-(D*2) 
    H = imHeight-(D*2)
    print "imWidth of the grid = %d , imHeight of the grid = %d" %(W,H)
    #smax = max(W,H)
    dmax = int(math.ceil(numpy.log(float(min(W, H))/smin)+float(3/2)))
    #v = range(D+1,1,-1) #size of sigma
    if (smin<=16):
        v= 0.5*smin
    else:
        v= 8
    print "smin = %d ,dmax = %d, Distance =%d" %(smin,dmax,D)
    print "image width = %d, image height = %d\n" % (W, H)
    print v
    Quadtree = nested_dict(1,list)
    Pos= nested_dict()
    Posi= nested_dict()
    Position= nested_dict()
    ThresholdD ={}
    for i in range(dmax):
        ThresholdD[i] = 0
    print ThresholdD
    global smin
    global Pos
    global Quadtree     # this allows us to modify as well as read Quadtree
    global ThresholdD
    NewBounds = numpy.array([0+(D), 0+(D), 0+(D)+W, 0+(D)+H])
    img = numpy.array(im.crop(NewBounds.astype(numpy.int_)))
    print NewBounds
    ##############################################################
    if (k==1):      # THRESH_Otsu***
        Glabal_thresh = threshold_otsu(img)
                        
#     elif (k==2):    # THRESH_TRIANGLE
#         
#         ret,thresh1 = cv2.threshold(img,128,255,cv2.THRESH_TRIANGLE) 
#         Glabal_thresh= thresh1.mean()
#         
# #     elif (k==3):    # THRESH_yen
# #                 
# #         Glabal_thresh = threshold_yen(img)        
#     elif (k==4):    # THRESH_BINARY
#         ret,thresh1 = cv2.threshold(img,128,255,cv2.THRESH_BINARY) 
#         Glabal_thresh = thresh1.mean()
        
    elif (k==2):    # default Sauvola's thresholding method***
        img_thresh = img.mean()
        img_thresh = img_thresh.mean()
        img_std= numpy.std(img)         
        Glabal_thresh = img_thresh * ( 1 + 0.2 * ( img_std / 128 - 1 ) )
        
    elif (k==3):    # Niblack thresholding method using (mean)
        img_thresh = img.mean()
        img_thresh = img_thresh.mean()
        img_std= numpy.std(img)    
        Glabal_thresh = img_thresh + 0.5 * img_std

    global Glabal_thresh                
    print Glabal_thresh
    ####################################################################
    
    checkCell(0,k, [0,0],v,smin,im,W,H,crd,D,dmax,0)
    print "the time consumed to built the grid and multilevel thresholding is %f"%(time.time()-temp)
    Position = Pos
    POsition= Position.to_dict()
    Depth= len(POsition.keys())
    print len(POsition[Depth-1].values())
    print "Depths =%d" %Depth
    print (len(Pos[Depth-1].items()))
    
    ############ weighting the average thresholding ####################
    for i in range(1,len(ThresholdD)):
        if ThresholdD[i]!=0:
            ThresholdD[i]=float(ThresholdD[i])/len(Quadtree[i])
        if (i!=0):
            ThresholdD[i]=float(ThresholdD[i]+Glabal_thresh)/2
            #ThresholdD[i]=float(ThresholdD[i]+ThresholdD[i-1])/2
    print ThresholdD
    print "Depth  Average thresholding number of quadarnts "
    for i in range(len(ThresholdD)):
        print " %d         %f           %f "%(i,ThresholdD[i],len(Quadtree[i]))
        
    ################# refining the quadtree ##############################
    Post=[]
    for i in xrange(len(ThresholdD)+1):
        for j in xrange(len(Quadtree[i])):
            if ((smin==2)and(i==Depth-1)):
                keys=i-1
            else:
                keys=i
            if (Pos[i][Quadtree[i][j]][0][4] >= ThresholdD[keys]**2/Glabal_thresh):
                pp={}
                for m in range(9):
                    pp[m]=Pos[i][Quadtree[i][j]][m]
                    Post.append(Pos[i][Quadtree[i][j]][m])
                    
                Posi[i][Quadtree[i][j]]=pp
    NodubPost = []
    seen = set()
    for (a, b,c,d,e) in Post:
        if not (a,b) in seen:
            NodubPost.append((a, b,c,d))
            seen.add((a,b))
    #print len(NodubPost)      
    #print (len(Pos[Depth-1].items()))
    #print len(Post)
    #print (len(Posi[Depth-1].items()))
    #Position= Posi.to_dict()
    ######################################################################
    
    #print "the time consumed to built the grid and multilevel thresholding is %f"%(time.time()-temp)
    #################printing the positions##############################
    P=[]
    S=[]
# #     for key, value in Posi[Depth-1].items_flat():
# # 
# #         P.append(value[0])
# #         S.append(value[1])
    for i in xrange(len(NodubPost)):
 
        P.append(NodubPost[i][0])
        S.append(NodubPost[i][1])        
    matplotlib.pyplot.imshow(im, alpha=.3)
              
    matplotlib.pyplot.scatter(P,S)
    matplotlib.pyplot.show()
    ######################################################################
#     Post =[]
#     
#     for j in xrange(len(Quadtree[Depth-1])):
#         for i in xrange(9):
#             Post.append(Position[Depth-1][Quadtree[Depth-1][j]][i])
#     print Post
#     Postt=set(Post)
#     print Postt
    #Post1=set(Postt)
    ########################################################################
    # Generate_Edges_Convs: this function can not work with big images.
    temp1=time.time()
    Edges,dir_conv,pos = Generate_Edges_Convs(Depth, [0,0],im,v,W,H,smin,NodubPost,dir_conv)
    print "the time consumed is %f"%(time.time()-temp1)
    Posi = Pos.to_dict()    #convert all nested dicts to python dicts
#    QuadTree = Quadtree.to_dict()
    global Posi
    global EDeges
    global CONVS
    #global QuadTree 
    
    ################## plotting ##############################################
    plotting(im,Depth,pos,Edges,dir_output,imWidth,imHeight)
    
    print 'The adaptive grid is built !'
    return Edges,dir_conv,pos



def random_network(dir_input):
    # Create the randomized generated graph with random weights
    G = nx.Graph()
    n = random.randrange(1,4)
    p = random.randrange(5,8)
    for n in range (n):
        for p in range (p):
            G.add_edge(n,p,weight=random.randrange(1,15))
    # use one of the edge properties to control line thickness
    edgewidth = [ d['weight'] for (u,v,d) in G.edges(data=True)]
    # layout
    pos = nx.spring_layout(G)
    # rendering
    fig=matplotlib.pyplot.figure(1,facecolor='black',figsize=(0.25,2))
    matplotlib.pyplot.jet()
    matplotlib.pyplot.axis('off')
    nx.draw_networkx_edges(G, pos, width=edgewidth,edge_cmap=matplotlib.pyplot.get_cmap('gray'),edge_color=edgewidth)
    matplotlib.pyplot.show(block=False) 
    # save the figure
    matplotlib.pyplot.savefig(os.path.join(dir_input,'Rnetwork.tif'), bbox_inches=0,dpi=1000)
    return
