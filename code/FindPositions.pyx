'''
Created on 19 Apr 2016

@author: haifa
'''
import numpy 
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
#         nx=int(dx/2)
#         m = 0
#         for j in range(int(3/h)):  
#             for i in range(3):
#                 posi[m]=((x1+0.5*numpy.mod(h,2))+dx*i,y1+(h+0.5)+dy*h*j,1,Depth,B_thresh1)
#                 m += 1
# #         for i in range(1,3,2):
# #             posi[m]=(x1+nx*i,y1,1,Depth)
# #             m += 1
# #         for j in range(0,3):
# #             posi[m]=(x1+dx*i,y1+dy,1,Depth)
# #             m += 1        
# #         for i in range(1,3,2):
# #             posi[m]=(x1+nx*i,y1+2*dy,1,Depth)
# #             m += 1
#         print "number of positions in mesh shape:%d"%m
#     
#     if(crd=='hexagonal'):
#         nx=int(dx/2)
#         m = 0
#           
#         for i in range(1,4,2):
#             posi[m]=(x1+nx*i,y1,1,Depth,B_thresh1)
#             m += 1
#         for j in range(0,3,3):
#             posi[m]=(x1+dx*i,y1+dy,1,Depth,B_thresh1)
#             m += 1        
#         for i in range(1,4,2):
#             posi[m]=(x1+nx*i,y1+2*dy,1,Depth,B_thresh1)
#             m += 1
        #print "number of positions in mesh shape:%d"%m
        
    
    print "number of positions in Quads:%d"%n
    
    return posi,ThrePos