'''
Created on 19 Apr 2016

@author: haifa
'''
import numpy

def grid_pos2D(nz,posi):
    '''grid_pos2D is a function to get the 2D and 3D coordinates from posi '''
    pos2D={}
    pos3D={}
    for n in range(len(posi)/nz):
        pos2D[n]=(posi[n][0],posi[n][1])
    for n in range(len(posi)):
        pos3D[n]=(posi[n][0],posi[n][1]) 
    return pos2D,pos3D

def periodiclist(lx,x1,pbc):

    if(pbc==0):

        dx=numpy.abs(numpy.subtract(range(lx),x1))

    else:

        dx=numpy.abs(numpy.minimum(numpy.minimum(numpy.abs(numpy.subtract(range(lx),x1)),numpy.abs(numpy.subtract(range(lx),lx+x1))),numpy.abs(numpy.subtract(range(lx),-lx+x1))))

    return dx


def edgekernel(lx,ly,v,x1,y1,x2,y2,pbcx,pbcy):
    ''' edgekernel is Edges detection function using Gaussian Kernel  '''
    dx1= periodiclist(lx,x1,pbcx)
    dx2= periodiclist(lx,x2,pbcx)
    dy1= periodiclist(ly,y1,pbcy)
    dy2= periodiclist(ly,y2,pbcy)
    ex1=numpy.ones((ly,1))*dx1**2/(2.0*v)
    ey1=numpy.transpose(numpy.ones((lx,1))*dy1**2/(2.0*v))
    ex2=numpy.ones((ly,1))*dx2**2/(2.0*v)
    ey2=numpy.transpose(numpy.ones((lx,1))*dy2**2/(2.0*v))
    ek=numpy.exp(-numpy.sqrt(ex1+ey1)-numpy.sqrt(ex2+ey2))
    return numpy.divide(ek,numpy.sum(ek))

def most_common (lst):
    return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]

def Generate_Edges_Convs(Depth, cellCoords,im,DisValue,imWidth,imHeight,MinSize,Post,dir_conv):
    dx = float(imWidth) / 2**Depth
    dy = float(imHeight) / 2**Depth
    dz = float(1)/2**Depth # imDepth =1 in 2D images
    
    #################### sorting the positions ######################
    Post=sorted(Post,key=lambda x: (float(x[1]), float(x[0])))

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
    #print Post
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
    
    #print Edges
    ########################### Finding Edges' nodes in the main grid #######################
    connectedNodes=[]
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
    ####################### finding the connected positions ##################
    FPost=[]
   #key1=[]
    #key2=[]
    #n=0
    for key, value in Post.iteritems():
        if key in connectedNodes:
            FPost.append((value[0],value[1],value[2]))
            #key1.append(key)
            #key2.append(n)
            #n+=1
    print "# of Positions=%d"%len(FPost)
    ####################### finding the connected edges ##################
#     FEdges=[]
#     for i in xrange(len(Edges)):
#         newkey1 = Edges[i][0]
#         newkey2 = Edges[i][1]
#         if Edges[i][0]in key1:
#             k = key1.index(Edges[i][0])
#             newkey1 = key2[k]
#         if Edges[i][1]in key1:
#             k = key1.index(Edges[i][1])
#             newkey2 = key2[k]
#         FEdges.append((newkey1,newkey2,1))
    ########################################################################
#     newconn=[]
#     for i in xrange(len(connectedNodes)):
#         if connectedNodes[i] in key1:
#             k = key1.index(connectedNodes[i])
#             newkey = key2[k]
#             newconn.append(newkey)  
#    FinalEdges =[FEdges[i] for i in xrange(len(FEdges))if ((FEdges[i][0] in connectedNodes)or (FEdges[i][1] in connectedNodes))] 
    FinalEdges=[]
    for key1 in range(len(FPost)):
        for key2 in range(key1):
            Dx= numpy.abs(FPost[key1][0]-FPost[key2][0])
            Dy= numpy.abs(FPost[key1][1]-FPost[key2][1])
            Dz= numpy.abs(FPost[key1][2]-FPost[key2][2])
            if((float(Dx)/dx)**2+(float(Dy)/dy)**2+(float(Dz)/dz)**2  < 1.1):
                FinalEdges.append((key1,key2,1))
    E=len(FinalEdges)
    print "# of Edges is %d" % E 

    ########################### Saving the positions #####################
    #numpy.save(dir_posi,Post)
    ######################## calculating the convolution ############################
       
    pos2D,pos3D = grid_pos2D(1,FPost)
    print len(pos3D)
    sh = numpy.shape(im)
    Width,Height = sh[1],sh[0]
    convs=[]
    for e in range(E):
        n=FinalEdges[e][0]
        m=FinalEdges[e][1]
        r=FinalEdges[e][2]
        x1=pos3D[n][0]
        y1=pos3D[n][1]
        x2=pos3D[m][0]
        y2=pos3D[m][1]
        conv= edgekernel(Width,Height,DisValue,x1,y1,x2,y2,0,0)
        numpy.save(dir_conv+'_L='+str(e).zfill(4),conv)
        convs.append(conv)
    return FinalEdges,dir_conv,FPost