'''
Created on 21 Apr 2016

@author: haifa
'''
import numpy
import networkx
import scipy.stats
import scipy.ndimage

def grid_graph(im,Edges,pos,dir_convs,dz): #dir_pos,dir_convs
    
    capas=[]
    #pos=numpy.load(dir_pos+'.npy').flatten()[0]
    for e in range(len(Edges)):#loop over n# of edges    
        conv=numpy.load(dir_convs+'_L='+str(e).zfill(4)+'.npy')        
        n = Edges[e][0]
        m = Edges[e][1] 
        z0=int(pos[n][2]/dz)
        z1=int(pos[m][2]/dz)
        
        if(z0==z1):
            #capas.append(numpy.sum(numpy.multiply(im,convs)))
        #else: 
            #print 'z0 =%d is not equal to z1=%d'%(z0,z1)
            #Capas.append(numpy.sum(numpy.multiply(0.5*(im),Q2)))  
            capas.append(numpy.sum(numpy.multiply(im,conv)))

    capas = numpy.divide(capas,numpy.sum(capas))
    print "creating the graph" 
    no=0
    graph=networkx.Graph()
            
    for e in range(len(Edges)):#loop over n# of edges
        n= Edges[e][0]
        m= Edges[e][1]
        graph.add_edge(n,m,capa=capas[e],lgth=1.0/capas[e])
        no+=1
        print "(%f,%f)"%(n,m)
    print "no. of graph edges = %d"%no
    return graph

def grid_all(im,file_path,Edges,pos,dir_convs,dz):  # ,dir_pos,dir_convs, 
    #ims = []
    im=1.0*scipy.ndimage.imread(file_path)
    if(len(numpy.shape(im))>2):
        im=im[:,:,0]
    #ims.append(im)
    graph= grid_graph(im,Edges,pos,dir_convs,dz) #dir_pos,dir_convs
    
    return graph
