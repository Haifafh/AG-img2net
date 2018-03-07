##################################################################### imports
import matplotlib.pyplot
import networkx
import numpy
import random
#import scipy
import scipy.ndimage
#import PIL
#import PIL.Image
import scipy.stats
import img2net_calc
reload(img2net_calc)
##################################################################### functions

def grid_pos2D(nz,posi):
    '''grid_pos2D is a function to get the 2D and 3D coordinates from posi '''
    pos2D={}
    pos3D={}
    for n in range(len(posi)/nz):
        pos2D[n]=(posi[n][0],posi[n][1])
    for n in range(len(posi)):
        pos3D[n]=(posi[n][0],posi[n][1]) 
    return pos2D,pos3D

def help_periodicdistance(lx,x1,x2,pbc):
    if(pbc==0):
        dx=numpy.abs(x1-x2)#absolute value
    else:
        dx=numpy.array([numpy.abs(x1-x2),numpy.abs(x1-x2+lx),numpy.abs(x1-x2-lx)]).min()
    return dx

def help_periodiclist(lx,x1,pbc):
    if(pbc==0):
        dx=numpy.abs(numpy.subtract(range(int(lx)),x1))
    else:
        dx=numpy.abs(numpy.minimum(numpy.minimum(numpy.abs(numpy.subtract(range(lx),x1)),numpy.abs(numpy.subtract(range(lx),lx+x1))),numpy.abs(numpy.subtract(range(lx),-lx+x1))))
    return dx

def help_edgekernel(lx,ly,v,x1,y1,x2,y2,pbcx,pbcy):
    ''' edgekernel is Edges detection function using Gaussian Kernel  '''
    dx1=help_periodiclist(lx,x1,pbcx)
    dx2=help_periodiclist(lx,x2,pbcx)
    dy1=help_periodiclist(ly,y1,pbcy)
    dy2=help_periodiclist(ly,y2,pbcy)
    ex1=numpy.ones((ly,1))*dx1**2/(2.0*v)
    ey1=numpy.transpose(numpy.ones((lx,1))*dy1**2/(2.0*v))
    ex2=numpy.ones((ly,1))*dx2**2/(2.0*v)
    ey2=numpy.transpose(numpy.ones((lx,1))*dy2**2/(2.0*v))
    ek=numpy.exp(-numpy.sqrt(ex1+ey1)-numpy.sqrt(ex2+ey2))
    return numpy.divide(ek,numpy.sum(ek))


def help_angle(vec):
    if(vec[0]<0):
        vec=-1*vec
    angle=180.0/numpy.pi*numpy.arccos(vec.dot([0,1])/numpy.sqrt(vec.dot(vec)))
    return angle

    

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
    graph=grid_graph(im,Edges,pos,dir_convs,dz) #dir_pos,dir_convs
    
    return graph


def graph_null(graphn,nx,ny,numo):
        graphg=graphn.copy()

        if(numo=='edges'):
            E=graphg.number_of_edges()
            idx=range(E)
            random.shuffle(idx) # HERE!!
            for e in range(E):
                graphg.edges(data=1)[e][2]['capa']=graphn.edges(data=1)[idx[e]][2]['capa']
                graphg.edges(data=1)[e][2]['lgth']=graphn.edges(data=1)[idx[e]][2]['lgth']

        elif(numo=='lines'):
            E=graphg.number_of_edges()
            idx=range(nx)
            idy=range(ny)
            random.shuffle(idx)
            random.shuffle(idy)
            links=graphg.edges(data=1)
            for e in range(E):
                [n0,n1]=links[e][0:2]
                [p0x,p0y]=[numpy.mod(n0,nx),n0/nx]
                [p1x,p1y]=[numpy.mod(n1,nx),n1/nx]
                if(p0x==p1x):
                    graphg.edges(data=1)[e][2]['capa']=graphn[idx[p0x]+p0y*nx][idx[p1x]+p1y*nx]['capa']
                    graphg.edges(data=1)[e][2]['lgth']=graphn[idx[p0x]+p0y*nx][idx[p1x]+p1y*nx]['lgth']
                if(p0y==p1y):
                    graphg.edges(data=1)[e][2]['capa']=graphn[p0x+idy[p0y]*nx][p1x+idy[p1y]*nx]['capa']
                    graphg.edges(data=1)[e][2]['lgth']=graphn[p0x+idy[p0y]*nx][p1x+idy[p1y]*nx]['lgth']

        elif(numo=='blocks'):
            Bx,By=3,4
            graphg=graphn.copy()
            E=graphg.number_of_edges()
            N=graphg.number_of_nodes()
            bx=nx/Bx
            by=ny/By
            B=Bx*By
            idb=range(B)
            random.shuffle(idb)
            links=graphg.edges(data=1)
            left=set(range(E))
            for e in range(E):
                [n0,n1]=links[e][0:2]
                [p0x,p0y]=[numpy.mod(n0,nx),n0/nx]
                [p1x,p1y]=[numpy.mod(n1,nx),n1/nx]
                [c0x,c0y]=[p0x/bx,p0y/by]
                [c1x,c1y]=[p1x/bx,p1y/by]
                [c0,c1]=[c0x+c0y*Bx,c1x+c1y*Bx]
                if(c0==c1 and c0<B):
                    left=left.difference([e])
                    [cnewx,cnewy]=[numpy.mod(idb[c0],Bx),idb[c0]/Bx]
                    [d0x,d0y]=[p0x+(cnewx-c0x)*bx,p0y+(cnewy-c0y)*by]
                    [d1x,d1y]=[p1x+(cnewx-c1x)*bx,p1y+(cnewy-c1y)*by]
                    [m0,m1]=[d0x+d0y*nx,d1x+d1y*nx]
                    graphg.edges(data=1)[e][2]['capa']=graphn[m0][m1]['capa']
                    graphg.edges(data=1)[e][2]['lgth']=graphn[m0][m1]['lgth']
            left=list(left)
            ew=[graphg.edges(data=1)[i][2]['capa'] for i in left]
            el=[graphg.edges(data=1)[i][2]['lgth'] for i in left]
            random.shuffle(left)
            for e,ee in enumerate(left):
                graphg.edges(data=1)[ee][2]['capa']=ew[e]
                graphg.edges(data=1)[ee][2]['lgth']=el[e]

        return graphg
    
def boxplot(data,color,labels,ttest):
    
    L=len(labels)
    bp = matplotlib.pyplot.boxplot(data,sym='',notch=0,widths=0.5)
    [matplotlib.pyplot.setp(bp[k],color = color,ls = '-',alpha=1.0,lw=2.0) for k in bp.keys()]
    lims=numpy.array([bp['whiskers'][l].get_data()[1] for l in range(2*L)])  
    matplotlib.pyplot.xticks(range(1,1+L),labels)
    limi,lima=matplotlib.pyplot.ylim()
    limd=lima-limi
    matplotlib.pyplot.ylim([limi,lima+0.1*limd])
    if(ttest==1):
        for i in range(len(labels)):
            #pval=scipy.stats.ttest_1samp(data[i],1)[1]
            pval = 0
            matplotlib.pyplot.text(i+1,lima+0.03*limd,"p=%.2f"%pval,ha='center')
    return
# def barplot(dn,ds1,ds2,ds3,labels,n_groups):
# 
#     dif_untreated=(dn,ds1)
#     dif_treated=(ds2,ds3)
#     index = numpy.arange(n_groups)
#     bar_width = 0.35
#     opacity = 0.4
#     rects1 = matplotlib.pyplot.bar(index,dif_untreated, bar_width,alpha=opacity,color='b',label='untreated')
#     rects2 = matplotlib.pyplot.bar(index + bar_width,dif_treated, bar_width,alpha=opacity,color='r',label='treated')
#     return
def graph_all(G,pos):
    
    data=[]
    label=[]
    N=G.number_of_nodes()

    ################################################# degree
    deg=G.degree(weight='capa').values()

    label.append('mean[degree]')
    data.append(numpy.mean(deg))
    print "mean[degree] =%f" %(numpy.mean(deg))
    print "median[degree] =%f" %(numpy.median(deg))
    label.append('sd[degree]')
    data.append(numpy.std(deg))
    print "sd[degree] =%f" %(numpy.std(deg))
    label.append('skewness[degree]')
    data.append(scipy.stats.skew(deg))
    print "skewness[degree] =%f" %(scipy.stats.skew(deg))
    ################################################# structure

    label.append('clustering coefficient')
    data.append(networkx.average_clustering(G,weight='capa'))
    print "clustering coefficient=%f" %(networkx.average_clustering(G,weight='capa'))
    label.append('assortativity')
    data.append(networkx.degree_pearson_correlation_coefficient(G,weight='capa'))
    print "assortativity =%f" %(networkx.degree_pearson_correlation_coefficient(G,weight='capa'))
    ################################################# distances
    dists=networkx.all_pairs_dijkstra_path_length(G,weight='lgth')
    dist=[[v for v in u.itervalues()] for u in dists.itervalues()]
    ecce=numpy.array(dist).max(0)

    label.append('mean[distance]')
    data.append(numpy.mean(dist))
    print "mean[distance]=%f" %(numpy.mean(dist))
    label.append('sd[distance]')
    data.append(numpy.std(dist))
    print "sd[distance]=%f" %(numpy.std(dist))
    label.append('skewness[distance]')
    data.append(scipy.stats.skew(numpy.reshape(dist,-1)))
    print "skewness[distance]=%f" %(scipy.stats.skew(numpy.reshape(dist,-1)))
    label.append('radius')
    data.append(ecce.min())
    print "radius = %f"%(ecce.min())
    label.append('diameter')
    data.append(ecce.max())
    print "diameter = %f"%(ecce.max())

    ################################################# eigenvalues
    spec=numpy.sort(networkx.laplacian_spectrum(G,weight='capa'))

    label.append('effective resistance')
    data.append(1.0/numpy.sum(numpy.divide(1.0*N,spec[1:N-1])))
    print "effective resistance =%f" %(1.0/numpy.sum(numpy.divide(1.0*N,spec[1:N-1])))
    label.append('algebraic connectivity')
    data.append(spec[1])
    print "algebraic connectivity= %f" %(spec[1])
    ################################################# betweenness
    flow=networkx.edge_current_flow_betweenness_centrality(G,weight='capa',normalized=1).values()

    label.append('mean[betweenness]')
    data.append(numpy.mean(flow))
    print "mean[betweenness]=%f" %(numpy.mean(flow))
    label.append('sd[betweenness]')
    data.append(numpy.std(flow))
    print "sd[betweenness]=%f" %(numpy.std(flow))
    label.append('skewness[betweenness]')
    data.append(scipy.stats.skew(flow))
    print "skewness[betweenness]=%f" %(scipy.stats.skew(flow))

    ################################################# angles
    angle_angle=[]
    angle_weight=[]
    for u,v,d in G.edges_iter(data=1):

        angle_angle.append(help_angle(numpy.subtract(pos[u][0:2],pos[v][0:2])))

        angle_weight.append(d['capa'])
        
    angle_angle=numpy.mod(angle_angle,180)

    label.append('angle 000')
    a1 = numpy.sum(numpy.where(angle_angle==0,angle_weight,0))/numpy.sum(numpy.where(angle_angle==0,1,0))     data.append(a1)
    print"angle 000 = %f" % a1
    
    label.append('angle 045')
    a2 = numpy.sum(numpy.where(angle_angle==45,angle_weight,0))/numpy.sum(numpy.where(angle_angle==45,1,0))
    data.append(a2)
    print"angle 045 = %f" %a2
    
    label.append('angle 060')
    a3 = numpy.sum(numpy.where(angle_angle==60,angle_weight,0))/numpy.sum(numpy.where(angle_angle==60,1,0))
    data.append(a3)
    print"angle 060 = %f" %a3
    
    label.append('angle 090')
    a4 = numpy.sum(numpy.where(angle_angle==90,angle_weight,0))/numpy.sum(numpy.where(angle_angle==90,1,0))
    data.append(a4)
    print"angle 090 = %f" %a4
    
    label.append('angle 120')
    a5 = numpy.sum(numpy.where(angle_angle==120,angle_weight,0))/numpy.sum(numpy.where(angle_angle==120,1,0))
    data.append(a5)
    print"angle 120 = %f" %a5
    
    label.append('angle 135')
    a6 = numpy.sum(numpy.where(angle_angle==135,angle_weight,0))/numpy.sum(numpy.where(angle_angle==135,1,0))
    data.append(a6)
    print"angle 135 = %f" %a6
    
    label.append('angle ratio 00-90')
    a7 = data[-6]/data[-3]
    data.append(a7)
    print"angle ratio 00-90 = %f" %a7

    return data,label