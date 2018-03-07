'''
Created on 21 Apr 2016

@author: haifa
'''
import numpy
#cimport numpy
import networkx
import random
import scipy.stats

def help_angle(vec):
    if(vec[0]<0):
        vec=-1*vec
    angle=180.0/numpy.pi*numpy.arccos(vec.dot([0,1])/numpy.sqrt(vec.dot(vec)))
    return angle 

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
    a1 = numpy.sum(numpy.where(angle_angle==0,angle_weight,0))/numpy.sum(numpy.where(angle_angle==0,1,0)) ### DIVISION BY ZERO !!!!!
    data.append(a1)
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