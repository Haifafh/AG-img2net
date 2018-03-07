###################################################### imports

import copy
import Image
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nwx
import nitime
import nitime.analysis
import nitime.timeseries
import nitime.viz
import numpy as np
import os
import pandas
import paramiko
import pp
import random
import re
import rpy2
import rpy2.robjects
import scipy
import scipy.misc
import scipy.ndimage
import scipy.optimize
import scipy.spatial
import scipy.stats
import skimage
import skimage.filters
import skimage.morphology
import skimage.feature
import sklearn
import subprocess
import sys
import threading
import time
import xml.dom
import matplotlib.path.Path.contains_point 
import mpl_toolkits.mplot3d
import igraph
import PIL

import img2net_help
reload(img2net_help)

import img2net_calc
reload(img2net_calc)

import img2net_AMR
reload(img2net_AMR)
################################################### multiprocessing

import multiprocessing

multiprocessing.freeze_support()

def foo(x):
    return x,x**2    
    
pool=multiprocessing.Pool(2)

N=10
res=[[] for n in range(N)]
for n in range(N):
    o=pool.apply_async(foo,args=(n,)).get()
    res[n]=o[1]
    #np.savetxt('out'+str(o[0]).zfill(4)+'.txt',np.ones(3)*o[1])

sys.exit()

############################################# PIL

#import Image

name='/media/breuer/Numerics/2013_CytoQuant/img2net_TestData/3D/Control/001/Control_e001_t001_z001.tif'

name='/media/breuer/Data/2014_img2net_TestData/4D/t/e/Control_e001_t001_z004.png'

im1=np.array(Image.open(name))
im2=scipy.ndimage.imread(name)

np.shape(np.array(PIL.Image.open(name)))

plt.imshow(im1-im2)
plt.show()

os.path.join(os.environ.get("_MEIPASS2",os.path.abspath(".")),'rel')

################################################## rename 3d files

path='/media/breuer/Data/2013_Cyto_Quant/3D/treatment_nodrug/experiment_001/'
files=sorted(os.listdir(path))

z=4
for fi,f in enumerate(files):
    os.rename(path+f,path+'image_t'+str((fi/z)+1).zfill(3)+'_z'+str(np.mod(fi,z)+1).zfill(3)+'.tif')


########################################### grids

#crd='triangular'#'hexagonal'#'rectangular'#
Binly,Binlx=140,30,1
Mesh = dict()
#dx,dy,dz=5,5,10
#pbx,pby,pbz=0,0,0
#vax,vay=4,4
#N,nx,ny,nz,pos,E,edges,convs=img2net_help.grid_grid(crd,lx,ly,lz,dx,dy,dz,pbx,pby,pbz,vax,vay,0)
Mesh,N,nx,ny,nz,pos,E,Edges,Depth = img2net_AMR.Adaptive_grid(Binlx,Binly,im2,Mesh,6,10,6,0,'','')
pos2D,pos3D=img2net_help.grid_pos2D(N,nz,pos)
gg=nwx.empty_graph(N)
for e in Edges:
    gg.add_edge(e[0],e[1])
    
plt.imshow(np.ones((Binly,Binlx)),origin='lower',extent=[0,Binlx,0,Binly])
nwx.draw_networkx(gg,pos2D,edge_color='red',node_size=0,with_labels=0,width=2)
plt.show()

################################################ boxplot

qr=scipy.randn(100,4)

qr=np.arange(0.0,18.0,1.0)

qr[0:2,:]=np.nan

qr*=np.nan

color='red'
labels=['a','c','d','g']
data=qr

def boxplot(data,color,labels):
    L=len(labels)
    bp=plt.boxplot(data,sym='',notch=0,widths=0.5)
    [plt.setp(bp[k],color=color,ls='-',alpha=1.0,lw=2.0) for k in bp.keys()]
    lims=np.array([bp['whiskers'][l].get_data()[1] for l in range(2*L)])
    plt.xticks(range(1,1+L),labels)
    return lims.min(),lims.max()

boxplot(data,color,labels)
plt.show()




plt.show()

a=[np.hstack(dn[t]) for t in range(T)]

plt.boxplot(a)
plt.show()

######################################## networkx igraph graphtools

import graph_tool
from graph_tool.all import *

N=20
p=0.4
g0=nx.erdos_renyi_graph(N,p)
g1=igraph.Graph.GRG(N,p)
nx.write_graphml(g0,"./img2net_save.xml.gz")
g2=graph_tool.load_graph("./img2net_save.xml.gz")

nx.draw_circular(g0)
plt.show()

g=Graph.TupleList([("a", "b", 3.0), ("c", "d", 4.0), ("a", "c", 5.0)], weights=True)

layout=g1.layout_circle()
igraph.plot(g1,layout=layout)

graph_tool.all.graph_draw(g2,output_size=(1000,1000),output="quant_plot.png")

t0=time.time()
apl0=nx.average_shortest_path_length(g0)
dt0=time.time()-t0
t1=time.time()
apl1=g1.average_path_length()
dt1=time.time()-t1
t2=time.time()
apl2=graph_tool.topology.shortest_distance(g2).get_2d_array(range(N)).sum()*1.0/(N*(N-1.0))
dt2=time.time()-t2
print 'nx',apl0,dt0,'igraph',apl1,dt1,'graphtools',apl2,dt2

#graph_tool.topology.shortest_distance(g, source=None, target=None, weights=None, max_dist=None, directed=None, dense=False, dist_map=None, pred_map=False)
#Calculate the distance from a source to a target vertex, or to of all vertices from a given source, or the all pairs shortest paths, if the source is not specified.

