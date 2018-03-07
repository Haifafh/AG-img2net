##################################################################### imports
import matplotlib
import os
import matplotlib.pyplot
import numpy
import scipy
#import cv2
import scipy.ndimage
#from skimage import exposure
import networkx
#import Image
import PIL.ImageOps
import math
import time
import PIL
from PIL import Image
#from graph import graph_all
#from grid import grid_all
import img2net_help
reload(img2net_help)
import img2net_AMR_2D
reload(img2net_AMR_2D)
import img2net_AMR_2D_condition
reload(img2net_AMR_2D_condition)
###################################################################### img2net

def img2net_calc(crd,dir_input,file_input):
    
    
    name = 'Adaptive_grid_'
    
        
    dir_output = dir_input+'Output_'+name+file_input+'/'
    #dir_output1 = dir_input+'Output_'+'img2net_treated_'+file_input+'/'
    subfolders = ['data_posi','data_conv','data_grph','data_datn','data_prop','data_readable','plot_grid']
    file_path = os.path.join(dir_input,file_input)
    ############################################# generate folders and check: overwrite?

    if(not os.path.isdir(dir_output)):
        os.makedirs(dir_output)
            
    for subfolder in subfolders:  
        if(not os.path.isdir(dir_output+subfolder)):
                os.makedirs(dir_output+subfolder)     
                
    if(os.path.isfile(os.path.join(dir_output,'data_readable','data_readable.txt'))):
        os.remove(os.path.join(dir_output,'data_readable','data_readable.txt'))

    
    ################################################################# img2net_AMR
#    img2net_AMR2.random_network(dir_input)
    
    print 'img2net_AMR'
    
    im = Image.open(file_path).convert('L')
    #im = PIL.ImageOps.invert(im)
    
    #im=scipy.ndimage.filters.median_filter(im,30)
    #im=scipy.ndimage.imread(file_path) # replaced by Image.open.
    ########################################## main parameters   
    AMR =1
    ##########################################
    sh = numpy.shape(im)
    imWidth,imHeight = sh[1],sh[0]
    
    ################################################################## features
    
    dir_conv = os.path.join(dir_output,'data_conv','data_conv')
    dir_Edges = os.path.join(dir_output,'data_Edges','data_Edges')
    dir_QuadTree = os.path.join(dir_output,'data_QuadTree','data_QuadTree')
    
    #iim=1.0*scipy.ndimage.imread(file_path)
    if(len(numpy.shape(im))>2):
        print "3D Adaptive_grid .."
        #Edges,dir_conv,pos = img2net_AMR_3D.Adaptive_grid(imWidth,imHeight,crd,im,dir_conv,dir_Edges,dir_QuadTree,dir_output)
    else:
        print "2D Adaptive_grid .."
        Edges,dir_conv,pos = img2net_AMR_2D.Adaptive_grid(imWidth,imHeight,crd,im,dir_conv,dir_Edges,dir_QuadTree,dir_output)
        #Edges,dir_conv,pos = img2net_AMR_2D_condition.Adaptive_grid(imWidth,imHeight,crd,im,dir_conv,dir_Edges,dir_QuadTree,dir_output)
    ################################################################# graph
    
    print 'graph'
    temp1=time.time()
    graph = img2net_help.grid_all(im,file_path,Edges,pos,dir_conv,1) #dz = 1
    numpy.save(os.path.join(dir_output,'data_grph','data_grph.npy'),networkx.to_edgelist(graph))
    print "the time consumed to build the graph is %f"%(time.time()-temp1)
    
    ################################################################# observed network properties
    
    print 'obs network'
    temp2=time.time()
    data,label= img2net_help.graph_all(graph,pos)
    print "the time consumed to measure the graph's properties quantitatively is %f"%(time.time()-temp2)

    numpy.save(os.path.join(dir_output,'data_datn','data_datn.npy'),data)

    numpy.save(os.path.join(dir_output,'data_prop','data_prop.npy'),label)
    ################################################################# save data

    labels=numpy.load(os.path.join(dir_output,'data_prop','data_prop.npy'))
 
    with open(os.path.join(dir_output,'data_readable','data_readable.txt'),"a") as out:
        out.write('\t'.join([str(a) for a in numpy.hstack(labels)]))
        out.write('\n')
        datn=numpy.load(os.path.join(dir_output,'data_datn','data_datn.npy'))
        out.write('\t'.join([str(a) for a in numpy.hstack(datn)]))
        out.write('\n')
        
 
    ################################################################# plot boxplots
#     matplotlib.rcParams.update({'font.size': 15})
#     
#     s+=1
#     for l,label in enumerate(labels):
#             
#         print 'label',l+1,len(labels)
#         print labels[l]
#         dn=[]
#         dd=[]
#         datn=numpy.load(os.path.join(dir_output,'data_datn','data_datn.npy'))
#         dn.append(datn[l])
#         for r in range(R):
#             datr=numpy.load(os.path.join(dir_output,'data_datr','data_datr_R='+str(r).zfill(4)+'.npy'))
#             dd.append(1.0*datn[l]/datr[l])
#                
#                
#         ds=[]
#         dp=[]
#         datn1=numpy.load(os.path.join(dir_output1,'data_datn','data_datn.npy'))
#         ds.append(datn1[l])
#         for r in range(R):
#             datr1=numpy.load(os.path.join(dir_output1,'data_datr','data_datr_R='+str(r).zfill(4)+'.npy'))
#             dp.append(1.0*datn1[l]/datr1[l])
#             
#         print '--------------------------------------------------------'
#         print dn
#         print dd
#         print ds
#         print dp
#         matplotlib.pyplot.clf()
#         matplotlib.pyplot.subplot(221)
#         matplotlib.pyplot.suptitle(label)
#         img2net_help.boxplot(dn,'black',[1],0)
#         matplotlib.pyplot.ylabel('absolute')
#         matplotlib.pyplot.xlabel('AMR img2net')
#             
#         matplotlib.pyplot.subplot(222)
#         matplotlib.pyplot.plot([0,1],[1,1],lw=2,color='gray',ls='--')
#         img2net_help.boxplot(dd,'black',[1],1)
#         matplotlib.pyplot.ylabel('relative')
#         matplotlib.pyplot.xlabel('null AMR img2net')
#            
#         matplotlib.pyplot.subplot(223)
#         matplotlib.pyplot.suptitle(label)
#         img2net_help.boxplot(ds,'black',[1],0)
#         matplotlib.pyplot.ylabel('absolute')
#         matplotlib.pyplot.xlabel('img2net')
#            
#         matplotlib.pyplot.subplot(224)
#         matplotlib.pyplot.plot([0,1],[1,1],lw=2,color='gray',ls='--')
#         img2net_help.boxplot(dp,'black',[1],1)
#         matplotlib.pyplot.ylabel('relative')
#         matplotlib.pyplot.xlabel('null img2net')
#            
# #         matplotlib.pyplot.subplot(325)
# #         matplotlib.pyplot.suptitle(label)
# #         img2net_help.boxplot(dn,'black',[1],0)####
# #         matplotlib.pyplot.ylabel('absolute')
# #         matplotlib.pyplot.xlabel('img2net')
# #          
# #         matplotlib.pyplot.subplot(326)
# #         matplotlib.pyplot.plot([0,1],[1,1],lw=2,color='gray',ls='--')
# #         img2net_help.boxplot(ds,'black',[1],1)
# #         matplotlib.pyplot.ylabel('obs')
# #         matplotlib.pyplot.xlabel('AMR img2net')
# #          
#         matplotlib.pyplot.tight_layout()
#         matplotlib.pyplot.subplots_adjust(top=0.92)
#         matplotlib.pyplot.savefig(os.path.join(dir_output,'plot_dist','plot_dist_p='+str(l).zfill(4)+'_'+label+'.png'))
#         matplotlib.pyplot.show()
# #    
             
#     matplotlib.rcParams.update({'font.size': 15})
#     
#     n_groups =2
#     
#     for l,label in enumerate(labels):
#             
#         print 'label',l+1,len(labels)
#         print labels[l]
#         ################## AMR results########
#         dn=0
#         dd=[]
#         datn=numpy.load(os.path.join(dir_output,'data_datn','data_datn.npy'))
#         dn.append(datn[l])
#         for r in range(R):
#             datr=numpy.load(os.path.join(dir_output,'data_datr','data_datr_R='+str(r).zfill(4)+'.npy'))
#             dd.append(1.0*datn[l]/datr[l])
#                
#         ################## AMR img2net results########    
#         ds1=0
#         dp1=[]
#         datn1=numpy.load(os.path.join(dir_output1,'data_datn','data_datn.npy')) ##
#         ds1.append(datn1[l])
#         for r in range(R):
#             datr1=numpy.load(os.path.join(dir_output1,'data_datr','data_datr_R='+str(r).zfill(4)+'.npy'))
#             dp1.append(1.0*datn1[l]/datr1[l])
#         ################# AMR img2net treated results #######
#         ds2=0
#         dp2=[]
#         datn2=numpy.load(os.path.join(dir_output2,'data_datn','data_datn.npy'))
#         ds2.append(datn1[l])
#         for r in range(R):
#             datr2=numpy.load(os.path.join(dir_output2,'data_datr','data_datr_R='+str(r).zfill(4)+'.npy'))
#             dp2.append(1.0*datn2[l]/datr2[l])
#         ################ img2net treated results ##########
#         ds3=0
#         dp3=[]
#         datn3=numpy.load(os.path.join(dir_output3,'data_datn','data_datn.npy'))
#         ds3.append(datn1[l])
#         for r in range(R):
#             datr3=numpy.load(os.path.join(dir_output3,'data_datr','data_datr_R='+str(r).zfill(4)+'.npy'))
#             dp3.append(1.0*datn3[l]/datr3[l])
#         #################################################### 
#         print '--------------------------------------------------------'
#         print dn # abs value of AMR img2net
#         print dd # null value of AMR img2net
#         print ds1 # abs value of img2net
#         print dp1 # null value of img2net
#         print ds2 # abs value of treated AMR img2net
#         print dp2 # null value of treated AMR img2net
#         print ds3 # abs value of treated img2net
#         print dp3 # null value of treated img2net
#         matplotlib.pyplot.clf()
#         matplotlib.pyplot.subplot(321)
#         matplotlib.pyplot.suptitle(label)
#         img2net_help.boxplot(dn,'black',[1],0)
#         matplotlib.pyplot.ylabel('absolute')
#         matplotlib.pyplot.xlabel('AMR img2net')
#             
#         matplotlib.pyplot.subplot(322)
#         matplotlib.pyplot.plot([0,1],[1,1],lw=2,color='gray',ls='--')
#         img2net_help.boxplot(dd,'black',[1],1)
#         matplotlib.pyplot.ylabel('relative')
#         matplotlib.pyplot.xlabel('null AMR img2net')
#            
#         matplotlib.pyplot.subplot(323)
#         matplotlib.pyplot.suptitle(label)
#         img2net_help.boxplot(ds1,'black',[1],0)
#         matplotlib.pyplot.ylabel('absolute')
#         matplotlib.pyplot.xlabel('img2net')
#            
#         matplotlib.pyplot.subplot(324)
#         matplotlib.pyplot.plot([0,1],[1,1],lw=2,color='gray',ls='--')
#         img2net_help.boxplot(dp1,'black',[1],1)
#         matplotlib.pyplot.ylabel('relative')
#         matplotlib.pyplot.xlabel('null img2net')
#            
#                                 #         matplotlib.pyplot.subplot(325)
#                                 #         matplotlib.pyplot.suptitle('comparing'+label+'between treated and untreated')
#                                 #         img2net_help.boxplot(dn,'black',[1],0)####
#                                 #         #matplotlib.pyplot.ylabel('absolute')
#                                 #         matplotlib.pyplot.xlabel('AMR img2net')
#                                 #          
#                                 #         matplotlib.pyplot.subplot(326)
#                                 #         matplotlib.pyplot.plot([0,1],[1,1],lw=2,color='gray',ls='--')
#                                 #         img2net_help.boxplot(ds,'black',[1],1)
#                                 #         #matplotlib.pyplot.ylabel('obs')
#                                 #         matplotlib.pyplot.xlabel('img2net')
#         matplotlib.pyplot.subplot(325)
#         matplotlib.pyplot.suptitle('comparing'+label+'between treated and untreated')
#         dif_untreated=(1,2)
#         dif_treated=(3,4)
#         index = numpy.arange(n_groups)
#         bar_width = 0.35
#         opacity = 0.4
#         rects1 = matplotlib.pyplot.bar(index,dif_untreated, bar_width,alpha=opacity,color='b',label='untreated')
#         rects2 = matplotlib.pyplot.bar(index + bar_width,dif_treated, bar_width,alpha=opacity,color='r',label='treated')
#         matplotlib.pyplot.xlabel('Group')
#         matplotlib.pyplot.ylabel('Values')
#         #matplotlib.pyplot.xticks(index + bar_width, ('AMR img2net', 'img2net'))
#   
#         matplotlib.pyplot.tight_layout()
#         matplotlib.pyplot.subplots_adjust(top=0.92)
#         matplotlib.pyplot.savefig(os.path.join(dir_output,'plot_dist','plot_dist_p='+str(l).zfill(4)+'_'+label+'.png'))
#         matplotlib.pyplot.show()
#    
        ################################################################# plot times series
 
#         matplotlib.pyplot.clf()
#         #T2=numpy.where(T>2,T,2).min()
#         #use=0.40/T
#         #gap=0.60/(T2-1.0)
#         data=numpy.load(os.path.join(dir_output,'data_datn','data_datn.npy'))[l]
#         #sec=t*use+t*gap+e*use/(E2-1.0)
#         matplotlib.pyplot.title(label)
#         #matplotlib.pyplot.plot(data,lw=2,color=matplotlib.pyplot.cm.jet(sec),label=treatments[t]+','+experiments[t][e])
#         matplotlib.pyplot.xlabel('frame')
#         matplotlib.pyplot.ylabel('absolute')
#         #matplotlib.pyplot.legend()
#         matplotlib.pyplot.tight_layout()
#         matplotlib.pyplot.subplots_adjust(top=0.92)
#         matplotlib.pyplot.savefig(os.path.join(dir_output,'plot_time','plot_time_'+label+'.svg'))
#         matplotlib.pyplot.show()
#         
#         ################################################################# plot times series
# 
# #         matplotlib.pyplot.clf()
# #         #T2=numpy.where(T>2,T,2).min()
# #         #use=0.40/T
# #         #gap=0.60/(T2-1.0)
# #         data=numpy.load(os.path.join(dir_output,'data_datn','data_datn.npy'))[l]
# #         #sec=t*use+t*gap+e*use/(E2-1.0)
# #         matplotlib.pyplot.title(label)
# #         #matplotlib.pyplot.plot(data,lw=2,color=matplotlib.pyplot.cm.jet(sec),label=treatments[t]+','+experiments[t][e])
# #         matplotlib.pyplot.xlabel('frame')
# #         matplotlib.pyplot.ylabel('absolute')
# #         #matplotlib.pyplot.legend()
# #         matplotlib.pyplot.tight_layout()
# #         matplotlib.pyplot.subplots_adjust(top=0.92)
# #         matplotlib.pyplot.savefig(os.path.join(dir_output,'plot_time','plot_time_'+label+'.svg'))
# #         matplotlib.pyplot.show()
# #         
#             
#     ################################################################# plot publication   
#     ################################################################# plot example grids
#     
   
    
    ly = imHeight
    lx = imWidth
    
    gn=networkx.from_edgelist(numpy.load(os.path.join(dir_output,'data_grph','data_grph.npy')))

    N=gn.number_of_nodes()

    gc=networkx.convert_node_labels_to_integers(networkx.subgraph(gn,range(N-N/1,N)),ordering='sorted') # lz =1

    posi=pos#numpy.load(os.path.join(dir_output,'data_posi','data_posi.npy')).flatten()[0]

    pos2D,pos3D=img2net_help.grid_pos2D(1,pos)    #lz =1

    en=numpy.array([d['capa'] for u,v,d in gn.edges_iter(data=1)])

    en=en/en.max()         

    ec=numpy.array([d['capa'] for u,v,d in gc.edges_iter(data=1)])

    ec=ec/en.max()    

    matplotlib.pyplot.clf()       
    #fig=matplotlib.pyplot.figure(1,projection='3d',axisbg='white')

    #fig.view_init(elev=30,azim=20)
    
    #matplotlib.pyplot.figure(1)                

    matplotlib.pyplot.imshow(im,cmap='gray',origin='lower')  #,extent=[bx,lx,by,ly]

    if (AMR==1):
        networkx.draw_networkx_edges(gn,pos2D,width=2,edge_color=en)
    else:
        networkx.draw_networkx_edges(gc,pos2D,width=2,edge_color=ec)

    matplotlib.pyplot.axis('off')              

    L=gn.number_of_edges()
    print L
    for l in range(L):
        [u,v]=gn.edges()[l][0:2]
        if(pos[u]!=pos[v]):
            alp=1.0
        else:
            alp=0.4
        colors = matplotlib.pyplot.get_cmap('jet')
        matplotlib.pyplot.plot([pos[u][0],pos[v][0]],[pos[u][1],pos[v][1]],color=colors(en[l]),alpha=alp,lw=2)#[pos[u][2],pos[v][2]]
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.xlim(0,lx)
    matplotlib.pyplot.ylim(0,ly)
    mng = matplotlib.pyplot.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())       
    matplotlib.pyplot.savefig(os.path.join(dir_output,'plot_grid','plot_network.pdf'), bbox_inches=0,dpi=1000)
    matplotlib.pyplot.show()
    #print time.time()-temp
    return 0
