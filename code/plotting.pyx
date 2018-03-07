'''
Created on 20 Apr 2016

@author: haifa
'''
import numpy 
import matplotlib.pyplot
import os
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