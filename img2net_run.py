##################################################################### imports

import img2net_calc
reload(img2net_calc)

class img2net:
    def __init__(self):
        dir_input = "./images/"
        
        file_name ="22.png"
        gridtype ='rectangular'
        img2net_calc.img2net_calc(gridtype,dir_input,file_name)
    
if __name__ == '__main__':
    app = img2net()


