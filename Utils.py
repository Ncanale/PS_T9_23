# config_file = r"./config.json"
print('Running Utilis')
import json
Swan = False
if Swan:
    config_file = r"./config.json"
else:
    config_file = r"./config.json"
# print (config_file)

with open(config_file, "r") as f:
            dizi = json.load(f)
        
dizi

def myGauss(x, a, mu, sigma):
    import numpy as np
    return a * np.exp(- (x-mu)**2 / (2*sigma**2))

def myGauss_line(x, a, mu, sigma,m,q):
    import numpy as np
    return a * np.exp(- (x-mu)**2 / (2*sigma**2)) + (m*x + q)

# def project_to_z_pos(z1,z2,z4):
def projectDistZ(x1,x2,y1,y2,z):
    
    mx = (x2-x1)/dizi['d_12']
    xProj = x1 + mx * z
    
    my = (y2-y1)/dizi['d_12']
    yProj = y1 + my * z
    
    return (xProj, yProj)

def Average(lst):
    return sum(lst) / len(lst)

def file_corrector(runs):
    import numpy as np
    import h5py
    from collections.abc import Iterable
    if Swan:
        data_dir = dizi['data_path']
    else:
        data_dir = "./data_22"

    pos = []
    infos = []
    phs = []
    tmis = []
    evis =[]

    if not isinstance(runs, Iterable):
        runs = [runs]

    for run in runs:
        data_path = f'{data_dir}/run{run}.h5'
        
        with h5py.File(data_path, 'r', libver='latest', swmr=True) as hf:
            print(hf.keys())
            # hf["xpos"].shape
            keys = list(hf.keys())
            #for k in hf.keys():
            #    comand = f'{k} = np.array(hf["{k}"])'
                # print(comand)
            #  exec(comand)
            pos.append(np.array(hf['xpos']))
            infos.append(np.array(hf['xinfo']))
            phs.append(np.array(hf['digi_ph']))
            tmis.append(np.array(hf['digi_time']))
            evis.append(np.array(hf['Ievent']))

    #print(np.shape(pos))
    # print(np.shape(infos))
            
    xpos = np.concatenate(pos,axis=0)
    xinfo = np.concatenate(infos,axis=0)
    ph = np.concatenate(phs,axis=0)
    tm = np.concatenate(tmis,axis=0)
    evi = np.concatenate(evis,axis=0)

    #print(np.shape(xpos))
    # print(np.shape(xinfo))

    ##purge errors
    logic = (xpos > -1) & (xpos < 15)
    logic2 = logic.all(axis = 1)
    xpos = xpos[logic2]   

    xinfo = xinfo[logic2]
    ph = ph[logic2]
    tm = tm[logic2]
    evi = evi[logic2]
    
    Rino1 = ph[:,1]
    Rino2 = ph[:,2]
    APC1 = (ph[:,3])
    APC2 = (ph[:,4])
    print('offset_y2 ', dizi['offset_y2'])
    print('offset_x2 ',dizi['offset_x2'])
    print(xpos[:,2][0])
    xpos[:,2] -= dizi['offset_y2']
    print(xpos[:,2][0])
    xpos[:,3] -= dizi['offset_x2']

    y1 = xpos[:,0]
    x1 = xpos[:,1]
    y2 = xpos[:,2] 
    x2 = xpos[:,3] 
    x_cry, y_cry = projectDistZ(x1,x2,y1,y2,dizi['d_1c'])
    return xpos,xinfo,ph,tm,evi,Rino1,Rino2,APC1,APC2,x1,y1,x2,y2,x_cry,y_cry 