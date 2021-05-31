


###############################################################################
#                       Apertura h5 da NYU train db 88GB                      #
###############################################################################



### CONTROLLO DELLA DIRECTORY:
import os

pth=os.getcwd()
#os.chdir(r"/Users/symon/Documents/01_PROGETTI/01_MEDIA/01_PythonWorkSpace/01_RI1.3/01_fastMRI/01_Dati")



import h5py
import matplotlib.pyplot as plt


### CONTROLLO DELLE DIMENSIONI DEI VARI CONTENUTI:
#print(bf.keys())
#print(bf['ismrmrd_header'])
#print(bf['kspace'])
#print(bf['reconstruction_esc'])
#print(bf['reconstruction_rss'])
#bf['reconstruction_rss'].shape
#bf['reconstruction_esc'].shape
#bf['kspace'].shape

### CONTROLLO DEL FILE "ismrmrd_header":
#import xml.etree.ElementTree as etree
#from fastmri.data.mri_data import et_query

#with h5py.File(data_path, "r") as hf:
#    et_root = etree.fromstring(hf["ismrmrd_header"][()])
#    print(et_query(et_root, ["encoding", "reconSpace", "fieldOfView_mm", "z"]))


import pandas as pd



### PER OGNI FILE H5 VENGONO LETTI I DATI E SALVATI IN PNG DISTINGUENDO ANCHE IL TIPO DI SLICE (FS/NoFS):

import pandas as pd

fileNames = [fileName for fileName in os.listdir(pth) if fileName.endswith(".h5")]

for ind in range(len(fileNames)): #
    bf = h5py.File(fileNames[ind],'r')
    data_path = os.path.join(fileNames[ind])

    %matplotlib inline

    with h5py.File(data_path, 'r') as h5_data:
        for c_key in h5_data.keys():
            print(c_key, h5_data[c_key].shape, h5_data[c_key].dtype)
            cur_images = h5_data['reconstruction_rss'][0:bf['reconstruction_rss'].shape[0]]

    filename=data_path+'_rss_'

    slice_type='_Error_'
    if bf.attrs['acquisition'] == 'CORPDFS_FBK' : slice_type='FS_'
    else :
        if bf.attrs['acquisition'] == 'CORPD_FBK' : slice_type='NoFS_'

    from tqdm import tqdm
    for ind in tqdm(range(cur_images.shape[0])):
        ims = []
        temp_stack = cur_images[ind,:,:]
        plt.close('all')
        fig, ax1 = plt.subplots(1,1, figsize=(8,8))
        c_aximg = ax1.imshow(temp_stack, cmap='gray', interpolation='lanczos', animated = True)
        ax1.axis('off')
        plt.tight_layout()
        def update_image(frame):
            c_aximg.set_array(temp_stack[frame])
            return c_aximg,
        import matplotlib.pyplot as plt
        from matplotlib import pyplot as plt

        fileindex="%s" % ind
        dir_path=r"/Users/symon/Documents/01_PROGETTI/01_MEDIA/01_PythonWorkSpace/01_RI1.3/01_fastMRI/03_DatiPNG/99_DaElaborare/"
        fileformat='.png'
        filepath=dir_path+filename+slice_type+fileindex+fileformat

        fig.savefig(filepath)




### SALVATAGGIO DELL'INFO DI NoFS/FS PER OGNI FILE H5

import pandas as pd

### Leggo il dataframe relativo alle info dei file h5 salvato nell'ultima sessione
#df=pd.read_excel("Info_H5.xlsx") # Solo le volte successive la prima


fileNames = [fileName for fileName in os.listdir(pth) if fileName.endswith(".h5")]

df = [] # --> Solo la prima volta assoluta
for ind in range(len(fileNames)):
    bf = h5py.File(fileNames[ind],'r')
    data_path = os.path.join(fileNames[ind])

    with h5py.File(data_path, 'r') as h5_data:
        for c_key in h5_data.keys():
            print(c_key, h5_data[c_key].shape, h5_data[c_key].dtype)
            cur_images = h5_data['reconstruction_rss'][0:bf['reconstruction_rss'].shape[0]]

    filename=data_path+'_rss_'

    slice_type='_Error_'
    if bf.attrs['acquisition'] == 'CORPDFS_FBK' : slice_type='FS_'
    else :
        if bf.attrs['acquisition'] == 'CORPD_FBK' : slice_type='NoFS_'

    df.append(  #df=df.append(  # --> Solo dalla seconda volta in poi
            {
                'File_Name': data_path+'_rss_',
                'Type' : slice_type
            }# , ignore_index=True # --> Solo dalla seconda volta in poi
    )

df=pd.DataFrame(df) # --> Solo la prima volta


### Salvo il dataframe relativo alle info dei file H5 per salvare la sessione e ricominciare successivamente
df=df.drop_duplicates(subset='File_Name', keep="last")
df2=df.sort_values('File_Name')
df.to_excel('Info_H5.xlsx', sheet_name='Sheet1', index=False)
