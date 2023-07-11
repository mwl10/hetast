from utils import intrp
import sys
import os

#hacky ugly, but model is giving screwed up accuracies locally so do it on hpc 

# data_folders = ['ZTF_3C273', 
#                 'ZTF_MCG+08-11-011', 
#                 'ZTF_Mrk142',
#                 'ZTF_rm_segs/Mrk142/X/ZTF_epoch3_',
#                 'ZTF_NGC5548', 
#                 'ZTF_Mrk817', 
#                 'ZTF_rm_segs/Mrk817/X/ZTF_epoch0_',
#                 'ZTF_rm_segs/Mrk817/X/ZTF_epoch2_'
#                ]

# cp_files = ['3C273/ZTF_3C273-1.3744778633117676.h5',
#             'MCG+08-11-011/ZTF_MCG+08-11-011-2.1896493434906006.h5',
#             'Mrk142/ZTF_Mrk142-1.336236834526062.h5',
#             'Mrk142/ZTF_epoch3_-1.330850601196289.h5',
#             'NGC5548/ZTF_NGC5548-1.2497953176498413.h5',
#             'Mrk817/ZTF_Mrk817-1.1086530685424805.h5',
#             'Mrk817/ZTF_epoch0_-0.5678340792655945.h5',
#             'Mrk817/ZTF_epoch2_-1.4073609113693237.h5']

data_folders = ['ZTF_3C273', 
                'ZTF_MCG+08-11-011', 
                'ZTF_Mrk142',
                'ZTF_NGC5548', 
                'ZTF_Mrk817', 
               ]

cp_files  = ['ZTF_gri0.8933992385864258.h5','ZTF_gri0.8933992385864258.h5',
            'ZTF_gri0.8933992385864258.h5','ZTF_gri0.8933992385864258.h5',
             'ZTF_gri0.8933992385864258.h5']


        
# data_folders = ['ZTF_MCG+08-11-011_g', 
#                 'ZTF_MCG+08-11-011_r']
                
# cp_files = ['ZTF_MCG+08-11-011_g-1.7458717823028564.h5',
#             'ZTF_MCG+08-11-011_r-1.6788063049316406.h5']

data_folders = [os.path.join('datasets', df) for df in data_folders]
cp_files = [os.path.join('datasets', cp) for cp in cp_files]


for data_folder,cp_file in zip(data_folders,cp_files):
    intrp(data_folder, cp_file, resolution=0.1, save_folder='intrps', device='cuda')