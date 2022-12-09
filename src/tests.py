from dataset import DataSet
import utils


def create_ds_test1():
    one_band_ds = DataSet(min_length=100, sep=',', start_col=1)
    one_band_ds.add_band('i', './ZTF_DR_data/i_band') 
    one_band_ds.preprocess()
    

def create_ds_test2():
    ds = DataSet(min_length=100, sep=',', start_col=1) \
            .add_band('i', './ZTF_DR_data/i_band') \
            .add_band('r', './ZTF_DR_data/r_band') \
            .add_band('g', './ZTF_DR_data/g_band') \ 
            .preprocess()
    
def get_data_test1():
    utils.get_data('./ZTF_DR_data')
    
    
def synth_data_test1():
    utils.get_synth_data()
    
    

def train_test1():
    pass
