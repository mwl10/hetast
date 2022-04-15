import pandas as pd
import numpy as np

# read tsv to numpy array, each one will have the length of 
# what kind of exceptions do we want to catch down the line?
    # more or less than 3 col

# time column, flux values, errors 
def file_to_np(*args):
    light_curves = []
    # find longest file in terms of lines in the files, set that to the array lengths
    for file in args:
         with open(file, 'r') as f:
             light_curve = pd.read_csv(file, sep='\t').to_numpy()
    
             # visually check there are three columns for each file 
             print(f"dims of {file}:\t{light_curve.shape}")
             light_curves.append(light_curve)
             
    return light_curves



# could also just load an entire directory of lcs
# or pickles? whatever becomes obvious eventually really...


# want to convert to torch dataloader now...
# just set the length to the max length of the light curves?
# it'll 


def main():
    pass
    
    
if __name__ == "__main__":
    main()





