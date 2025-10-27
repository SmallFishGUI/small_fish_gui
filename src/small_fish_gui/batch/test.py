import FreeSimpleGUI as sg
import small_fish_gui.batch as batch
import pandas as pd

try :
    batch.batch_promp(pd.DataFrame(), pd.DataFrame(), acquisition_id=0,)
except Exception as e :
    raise e

print('end')