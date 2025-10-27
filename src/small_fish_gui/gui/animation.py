import FreeSimpleGUI as sg
import numpy as np

WAITING_TEXT = [
    'Running...',
    'Computing science...',
    'Preparing some good results...',
    'A good day for science...'
]

def add_default_loading(funct) :
    def inner(*args,**kwargs) :

        hide_loading = kwargs.get("hide_loading")
        if 'hide_loading' in kwargs : del kwargs['hide_loading']

        if not hide_loading :
            random_text = np.random.randint(0,len(WAITING_TEXT))
            waiting_layout = [
            [sg.Text(WAITING_TEXT[random_text], font= '10')]
            ]
            window = sg.Window(
            title= 'small_fish',
            layout= waiting_layout,
            grab_anywhere= True,
            finalize=True
        )

            window.read(timeout= 30, close= False)
            try :
                return funct(*args, **kwargs)
            finally :
                window.close()

        else : 
            return funct(*args, **kwargs)

    return inner

