def yielder(n) :
    for i in range(n) : 
        yield i

def constructor(n) :
    def inner() :
        return yielder(n)
    return inner

#test

N = 10

gen1 = yielder(N)

a = constructor(10)

print(type(a))

gen2 = a()
gen3 = a()

print('gen1 : ', list(gen1))
print('gen2 : ', list(gen2))
print('gen3 : ', list(gen3))
print('gen3 : ', list(gen3))


#Napari test

import napari
import numpy as np
from napari.utils.events import Event
from napari.layers import Points

seed = 0
points_number = 500
rand_gen = np.random.default_rng(seed=seed)

im_shape = (10,1000,1000)
scale = (1,1,1)

im = np.zeros(shape=im_shape)

points = np.array(
    list(zip(
        rand_gen.integers(0,im_shape[0],size=points_number), #Z
        rand_gen.integers(0,im_shape[1],size=points_number), #Y
        rand_gen.integers(0,im_shape[2],size=points_number), #X
        )),
    dtype= int
)

spots_number = rand_gen.integers(0,20,points_number, dtype=int)
id_list = np.arange(points_number)

Viewer = napari.Viewer(title= "test", show=False)
Viewer.add_image(im),
cluster_layer = Viewer.add_points(
    points, 
    size = 10, 
    scale=scale, 
    face_color= 'blue', 
    opacity= 0.7, 
    symbol= 'diamond', 
    name= 'foci', 
    features= {"spot_number" : spots_number, "id" : id_list}, 
    feature_defaults= {"spot_number" : 0, "id" : -1})


class Points_callback :
    """
    Custom class to handle points number evolution during Napari run.
    """
    
    def __init__(self, points, next_id) -> None:
        self.points = points
        self.next_id = next_id
        self._set_callback()
    
    def __str__(self) -> str:
        string = 'Points_callback object state :\ncurrent_points_number : {0}\ncurrnet_id : {1}'.format(self.current_points_number, self.next_id)
        return string
    
    def get_points(self) :
        return self.points
    
    def get_next_id(self) : 
        return self.next_id
    
    def _set_callback(self) :
        def callback(event:Event) :

            old_points = self.get_points()
            new_points:Points = event.source.data
            features = event.source.features
            
            current_point_number = len(old_points)
            next_id = self.get_next_id()
            new_points_number = len(new_points)

            if new_points_number > current_point_number :
                features.at[new_points_number - 1, "id"] = next_id
                self.next_id += 1

            #preparing next callback
            self.points = new_points
            self._set_callback()
        self.callback = callback

live_points_number = len(points)
next_point_id = id_list[-1] + 1
_callback = Points_callback(points=points, next_id=next_point_id)
Viewer.show(block=False)
points_callback = Viewer.layers[1].events.data.connect((_callback, 'callback'))
napari.run()
