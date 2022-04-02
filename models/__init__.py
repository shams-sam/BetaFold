from .models import *

def get_model(name, **kwargs):
    if name == 'dist':
        return deepcon_rdd_distances(**kwargs)
    elif name == 'contact':
        return deepcon_rdd(**kwargs)
    elif name == 'binned':
        return deepcon_rdd_binned(**kwargs)
