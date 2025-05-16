from numpy import sin, cos, pi
def convert_actual_to_scaled(model, tensor):
    return 0

def convert_scaled_to_actual(model,tensor):
    return 0

def convert_wind(magnitude,direction,degrees=True):
    if degrees:
        direction = direction/(180)*pi
    return magnitude * cos(direction), magnitude*sin(direction)

# print(convert_wind(1,45))