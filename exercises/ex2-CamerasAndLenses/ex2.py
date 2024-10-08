import math
print(math.degrees(math.atan2(3,10)))

def camera_b_distance(f, g):
    """
    camera_b_distance returns the distance (b) where the CCD should be placed
    when the object distance (g) and the focal length (f) are given
    :param f: Focal length
    :param g: Object distance
    :return: b, the distance where the CCD should be placed
    """
    return (f*g)/(g-f)

print(camera_b_distance(15,100))

print(camera_b_distance(8,5000))

print((camera_b_distance(8,5000)*1800)/5000)

print((4.8*6.4)/(640*480))
