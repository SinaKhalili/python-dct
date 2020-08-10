import numpy as np
import math


test = np.matrix([100,110,120,130,140,150,160,170])

mat =  np.matrix([[0.3536,0.3536,0.3536,   0.3536,0.3536,  0.3536,0.3536,0.3536],
                  [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
                  [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
                  [0.4157,-0.0975,-0.4904,  -0.2778,  0.2778, 0.4904, 0.0975,-0.4157],
                  [0.3536,-0.3536,-0.3536, 0.3536,0.3536, -0.3536,-0.3536,0.3536],
                  [0.2778,-0.4904,0.0975,  0.4157,-0.4157,-0.0975,0.4904,-0.2778],
                  [0.1913,-0.4619,0.4619, -0.1913,-0.1913, 0.4619,-0.4619,0.1913],
                  [0.0975,-0.2778,0.4157, -0.4904,0.4904, -0.4157,0.2778,-0.0975]])

def dct(N):
    """
    Returns an NxN 1d dct transform matrix
    """
    transform_matrix = np.identity(N)
    denom = math.sqrt(2.0)
    num = math.sqrt(2.0/N)

    for i in range(N):
        transform_matrix[0,i] = num / denom

    for u in range(1,N):
        for i in range(N):
            transform_matrix[u,i] =   (math.sqrt(2.0/N)
                                    * math.cos(
                                        (math.pi/N) 
                                        * u         
                                        * (i + 0.5) 
                                      ))
     
    return transform_matrix

def transform(array):
    """
    Applies the DCT transform with 6 levels of precision
    """
    array = np.matrix(array)
    _, N = array.shape

    return np.round(dct(N) * array.transpose(), 6)


if __name__ == "__main__":
    arr = input("Enter the input space-separated (ex. 100 110 120 130 140 150 160 170): ")
    arr = arr.split(' ')
    arr = [int(elem) for elem in arr]
    N = len(arr)
    print(f"Your transformation matrix for size {N} is: ")
    print(dct(N))
    
    print(f"Your transformed input is: ")
    print(transform(arr))
