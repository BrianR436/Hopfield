import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy.spatial.distance import hamming

def imagen2vector():
    imagen0 = Image.open("0.jpg").convert(mode="L")
    imagen0_matriz = np.asarray(imagen0, dtype=np.uint8)
    vector1 = imagen0_matriz.flatten()
    
    imagen1 = Image.open("1.jpg").convert(mode="L")
    imagen1_matriz = np.asarray(imagen1,dtype=np.uint8)
    vector2 = imagen1_matriz.flatten()
    
    imagen2 = Image.open("2.jpg").convert(mode="L")
    imagen2_matriz = np.asanyarray(imagen2,dtype=np.uint8)
    vector3 = imagen2_matriz.flatten()

    imagen3 = Image.open("3.jpg").convert(mode="L")
    imagen3_matriz = np.asanyarray(imagen3,dtype=np.uint8)
    vector4 = imagen3_matriz.flatten()

    imagen4 = Image.open("4.jpg").convert(mode="L")
    imagen4_matriz = np.asanyarray(imagen4,dtype=np.uint8)
    vector5 = imagen4_matriz.flatten()

    imagen5 = Image.open("5.jpg").convert(mode="L")
    imagen5_matriz = np.asanyarray(imagen5,dtype=np.uint8)
    vector6 = imagen5_matriz.flatten()
    
    imagen6 = Image.open("6.jpg").convert(mode="L")
    imagen6_matriz = np.asanyarray(imagen6,dtype=np.uint8)
    vector7 = imagen6_matriz.flatten()
    
    imagen7 = Image.open("7.jpg").convert(mode="L")
    imagen7_matriz = np.asanyarray(imagen7,dtype=np.uint8)
    vector8 = imagen7_matriz.flatten()

    imagen8 = Image.open("8.jpg").convert(mode="L")
    imagen8_matriz = np.asanyarray(imagen8,dtype=np.uint8)
    vector9 = imagen8_matriz.flatten()

    imagen9 = Image.open("9.jpg").convert(mode="L")
    imagen9_matriz = np.asanyarray(imagen9,dtype=np.uint8)
    vector10 = imagen9_matriz.flatten()
    
    patrones = np.zeros([10,400])
    
    for k in range(400):
        if vector6[k] == 255:
            patrones[0][k] = 1
        else:
            patrones[0][k] = -1

    for k in range(400):
        if vector7[k] == 255:
            patrones[1][k] = 1
        else:
            patrones[1][k] = -1

    for k in range(400):
        if vector8[k] == 255:
            patrones[2][k] = 1
        else:
            patrones[2][k] = -1
    
    for k in range(400):
        if vector9[k] == 255:
            patrones[3][k] = 1
        else:
            patrones[3][k] = -1

    for k in range(400):
        if vector10[k] == 255:
            patrones[4][k] = 1
        else:
            patrones[4][k] = -1
    
    for k in range(400):
        if vector5[k] == 255:
            patrones[5][k] = 1
        else:
            patrones[5][k] = -1

    for k in range(400):
        if vector4[k] == 255:
            patrones[6][k] = 1
        else:
            patrones[6][k] = -1
    
    for k in range(400):
        if vector3[k] == 255:
            patrones[7][k] = 1
        else:
            patrones[7][k] = -1

    for k in range(400):
        if vector2[k] == 255:
            patrones[8][k] = 1
        else:
            patrones[8][k] = -1
    
    for k in range(400):
        if vector1[k] == 255:
            patrones[9][k] = 1
        else:
            patrones[9][k] = -1

    
    P=10
    N=400

    return patrones,P,N     

def matriz_pesos(patrones,M,N):
    FILAS=N
    matrizpesos = np.zeros([N,N])
    i=0
    peso = 0
    for i in range(FILAS):
        print("Cargando ", i," de 399")
        for x in range(N):  
            if i==x:
                matrizpesos[x][i]=0
            else:
                for S in range(M):
                    peso=peso+(patrones[S,x]*patrones[S,i])
                matrizpesos[x][i]=(1/N)*peso
                peso=0
    
    return matrizpesos

def imagen_de_prueba(N):
    vector_prueba = np.zeros([N])
    imagen_prueba = input("Ingrese la imagen de prueba:")
    imagen = Image.open(imagen_prueba).convert(mode="L")
    imagen_matriz = np.asanyarray(imagen,dtype=np.uint8)
    
    vector_imagen = imagen_matriz.flatten()
    #Paso mi imagen a vector y luego le aplico una funcion de tranferencia de 1 si =255 o -1 si = 0
    for k in range(400):
        if vector_imagen[k] == 255:
            vector_prueba[k] = 1
        else:
            vector_prueba[k] = -1
    return vector_prueba,imagen_prueba

def procesar(m_pesos,N):
    vector_prueba = np.zeros([N])
    vector_prueba,imagen_prueba= imagen_de_prueba(N)
    intentos = 0
    while intentos < 100 :
        vector_resultado = np.zeros([N])
        nuevo_valor = 0
        for x in range(N):
            for s in range (N):
                nuevo_valor = nuevo_valor + m_pesos[x][s] * vector_prueba[s]
            vector_resultado[x] = nuevo_valor
            nuevo_valor = 0
        #Aplico funcion de transferencia donde -1, si x < 0 o 1, si x>=0
        for d in range(N):
            if vector_resultado[d]<0:
                vector_resultado[d]=-1
            else:
                vector_resultado[d]=1

        intentos = condicion_convergencia(vector_prueba,vector_resultado,intentos) #Vemos si y(n)=y(n-1)
        
        vector_prueba = vector_resultado
    
    return vector_prueba,imagen_prueba

def condicion_convergencia(vector_prueba,vector_resultado,intentos):
    if np.array_equal(vector_resultado,vector_prueba) == True:
        intentos=100
    else :
        intentos=intentos+1
    return intentos

def distancia_HAMMING(vector_resultado,P,N,patrones):
    
    distancias = np.zeros([P])
    comparacion = np.zeros([N])
    for i in range(P):
        for j in range(N):
            comparacion[j] = patrones[i][j]
        distancias[i]= hamming(vector_resultado,comparacion)
        
        distancia_minima = min(distancias)
    for k in range(P):
        if distancias[k]==distancia_minima:
            imagen_asociada=k
    return imagen_asociada
    
def imprimir_imagen(imagen_asociada,imagen_prueba):
    if imagen_asociada==0:
        img = mpimg.imread('5.jpg')
        img2 = mpimg.imread(imagen_prueba)
        plt.subplot(1,2,1), plt.title("Imagen de prueba")
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,2,2), plt.title("Asocia con la imagen 5")
        plt.imshow(img), plt.axis('off')

        plt.show()

    if imagen_asociada==1:
        img = mpimg.imread('6.jpg')
        img2 = mpimg.imread(imagen_prueba)
        plt.subplot(1,2,1), plt.title("Imagen de prueba")
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,2,2), plt.title("Asocia con la imagen 6")
        plt.imshow(img), plt.axis('off')

        plt.show()
    if imagen_asociada==2:
        img = mpimg.imread('7.jpg')
        img2 = mpimg.imread(imagen_prueba)
        plt.subplot(1,2,1), plt.title("Imagen de prueba")
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,2,2), plt.title("Asocia con la imagen 7")
        plt.imshow(img), plt.axis('off')

        plt.show()

    if imagen_asociada==3:
        img = mpimg.imread('8.jpg')
        img2 = mpimg.imread(imagen_prueba)
        plt.subplot(1,2,1), plt.title("Imagen de prueba")
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,2,2), plt.title("Asocia con la imagen 8")
        plt.imshow(img), plt.axis('off')

        plt.show()

    if imagen_asociada==4:
        img = mpimg.imread('9.jpg')
        img2 = mpimg.imread(imagen_prueba)
        plt.subplot(1,2,1), plt.title("Imagen de prueba")
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,2,2), plt.title("Asocia con la imagen 9")
        plt.imshow(img), plt.axis('off')

        plt.show()
    if imagen_asociada==5:
        img = mpimg.imread('4.jpg')
        img2 = mpimg.imread(imagen_prueba)
        plt.subplot(1,2,1), plt.title("Imagen de prueba")
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,2,2), plt.title("Asocia con la imagen 4")
        plt.imshow(img), plt.axis('off')

        plt.show()

    if imagen_asociada==6:
        img = mpimg.imread('3.jpg')
        img2 = mpimg.imread(imagen_prueba)
        plt.subplot(1,2,1), plt.title("Imagen de prueba")
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,2,2), plt.title("Asocia con la imagen 3")
        plt.imshow(img), plt.axis('off')

        plt.show()
    if imagen_asociada==7:
        img = mpimg.imread('2.jpg')
        img2 = mpimg.imread(imagen_prueba)
        plt.subplot(1,2,1), plt.title("Imagen de prueba")
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,2,2), plt.title("Asocia con la imagen 2")
        plt.imshow(img), plt.axis('off')

        plt.show()

    if imagen_asociada==8:
        img = mpimg.imread('1.jpg')
        img2 = mpimg.imread(imagen_prueba)
        plt.subplot(1,2,1), plt.title("Imagen de prueba")
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,2,2), plt.title("Asocia con la imagen 1")
        plt.imshow(img), plt.axis('off')

        plt.show()


    if imagen_asociada==9:
  
        img2 = mpimg.imread(imagen_prueba)
        img = mpimg.imread('0.jpg')
        plt.subplot(1,2,1), plt.title("Imagen de prueba")
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,2,2), plt.title("Asocia con la imagen 0")
        plt.imshow(img), plt.axis('off')

        plt.show()

def main():
    patrones,P,N= imagen2vector()
    pesos = matriz_pesos(patrones,P,N)
    desicion = 1
    while desicion == 1:
        vector_resultado,imagen_prueba = procesar(pesos,N)
        imagen_asociada = distancia_HAMMING(vector_resultado,P,N,patrones)
        imprimir_imagen(imagen_asociada,imagen_prueba)
        
        desicion = int(input("¿Desea probar otra imagen? 1-Si 2-No "))

if __name__ == '__main__':
    main()
