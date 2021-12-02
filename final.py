import numpy as np
import cv2
import math
import sys
from matplotlib import pyplot as plt
from PIL import ImageChops
import math, operator



#-----------------------------load the image and convert into gray scale -----------------------------#
def load():
    img = cv2.imread('F:\\Sadaru Kavisha\\Semester 5\\Image Processing\\images\\1.jpg',0)
    return img

#-------------------------------------------------------------------------------------------------------#

#-------------------------------------compute histogram*------------------------------------------#
def computeHistogram(img):
    h=[]
    for l in range(0,256):
        counter=np.sum(img == l) 
        h.append(counter)
    
    return h

#show cloumn chart
def showColumnChart(arr):
    new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    plt.bar(new_arr, arr)
    plt.title('Country Vs GDP Per Capita')
    plt.xlabel('Country')
    plt.ylabel('GDP Per Capita')
    plt.show()
    return


#show histogram of a given image
def showHistogram(img):
    color = "black"
  
    plt.xlim([0, 256])
    
    histogram, bin_edges = np.histogram(
        img[:], bins=512, range=(0, 512)
    )
    plt.plot(bin_edges[0:-1], histogram, color=color)

    plt.xlabel("Color value")
    plt.ylabel("Pixels")

    plt.show()
    return
#------------------------------------------------------------------------------------------------------#

#get pixel count of a given image
def pixelCount(img):
    return len(img)*len(img[0])

#-------------------------------------contrast streching----------------------------#
#change the contrast 0-75
def changeContrastAlpha(img):
    alpha=1

    new_img=img*2
    #print(img)
    #print(new_img)
    return new_img
#change the contrast 75-200
def changeContrastBeta(img):
    return

#change the contrast above 201
def changeContrastGama(img):
    return
#----------------------------------------------------------------------------------#

#---------------------------Mededian filter---------------------------------------#
def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data[i][j] = temp[len(temp) // 2]
            temp = []
    return data
#---------------------------------------------------------------------------------------------#

#------------------------------------------------Sharpening the image------------------------------------#
def sharp(image):
    kernel = np.array([[0, -1,0],
                   [-1,4,-1],
                   [0,-1,0]])
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) 
    return new_image
#-------------------------------------------------------------------------------------------------------#



#----------------------------------------Computing the RMS error-----------------------#
def rmsdiff(im1, im2):
    a = np.array(im1) # your x
    b = np.array(im2) # your y
    mses = ((a-b)**2).mean()
    return mses

#------------------------------------------------------------------------------------------#






#-------------------------------computing the entropy--------------------#
def computeEntropy(arr,img):
    totalPixel=pixelCount(img)
    
    H=0
    for i in range(0,len(arr)):
        p=arr[i]/totalPixel
        if(p==0):
            continue
        else:
            try:
                H+=-i*math.log10(float(p))
            except:
                break

    return H
#-------------------------------------------------------------#


#compute probabilities
def computeProbabilities(arr,img):
    totalPixel=pixelCount(img)
    prob={}
    level=0
    for i in range(0,len(arr)):
        
        p=arr[i]/totalPixel
        prob[level]=p

        level+=1
    return prob

#--------------------------------------Huffman Encoding------------------------------------------------------------------#
# A Huffman Tree Node
class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        # probability of symbol
        self.prob = prob

        # symbol 
        self.symbol = symbol

        # left node
        self.left = left

        # right node
        self.right = right

        # tree direction (0/1)
        self.code = ''



#calculate codes
codes = dict()

def Calculate_Codes(node, val=''):
    # huffman code for current node
    newVal = val + str(node.code)

    if(node.left):
        Calculate_Codes(node.left, newVal)
    if(node.right):
        Calculate_Codes(node.right, newVal)

    if(not node.left and not node.right):
        codes[node.symbol] = newVal
         
    return codes        

#calculate probabilities with data
def Calculate_Probability(data):
    symbols = dict()
    for element in data:
        for i in range(0,len(element)):
            if symbols.get(element[i]) == None:
                symbols[element[i]] = 1
            else: 
                symbols[element[i]] += 1     
    return symbols



#encoded output function
def Output_Encoded(data, coding):
    encoding_output = []
    for c in data:
        for pixel in c:
      #  print(coding[c], end = '')
            encoding_output.append(coding[pixel])
        
    string = ''.join([str(item) for item in encoding_output])    
    return string
        

         

def Huffman_Encoding(data):
    symbol_with_probs = Calculate_Probability(data)
    symbols = symbol_with_probs.keys()
    probabilities = symbol_with_probs.values()
    #print("symbols: ", symbols)
    #print("probabilities: ", probabilities)
    
    nodes = []
    
    # converting symbols and probabilities into huffman tree nodes
    for symbol in symbols:
        nodes.append(Node(symbol_with_probs.get(symbol), symbol))
    
    while len(nodes) > 1:
        # sort all the nodes in ascending order based on their probability
        nodes = sorted(nodes, key=lambda x: x.prob)
        # for node in nodes:  
        #      print(node.symbol, node.prob)
    
        # pick 2 smallest nodes
        right = nodes[0]
        left = nodes[1]
    
        left.code = 0
        right.code = 1
    
        # combine the 2 smallest nodes to create new node
        newNode = Node(left.prob+right.prob, left.symbol+right.symbol, left, right)
    
        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)
            
    huffman_encoding = Calculate_Codes(nodes[0])
    print("symbols with codes", huffman_encoding)

    encoded_output = Output_Encoded(data,huffman_encoding)
    return encoded_output, nodes[0]  
#-------------------------------------------------------------------------------------------------#   





#--------------------Huffman decording--------------------------------------------------------#
def Huffman_Decoding(encoded_data, huffman_tree):
    tree_head = huffman_tree
    decoded_output = []
    for x in encoded_data:
        if x == '1':
            huffman_tree = huffman_tree.right   
        elif x == '0':
            huffman_tree = huffman_tree.left
        try:
            if huffman_tree.left.symbol == None and huffman_tree.right.symbol == None:
                pass
        except AttributeError:
            decoded_output.append(huffman_tree.symbol)
            huffman_tree = tree_head
        
    string = ''.join([str(item) for item in decoded_output])
    return decoded_output        

#-----------------------------------------------------------------------------------------


#-------------------Write encoded data into a file------------------------------------------------#
def writeEncoded(data):
    file1 = open("encoded.txt", "w") 
    file1.write(data)
    file1.close() 
    return
#-------------------------------------------------------------------------------------------------#


#------------------Read encoded data---------------------------------------------------#
def readEncoded():
    f = open("encoded.txt", "r")
    encodedImg=f.read()
    print(encodedImg)
    return




def display_180321j(output):
    
    cv2.imshow("Output_180321J",output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main_180321j():
    
    gray=load()
    
    
    arr=computeHistogram(gray)

    print("Entropy is "+str(computeEntropy(arr,gray)))
    aftermedian=median_filter(gray,3)
    newGray=load()
    print("RMS different is "+str(rmsdiff(newGray,aftermedian)))
    probDic=computeProbabilities(arr,gray)
    encodedImage,Tree=Huffman_Encoding(gray)
    print(Huffman_Decoding(encodedImage,Tree))
    writeEncoded(encodedImage)
    showHistogram(aftermedian)
    display_180321j(newGray)
    display_180321j(aftermedian)
    aftersharp=sharp(aftermedian)
    display_180321j(aftersharp)
    return

main_180321j()

#while True:
#    x=input("Enter the number")
 #   print(x)
  #  if (x=='0'):
     #   break
 
        
