from PIL import Image
import io 
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask,render_template,request,redirect,url_for
from io import BytesIO
import matplotlib.lines as mlines

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/",methods=['POST'])
def img_processing():
    input_obtained =  request.form['img_input']
    
    output_img = image_processor(input_obtained)
    return render_template('output.html',output_image=output_img)

def image_processor(input_obtained):
    input_img = input_obtained

    #Getting the base64 image
    im = Image.open(BytesIO(base64.b64decode(input_img)))
    im.save('Processed_Images/processed.png', 'PNG')

    im = Image.open("Processed_Images/processed.png")
    rgb_im = im.convert('RGB')
    rgb_im.save('Processed_Images/converted.jpeg')

    #Resize the image    
    orig_img = cv2.imread('Processed_Images/converted.jpeg',0)
    aspect_ratio = orig_img.shape[0] / orig_img.shape[1]
    height = 400
    width = int(height / aspect_ratio)
    
    dims = (width,height)
    resized_img = cv2.resize(orig_img, dims, interpolation = cv2.INTER_AREA)

    #Write the resized image
    cv2.imwrite('Processed_Images/resized_bw.jpeg',resized_img)    

    #Detect the edges
    sobel_img = sobel_filter(resized_img)
    cv2.imwrite('Processed_Images/sobel_output.jpeg',sobel_img)
    cv2.imwrite('static/sobel_output.jpeg',sobel_img)

    #Normalize and convert to binary
    sobel_norm = sobel_img/sobel_img.max()
    binary_img = sobel_norm.copy()
    for i in range(sobel_norm.shape[0]):
        for k in range(sobel_norm.shape[1]):
            if sobel_norm[i][k] < 0.4:
                binary_img[i][k] = 0
            else:
                binary_img[i][k] = 1

    #Hough transform
    hough_accumulator, thetas, rhos = hough_transform(binary_img)

    #Plot hough lines
    hough_plot(hough_accumulator,binary_img,rhos,thetas)

    return "Success"

def sobel_filter(img):
    sobel_img = np.copy(img)
    size = sobel_img.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            sobel_img[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return sobel_img

def hough_transform(input_img):
    #Init theta from -90 to +90
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    #Init the rho from -len(diag) to len(diag)
    img_height,img_width = input_img.shape[:2]
    #print("Image of width and height: ",img_width,img_height)    
    diag = int(round(np.sqrt((img_width**2 + img_height**2))))
    #print("Diagonal: ",diag)
    rhos = np.linspace(-diag, diag, diag * 2.0)

    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    n_thetas = len(thetas)
     
    #Hough accumulator
    hough_accumulator = np.zeros([2*diag, n_thetas],dtype=int)  
    y_idxs, x_idxs = np.nonzero(input_img)
    #print("Length of x_idxs and y_idxs",len(x_idxs),len(y_idxs))

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(n_thetas):
        # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_theta[t_idx] + y * sin_theta[t_idx]) + diag)            
            hough_accumulator[rho, t_idx] += 1
    return hough_accumulator, thetas, rhos

def hough_plot(hough_accumulator,binary_img,rhos,thetas):
    idx = np.argmax(hough_accumulator)
    rho = rhos[int(idx / hough_accumulator.shape[1])]
    theta = thetas[idx % hough_accumulator.shape[1]]   
    
    #Threshold value
    t_count=250
    fig2, ax1 = plt.subplots(ncols=1, nrows=1)
    ax1.imshow(binary_img)
    edge_height_half, edge_width_half = binary_img.shape[1] / 2, binary_img.shape[0] / 2
    for y in range(hough_accumulator.shape[0]):
        for x in range(hough_accumulator.shape[1]):
            if hough_accumulator[y][x] > t_count:
                rho = rhos[y]
                theta = thetas[x]
                #a = np.cos(np.deg2rad(theta))
                #b = np.sin(np.deg2rad(theta))
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = (a * rho) + edge_width_half
                y0 = (b * rho) + edge_height_half
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                ax1.plot([x1, x2],[y1, y2],'xb-',linewidth=3)
                ax1.set_ylim([binary_img.shape[0],0])
                ax1.set_xlim([0,binary_img.shape[1]])
    fig2.savefig('static/output.png')

app.run(host='0.0.0.0', port=5000)