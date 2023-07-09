# Object Detection on a video.
# our goal is to detect the dog in video.
#.pth file contains th eweights of the already pre-trained neural network and through mapping with a dictionary, 
#we will transfer these weights to the model we implement.
#opencv was not able to detect this dog. therefore, not a powerful model.

# Importing the libraries
import torch
#have dynamic graphs that helps in efficient computation of the gradient.
from torch.autograd import Variable
#we will convert the tensors into some tensors variable that will contain both tensor and variable and will be one element of the graph.
import cv2
#just to make the rectangle and not implementing using opencv
from data import BaseTransform, VOC_CLASSES as labelmap
#base transform will be used to transform in this format that they can be accepted into the neural network.
#voc_classess is just dictionary that will do including of the classes.
from ssd import build_ssd
import imageio# to process image of a video.

#it will not do the detection on each image of the video.
#it will not do the detection of the video directly.it will do the detection of the image of the video and then, with IO we will manage to extract all the frames of the videos and then reassemble the whole thing to make the video with rectangles detecting the dog and the human..
#so, it does frame by frame detection.

# Defining a function that will do the detections
def detect(frame, net, transform): # We define a detect function that will take as inputs, a frame[image on which detect function is applied to detect the object], a ssd neural network, and a transformation to be applied on the images[so that they are compatible with the neural network], and that will return the frame with the detector rectangle.
    height, width = frame.shape[:2] # We get the height and the width of the frame.
    #frame has three attributes shape that returns 3 parameters: height, width, no of channels.
    #if you have black and white channel you will have 1 channel and for coloured you will have 3 for red,blue,green.
    
    #now we going to do series of transformation to do before getting to this torch variable.
    #1) to do transform transformation to ensure that the image has right format that is right dimensions and the right color values.
    frame_t = transform(frame)[0] # We apply the transformation to our frame.
    
    #2) then, transform this frame from numpy array to a torch sensor.
    #a sensor is more advanced matrix and a more advanced array.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the frame into a torch tensor.
    #permute is for permutation of the colour that is form red,blue,green to green,red,blue.
    
    #3)We add a fake dimension corresponding to the batch.
    #4) to covert it into a torch variable.
    x = Variable(x.unsqueeze(0))
    
    y = net(x) # We feed the neural network ssd with the image and we get the output y.
    detections = y.data # We create the detections tensor contained in the output y.
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height]. is used to do normalization (between zero and one) the scale of values of the positions of the object detected in the image.
    #the first will correspond to the scale of the values of the upper left corner of the rectangular detector and the second will correpond to the scales values of the lower right corner of the same rectangular detector.
    
    #detection = [batch, no of classes(means object that can be detected)(each class stands for each object that can be detected), no of occurrence of the class, 
    #tuple of 5 elements that are (score, x0, y0, x1, y1) {for each occurence of a class we get a score and the coordinates of the upper left corner and lower right corner. if the score is lower than 0.6, then occurence of the class won't be found be found in that image. 
    #means it is threshold for occurrence of each class}]
    
    #we need for loop to iterate through all the classes and all the occurrence of all the classes.
    #detections.size(1) exactly tells the no of classes.
    for i in range(detections.size(1)): # For every class:
        j = 0 # We initialize the loop variable j that will correspond to the occurrences of the class.
        #score >= 0.6
        while detections[0, i, j, 0] >= 0.6: # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
            #convert to numpy array becoz opencv works with numpy array.
            pt = (detections[0, i, j, 1:] * scale).numpy() # We get the coordinates of the points at the upper left and the lower right of the detector rectangle.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # We draw a rectangle around the detected object.
            #to print the lable onto the rectangle.Lable that itt is a dog or it is a person.
            #image, lable with help of library we imported, coordinates of the rectangular box, fonts, size of the text, color of the text, thickness of the text, display our text with continueous lines and not with little dots.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # We put the label of the class right above the rectangle.
            j += 1 # We increment j to get to the next occurrence.
    return frame # We return the original frame with the detector rectangle and the label around the detected object.

# Creating the SSD neural network
#we input the phase inside it that is train phase and test phase and since we have already train our model. therefore, we will test our model.
net = build_ssd('test') # We create an object that is our neural network ssd.
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).

# Creating the transformation
#target size of the image to feed the NN, right scale for core values on which NN was trained
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network. to make sure they are compatible.

# Doing some Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('output.mp4', fps = fps) # We create an output video with this same fps frequence.
for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(frame) # We add the next frame in the output video.
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.