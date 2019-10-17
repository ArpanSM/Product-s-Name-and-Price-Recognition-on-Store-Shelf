from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from utils import label_map_util
from utils import visualization_utils as vis_util
import pytesseract
import smtplib 
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    # device_count = {'GPU': 1}
    )
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)
img_width, img_height = 150, 150 
model = tf.keras.models.load_model('./FINAL_FILES/tf_model.h5')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])



##############  INPUT INPUT INPUT ########################################
#INPUT THE IMAGE YOU WANT TO TEST
test_img_Location = './FINAL_FILES/test_images/xxxxxx.jpg'

#INPUT THE MAIL ADDRESS AS REQUIRED
login_mail='example@gmail.com'
login_password='xxxxxxxxx'
to_mail='exapmle@gmail.com'


##INPUT THE LOCATION OF PyTESSERACT.EXE FILE LOCATION
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract\\tesseract.exe'
##########################################################################




image_np = cv2.imread(test_img_Location)
y_len = np.size(image_np, 0)
x_len = np.size(image_np, 1)

# predicting images
img = image.load_img(test_img_Location, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images)
#print(classes)
if ((classes[0][0]) > 0.5 ):
    #print("not empty")
    out = 1
else:
    #print("empty")
    out = 0
lower = np.array([246,246,246]) 
upper = np.array([255,255,255])
if (y_len == 4032):
    x1=755
    y1=0
    x2=3024
    y2=290
    crop_img = image_np[y1:y2, x1:x2]
    dim = (1500,300)
    resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)#x1 y1 top left vertex
    mask = cv2.inRange(resized, lower, upper)
    mask = cv2.resize(mask, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("cropped", mask)
    #cv2.imwrite("filename.jpg", mask)
    text_time = pytesseract.image_to_string(mask)
    #print(text)
    
    x1l=755
    y1l=290
    x2l=3024
    y2l=1000
    crop_imgl = image_np[y1l:y2l, x1l:x2l]
    dim = (700,300)
    resizedl = cv2.resize(crop_imgl, dim, interpolation = cv2.INTER_AREA)#x1 y1 top left vertex
    maskl = cv2.inRange(resizedl, lower, upper)
    maskl = cv2.resize(maskl, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    maskl = cv2.medianBlur(maskl, 3)
    #cv2.imshow("cropped", maskl)
    #cv2.imwrite("filenamel.jpg", maskl)
    textl_loc = pytesseract.image_to_string(maskl)
    
else:
    x11=1700
    y11=0
    x22=4200
    y22=290
    dim = (1500,300)
    crop_img1 = image_np[y11:y22, x11:x22]
    resized = cv2.resize(crop_img1, dim, interpolation = cv2.INTER_AREA)#x1 y1 top left vertex
    mask = cv2.inRange(resized, lower, upper)
    mask = cv2.resize(mask, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("cropped", mask)
    #cv2.imwrite("filename.jpg", mask)
    text_time = pytesseract.image_to_string(mask)
    #print(text)
    
    
    
    x11l=1700
    y11l=290
    x22l=4200
    y22l=1000
    diml = (700,300)
    crop_img1l = image_np[y11l:y22l, x11l:x22l]
    resizedl = cv2.resize(crop_img1l, diml, interpolation = cv2.INTER_AREA)#x1 y1 top left vertex
    maskl = cv2.inRange(resizedl, lower, upper)
    maskl = cv2.resize(maskl, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    maskl = cv2.medianBlur(maskl, 3)
    #cv2.imshow("cropped", maskl)
    #cv2.imwrite("filename.jpg", maskl)
    textl_loc = pytesseract.image_to_string(maskl)
    #print(textl)



    
if (out == 0):

    print("Shelf is EMPTY")
    print("Finding name and price of the item from the shelf...")
    # title of our window
    #title = "out"
    # # Model preparation 
    PATH_TO_FROZEN_GRAPH = './FINAL_FILES/frozen_inference_graph_item.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './FINAL_FILES/labelmap_item.pbtxt'
    NUM_CLASSES = 1
    # ## Load a (frozen) Tensorflow model into memory.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    # # Detection
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        #while True:
          # Get raw pixels from the screen, save it to a Numpy array
          image_np = cv2.imread(test_img_Location)
          y = np.size(image_np, 0)
          x = np.size(image_np, 1)
          #cv2.imshow('image_np',image_np)
          #image_np = cv2.resize(image_np,(800,640))
          # To get real color we do this:
          image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Visualization of the results of a detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=3)
          
          total_final=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          # Show image with detection
          #cv2.imshow(title,total_final )
          
        
        
                  
          
          ymin = int((boxes[0][0][0]*y))
          xmin = int((boxes[0][0][1]*x))
          ymax = int((boxes[0][0][2]*y))
          xmax = int((boxes[0][0][3]*x))
          #print(scores[0])
    
          Result = np.array(total_final[ymin:ymax,xmin:xmax])
          #in_range = cv2.inRange(total_final,(0,127,0),(0,255,127))
          #cv2.imshow('in_range',total_final[ymin:ymax,xmin:xmax])
          mask = cv2.resize(total_final[ymin:ymax,xmin:xmax], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
          mask = cv2.medianBlur(mask, 7)
          lower_black = np.array([0,0,0]) #example value
          upper_black = np.array([105,105,105]) #example value
          mask = cv2.inRange(mask, lower_black, upper_black)
         #mask = cv2.inRange(total_final[ymin:ymax,xmin:xmax], lower_black, upper_black)
          #mask = cv2.cvtColor(maskc,cv2.COLOR_BGR2GRAY)
          #cv2.imshow('in',mask)
          text1 = pytesseract.image_to_string(mask, lang="eng")
          #print(text1)
          
          #text1b = pytesseract.image_to_string(mask, lang="eng")
          
    # # Model preparation 
    PATH_TO_FROZEN_GRAPH = './FINAL_FILES/frozen_inference_graph_price.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './FINAL_FILES/labelmap_price.pbtxt'
    NUM_CLASSES = 1
        # ## Load a (frozen) Tensorflow model into memory.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    
    # # Detection
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        #while True:
          # Get raw pixels from the screen, save it to a Numpy array
          image_np = cv2.imread(test_img_Location)
          y = np.size(image_np, 0)
          x = np.size(image_np, 1)
          #cv2.imshow('image_np',image_np)
          #image_np = cv2.resize(image_np,(800,640))
          # To get real color we do this:
          image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Visualization of the results of a detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=3)
          
          total_final=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          # Show image with detection
          #cv2.imshow(title,total_final )
          ymin = int((boxes[0][0][0]*y))
          xmin = int((boxes[0][0][1]*x))
          ymax = int((boxes[0][0][2]*y))
          xmax = int((boxes[0][0][3]*x))
          #print(scores[0])
    
          Result = np.array(total_final[ymin:ymax,xmin:xmax])
          #in_range = cv2.inRange(total_final,(0,127,0),(0,255,127))
          #cv2.imshow('in_range',total_final[ymin:ymax,xmin:xmax])
          mask = total_final[ymin:ymax,xmin:xmax]
         #mask = cv2.resize(mask,(750,300))
          mask = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
          mask = cv2.medianBlur(mask, 5)
          lower_black = np.array([0,0,0]) #example value
          upper_black = np.array([135,135,135]) #example value
          mask = cv2.inRange(mask, lower_black, upper_black)
          #mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
          cv2.imshow('price',mask)
          #cv2.imwrite('temp',total_final[ymin:ymax,xmin:xmax])
          text = pytesseract.image_to_string(mask, lang="eng")
          #print(text)
          message = "ITEM:" + text1 + ' of Price ' + text + ' $ is missing from shelf.'+ '\n' + 'Location: '+ '\n' + textl_loc + '\n' + '\n'+ 'Time:'+ '\n' + text_time
          msg = message
          print(message)
          server = smtplib.SMTP('smtp.gmail.com:587')
          server.ehlo()
          server.starttls()
          server.login(login_mail,login_password)
          server.sendmail(login_mail,to_mail, msg)
          server.quit()
          print("Success: Email sent!")
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          
if (out == 1):

    print("Shelf is NOT EMPTY") 
    print("Finding name and price of the item from the shelf...")
    # title of our window
    #title = "out"
    # # Model preparation 
    PATH_TO_FROZEN_GRAPH = './FINAL_FILES/frozen_inference_graph_item.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './FINAL_FILES/labelmap_item.pbtxt'
    NUM_CLASSES = 1
    # ## Load a (frozen) Tensorflow model into memory.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    # # Detection
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        #while True:
          # Get raw pixels from the screen, save it to a Numpy array
          image_np = cv2.imread(test_img_Location)
          y = np.size(image_np, 0)
          x = np.size(image_np, 1)
          #cv2.imshow('image_np',image_np)
          #image_np = cv2.resize(image_np,(800,640))
          # To get real color we do this:
          image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Visualization of the results of a detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=3)
          
          total_final=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          # Show image with detection
          #cv2.imshow(title,total_final )
          
        
        
                  
          
          ymin = int((boxes[0][0][0]*y))
          xmin = int((boxes[0][0][1]*x))
          ymax = int((boxes[0][0][2]*y))
          xmax = int((boxes[0][0][3]*x))
          #print(scores[0])
    
          Result = np.array(total_final[ymin:ymax,xmin:xmax])
          #in_range = cv2.inRange(total_final,(0,127,0),(0,255,127))
          #cv2.imshow('in_range',total_final[ymin:ymax,xmin:xmax])
          maskc = cv2.resize(total_final[ymin:ymax,xmin:xmax], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
          mask = cv2.medianBlur(maskc, 7)
          lower_black = np.array([0,0,0]) #example value
          upper_black = np.array([125,125,125]) #example value
          mask = cv2.inRange(mask, lower_black, upper_black)
          mask = cv2.inRange(total_final[ymin:ymax,xmin:xmax], lower_black, upper_black)
          
          #cv2.imshow('in',mask)
          text1 = pytesseract.image_to_string(mask, lang="eng")
          #print(text1)
          #mask = cv2.cvtColor(maskc,cv2.COLOR_BGR2GRAY)
          #text1b = pytesseract.image_to_string(mask, lang="eng")
          
    # # Model preparation 
    PATH_TO_FROZEN_GRAPH = './FINAL_FILES/frozen_inference_graph_price.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './FINAL_FILES/labelmap_price.pbtxt'
    NUM_CLASSES = 1
        # ## Load a (frozen) Tensorflow model into memory.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    
    # # Detection
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        #while True:
          # Get raw pixels from the screen, save it to a Numpy array
          image_np = cv2.imread(test_img_Location)
          y = np.size(image_np, 0)
          x = np.size(image_np, 1)
          #cv2.imshow('image_np',image_np)
          #image_np = cv2.resize(image_np,(800,640))
          # To get real color we do this:
          image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Visualization of the results of a detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=3)
          
          total_final=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          # Show image with detection
          #cv2.imshow(title,total_final )
          ymin = int((boxes[0][0][0]*y))
          xmin = int((boxes[0][0][1]*x))
          ymax = int((boxes[0][0][2]*y))
          xmax = int((boxes[0][0][3]*x))
          #print(scores[0])
    
          Result = np.array(total_final[ymin:ymax,xmin:xmax])
          #in_range = cv2.inRange(total_final,(0,127,0),(0,255,127))
          #cv2.imshow('in_range',total_final[ymin:ymax,xmin:xmax])
          mask = total_final[ymin:ymax,xmin:xmax]
         #ask = cv2.resize(mask,(350,140))
          mask = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
          mask = cv2.medianBlur(mask, 5)
          lower_black = np.array([0,0,0]) #example value
          upper_black = np.array([125,125,125]) #example value
          mask = cv2.inRange(mask, lower_black, upper_black)
          #mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
          #cv2.imshow('price',mask)
          #cv2.imwrite('temp',total_final[ymin:ymax,xmin:xmax])
          text = pytesseract.image_to_string(mask, lang="eng")
          #print(text)
          message = "ITEM:" + text1 +  ' of Price ' + text + ' $ is prsent in the shelf.'+ '\n' + 'Location: '+ '\n' + textl_loc + '\n' + '\n'+ 'Time:'+ '\n' + text_time
          msg = message
          print(message)
          server = smtplib.SMTP('smtp.gmail.com:587')
          server.ehlo()
          server.starttls()
          server.login(login_mail,login_password)
          server.sendmail(login_mail,to_mail, message)
          server.quit()
          print("Success: Email sent!")
          cv2.waitKey(0)
          cv2.destroyAllWindows()
                
    
