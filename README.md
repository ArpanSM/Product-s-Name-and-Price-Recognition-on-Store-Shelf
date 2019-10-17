# Product-s-Name-and-Price-Recognition-on-Store-Shelf
High accuracy image classification model was trained to identify whether a shelf is empty or not. Tensorflow Object Detection, PyTesseract and image processing was used to recognise the product's name and price which was sent through email.

FINSL_FILES folder is available through this drive link
https://drive.google.com/open?id=1i1ht7K9fm6nRS3q25Ybwa8OJAaIGTFrR

ML_TEST

Steps to Run 
1.	Copy “FINAL_FILES” and “FINAL.py” into the Python Object Detection Folder. (generally it is “.\Anaconda3\Lib\site-packages\object_detection....”)
2.	Open FINAL.py . In the input section, enter the test image location in  “test_img_Location ”
3.	Enter ‘login_mail’ – your mail id ,’ login_password’ – your mail id’s password and ‘to_mail’-the receiver’s mail id.
4.	Enter the location of tesseract.exe file. (could be ‘C:\Program Files\Tesseract\\tesseract.exe’)
5.	That’s it.Run the python command to get the results.



Method
1.Image Segmentaion for Empty vs non empty class was done using Transfer Learning. Val_acc of  88% was achieved 

2.Tensorflow Object Detection was used to detect the ITEM_TAG and PRICE_TAG. Accuracy of these    models were excellent with above 98% correct segmentation.

3.Image resize and cropping techniques were used to extract the TIME and LOCATION segment from the image.

4.Image processing techniques such as Masking,Blurring,Erosion and Dilation were used to improve the clarity for text recognisation.

5.Google’s PyTesseract Model was used to identify text from  ITEM_TAG, PRICE_TAG ,TIME and LOCATION segments. Python’s “smptlib” library was used to send the mails.


