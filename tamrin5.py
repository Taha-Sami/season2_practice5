import numpy as np
import cv2

##second part of this practice

kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
img=cv2.imread("00tennisballs1-superJumbo.JPG")
img=cv2.resize(img,None,fx=0.6,fy=0.6)
img_show=img.copy()
img_show2=img.copy()

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(7,7),0)
sharp=cv2.filter2D(gray,-1,kernel)
#edges=cv2.Canny(blur,100,200)
circles=cv2.HoughCircles(sharp,cv2.HOUGH_GRADIENT,1.5,200,param1=57,param2=47,minRadius=50,maxRadius=64)
j=0
if circles is not None :
	circles=circles[0].astype(np.uint32)
	for circle in circles :
		cv2.circle(img_show,(circle[0],circle[1]),circle[2],(255,0,0),2)
		j+=1
print(f'There are {j} balls in this image' )
cv2.imshow("img",img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()

##second part of this practice

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
red_mask1 = cv2.inRange(hsv, lower_red, upper_red)
lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])
red_mask2 = cv2.inRange(hsv, lower_red, upper_red)
red_mask = red_mask1 + red_mask2
red_img = cv2.bitwise_and(img, img, mask=red_mask)
sharp=cv2.filter2D(red_img,-1,kernel)

gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=1000,
                           param1=45, param2=34, minRadius=50, maxRadius=60)

# Filter out non-red circles
red_circles = []
if circles is not None :
	circles=circles[0].astype(np.uint32)
	for circle in circles :
		cv2.circle(img_show2,(circle[0],circle[1]),circle[2],(255,0,0),5)
		print(f'the x location of red ball is {circle[0]}, the y location of red ball is {circle[1]}, and also the radius of red ball is {circle[2]} ')
		



# Display result
cv2.imshow('Result', img_show2)
cv2.waitKey(0)
cv2.destroyAllWindows()