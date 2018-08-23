import cv2


image_path = "/home/eslam/2101.jpg"
image = cv2.imread(image_path)
xmin = 0.5044203*1600
ymin = 0.421273260*1600
xmax = 0.57788480*1600
ymax = 0.47667771*1600
cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 255), 3)
cv2.rectangle(image, (int(0.586918830*1600), int(0.235731420*1600)), (int(0.655723330*1600), int(0.286894441*1600)), (255, 0, 255), 3)
cv2.rectangle(image, (int(0.677062630*1600), int(0.232882260*1600)), (int(0.7442920*1600), int(0.281310080*1600)), (0, 0, 255), 3)
cv2.rectangle(image, (int(0.179971580*1600), int(0.249964190*1600)), (int(0.245431720*1600), int(0.29740530*1600)), (0, 0, 255), 3)
cv2.rectangle(image, (int(0.6763370*1600), int(0.234253350*1600)), (int(0.74438560*1600), int(0.28696650*1600)), (0, 255, 255), 3)
cv2.imwrite(image_path[:-4] + '_detected_' + image_path[-4:], image)