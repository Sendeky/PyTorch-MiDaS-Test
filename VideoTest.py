import cv2
import torch
import matplotlib.pyplot as plt
import time


# get MiDaS model from torch.hub
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# get midas model from torch.hub
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# move model to gpu if avilable
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# midas transforms for images (model requires image of specific size/dimensions)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# load necessary transforms for model type
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform





# get the video stream
cap = cv2.VideoCapture("highway_test2.mp4")
# create a named window (to show the video)
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	


# while capture is opened
while cap.isOpened():
		try:
			# Read frame from the video
			ret, frame = cap.read()
		except:
			continue

		if ret:	
			# img = cv2.imread(frame)
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

			input_batch = transform(img).to(device)
			cv2.imshow("Detected lanes", img)
			
            # start time
			start_time = time.time()
			
			#
			with torch.no_grad():
				prediction = midas(input_batch)
				prediction = torch.nn.functional.interpolate(
					prediction.unsqueeze(1),
                    size=img.shape[:2], mode="bicubic",
		            align_corners=False, 
            ).squeeze()
				
			output = prediction.cpu().numpy()
			output2 = output.astype(float) / 255

			end_time = time.time()
			print("inference time: ", end_time - start_time)
			cv2.imshow("Detected depth", output2)
			print("output matrix: ", output2)
			# plt.imshow(output2)
			# plt.show()
			
		else:
			break

		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()