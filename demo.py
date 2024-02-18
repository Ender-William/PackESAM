import cv2
from PackESAM import EfficientSAM

ESAM = EfficientSAM(mode=0, weight=0, device='cuda')

image = cv2.imread("Test_Img_1.jpg")

ESAM.set_pts(input_points=[[320, 560]], mode=0)  # [[x, y]]

while True:
    ret, [image, predicted_logits, mask, masked_image_np] = ESAM.detect(image=image)

    # masked_image_np = ESAM.detect(image=image)
    print(f"FPS: {1/ESAM.get_process_time()}")
    cv2.imshow("Image", masked_image_np)
    cv2.waitKey(1)
    
cv2.destroyAllWindows()


