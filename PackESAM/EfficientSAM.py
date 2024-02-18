import os

ROOT_PATH = os.path.abspath(__file__)

from time import time
import numpy as np

import torch 
from torchvision import transforms


class EfficientSAM:
    def __init__(self, mode:int=0, weight:int=1, device:chr='cuda') -> None:
        """
        describe: init efficient SAM Model
        
        @param: mode:int default 0. 0 for onnx, 1 for jit
        @param: weight:int default 1. 0 for sam_vitt, 1 for sam_vits
        @param: device:chr default 'cuda'. When cuda is available, use input set, else for cpu
        
        return: None
        """
        
        if device not in ['cuda', 'cpu']:
            raise ValueError(f"unsupport device: {device}")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.mode = mode
        
        if self.mode == 0:
            import onnxruntime
            weight_map = {
                0: ROOT_PATH.replace("EfficientSAM.py", "weights") + "/efficientsam_ti.onnx",
                1: ROOT_PATH.replace("EfficientSAM.py", "weights") + "/efficientsam_s.onnx"
            }

            if device == 'cuda':
                provider = ['CUDAExecutionProvider']
            if device == 'cpu':
                provider = ['CPUExecutionProvider']

            self.model = onnxruntime.InferenceSession(
                weight_map[weight],
                providers=provider
            )
        
        if self.mode == 1:
            if self.device == 'cpu':
                weight_map = {
                    0: ROOT_PATH.replace("EfficientSAM.py", "weights") + "/efficientsam_ti_gpu.jit",
                    1: ROOT_PATH.replace("EfficientSAM.py", "weights") + "/efficientsam_s_gpu.jit"
                }
                self.model = torch.jit.load(weight_map[weight], map_location=self.device)
                self.model.eval()
                
            elif self.device == 'cuda':
                weight_map = {
                    0: ROOT_PATH.replace("EfficientSAM.py", "weights") + "/efficientsam_ti_cpu.jit",
                    1: ROOT_PATH.replace("EfficientSAM.py", "weights") + "/efficientsam_s_cpu.jit"
                }
                self.model = torch.jit.load(weight_map[weight], map_location=self.device)
                self.model.eval()
            else:
                raise ValueError(f"unsupport device: {self.device}")
                
                
    def set_pts(self, input_points:list=None, mode:int=0):
        """
        describe: set detect point
        
        @param: input_points:list default None. The list of detect points
        @param: mode:int default 0. 0 for point prompt, 1 for box prompt
        
        return: None 
        """
           
        if mode == 0:
            input_labels = [1 for _ in range(len(input_points))]
            self.input_points = [[input_points]]
            self.input_labels = [[input_labels]]
        elif mode == 1:
            self.input_points = [[input_points]]
            self.input_labels = [[[2, 3]]]
        else:
            raise ValueError(f"incorrect mode choice [{str(mode)}] for input points")
    
    def detect(self, image:np.ndarray=None):
        """
        describe: Get the Mask of the area where the coordinates 
        of the entered point are located
        
        @param: image:np.ndarray default None. Bgr image and ndarray format
        
        return:
        @r1: bool, True for set successful, False for failure
        @r2: list, Failure is None, success for Datapack
        Datapack: [image, predicted_logits_show, mask, masked_image_np]
        """
        
        self.start_time = time()
        
        if image is None:
            return False, None
        
        if self.mode == 0:
            
            input_image = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            
            predicted_logits, predicted_iou, predicted_lowres_logits = self.model.run(
                output_names=None,
                input_feed={
                    "batched_images": input_image,
                    "batched_point_coords": self.input_points,
                    "batched_point_labels": self.input_labels,
                },
            )
            
            origin_mask = predicted_logits[0, 0, 0, :, :] >= 0
            masked_image_np = image.copy().astype(np.uint8) * origin_mask[:,:,None]
            
            return True, [image, predicted_logits, origin_mask, masked_image_np]
        
        elif self.mode == 1:
            
            self.input_points = torch.tensor(self.input_points).to(self.device)
            self.input_labels = torch.tensor(self.input_labels).to(self.device)
            
            image_tensor = transforms.ToTensor()(image).to(self.device)
            predicted_logits, predicted_iou = self.model(
                image_tensor[None, ...],
                self.input_points,
                self.input_labels,
            )
            
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)

            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            
            origin_mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
            masked_image_np = image.copy().astype(np.uint8) * origin_mask[:,:,None]
            
            return True, [image, predicted_logits, origin_mask, masked_image_np]
        
        else:
            return False, None
    
    def get_process_time(self) -> float:
        """
        describe: Get the processing time
        
        return:
        @r1: float, time 
        """
        self.end_time = time()
        return self.end_time - self.start_time

    def __del__(self):
        print(f"========== start the final process ==========")
        try:
            del self.model
        except Exception as error:
            print(f"FastSAM Final ERROR: {error}")
        try:
            torch.empty()
        except Exception as error:
            print(f"FastSAM Final ERROR: {error}")
            