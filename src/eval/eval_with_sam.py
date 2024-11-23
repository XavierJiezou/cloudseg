import numpy as np
import torch
import matplotlib.pyplot as plt
import requests
import os
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from src.metrics.metric import IoUMetric
from src.data.components.hrc_whu import HRC_WHU


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   


class EvalWithSAM:
    def __init__(self, sam_checkpoint="sam_vit_h.pth", dataset_name="hrc_whu", device='cuda'):
        self.device = device
        self.sam =  self.load_sam(sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
    
    def download_checkpoint(self, sam_checkpoint):
        weights_url = {
            'sam_vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth', # ViT-Large
            'sam_vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', # ViT-Base
            'sam_vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', # ViT-Huge, default
        }
        
        if not os.path.exists(sam_checkpoint):
            url = weights_url[sam_checkpoint]
            r = requests
            r = requests.get(url, stream=True)
            with open(sam_checkpoint, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                print(f"Downloaded {sam_checkpoint}")
        else:
            print(f"{sam_checkpoint} already exists.")
        
    
    def load_sam(self, sam_checkpoint):
        model_type = "_".join(sam_checkpoint.split("_")[1:3])
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(self.device)
        return sam

    def prepare_image(self, image, transform):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=self.device) 
        return image.permute(2, 0, 1).contiguous()
    
    def prepare_point(self, gt, point_count=1, point_label=1):
        # return torch.tensor([[point1_x, point1_y], [point2_x, point2_y]], device=self.device)
        h, w = gt.shape
        half = point_count // 2
        pos_points = np.argwhere(gt == 1)
        neg_points = np.argwhere(gt == 0)
        np.random.seed(42)
        np.random.shuffle(pos_points)
        np.random.shuffle(neg_points)
        pos_points = pos_points[:half]
        neg_points = neg_points[:point_count - half]
        point_coords = torch.tensor(np.vstack((pos_points, neg_points)), device=self.device)
        point_labels = torch.tensor([1]*half+[0]*(point_count - half), device=self.device)
        return point_coords, point_labels
        
    
    def inference(self, point_count=10, point_label=1):
        import albumentations as albu
        resize_transform = ResizeLongestSide(1024)
        metrics = IoUMetric(2, iou_metrics=["mIoU", "mDice", "mFscore"])
        for sample in HRC_WHU(phase='test', all_transform=albu.Resize(256, 256)):
            image = sample['img'] # HxWxC, uint8
            mask = sample['ann'] # HxW, int64
            
            point_coords, point_labels = self.prepare_point(mask, point_count, point_label)
            
            # print(point_coords.shape, point_labels.shape)
            
            # print(point_coords)
            
            # print(point_labels)
            
            input = [
                {
                    'image': self.prepare_image(image, resize_transform),
                    'point_coords': resize_transform.apply_coords_torch(point_coords, image.shape[:2]).unsqueeze(1),
                    'point_labels': point_labels.unsqueeze(1),
                    'original_size': image.shape[:2]
                },
            ]
            
            output = self.sam(input, multimask_output=False)
            
            # print(output[0]['masks'].shape, output[0]['masks'].dtype) # torch.Size([1, 1, 256, 256]) torch.bool
            
            # print(output[0]['masks'])
            # break
            # preds shape must be HxW, and dtype must be int64
            masks, scores, logits = output[0]['masks'], output[0]['iou_predictions'], output[0]['low_res_logits']
            # print(np.argmax(scores.cpu()))
            
            mask_input = logits[np.argmax(scores.cpu()), :, :]
            masks = masks[np.argmax(scores.cpu()), :, :]
            
            # print(masks.shape)
            
            # optimizer
            # input = [
            #     {
            #         'image': self.prepare_image(image, resize_transform),
            #         'point_coords': resize_transform.apply_coords_torch(image_point, image.shape[:2]).unsqueeze(1),
            #         'point_labels': torch.tensor([point_label]*point_count, device=self.device).unsqueeze(1),
            #         'mask_input': mask_input[None, :, :],
            #         'original_size': image.shape[:2]
            #     },
            # ]
            
            # output = self.sam(input, multimask_output=False)
            
            # masks, scores, logits = output[0]['masks'], output[0]['iou_predictions'], output[0]['low_res_logits']
            # masks = masks[np.argmax(scores.cpu()), :, :]
            
            
            preds = masks.squeeze(0).cpu().numpy().astype(np.int64) # HxW, int64
            # to one-hot: 2xHxW, torch.int64
            preds = torch.nn.functional.one_hot(torch.tensor(preds), num_classes=2).permute(2, 0, 1).contiguous()
            mask = torch.nn.functional.one_hot(torch.tensor(mask), num_classes=2).permute(2, 0, 1).contiguous()
            metrics.results.append(IoUMetric.intersect_and_union(preds, mask, num_classes=2, ignore_index=255))
            
            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # plt.imshow(image)
            # for mask in output[0]['masks']:
            #     show_mask(mask.cpu().numpy(), ax, random_color=True)
            # for point, label in zip(point_coords, point_labels):
            #     show_points(point.cpu().numpy(), label.cpu().numpy(), ax)
            # ax.axis('off')
            # plt.tight_layout()
            # plt.savefig('output.png')
            # import time
            # time.sleep(5)
            
        result = metrics.compute_metrics(metrics.results)
        print(result)
        
            
if __name__ == '__main__':
    EvalWithSAM(
        sam_checkpoint='sam_vit_h_4b8939.pth',
        dataset_name='hrc_whu',
        device='cuda'
    ).inference()