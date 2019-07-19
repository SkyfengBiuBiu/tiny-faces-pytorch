import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import trainer
from datasets import get_dataloader
from models.model import DetectionModel
from utils import visualize
from utils.online_tubes import VideoPostProcessor
import matplotlib.pyplot as plt
import cv2
from os.path import isfile, join
import glob

COLOR_WHEEL = ('red', 'blue', 'brown', 'darkblue', 'green',
               'darkgreen', 'brown', 'coral', 'crimson', 'cyan',
               'fuchsia', 'gold', 'indigo', 'red', 'lightblue',
               'lightgreen', 'lime', 'magenta', 'maroon', 'navy',
               'olive', 'orange', 'orangered', 'orchid', 'plum',
               'purple', 'tan', 'teal', 'tomato', 'violet')

def arguments():
    
    parser = argparse.ArgumentParser("Model Evaluator")
    parser.add_argument("dataset")
    parser.add_argument("--split", default="demo")
    parser.add_argument("--dataset-root")
    parser.add_argument("--checkpoint",
                        help="The path to the model checkpoint", default="")
    parser.add_argument("--prob_thresh", type=float, default=0.03)
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def dataloader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    val_loader, templates = get_dataloader(args.dataset, args,
                                           train=False, split=args.split,
                                           img_transforms=val_transforms)
    return val_loader, templates


def get_model(checkpoint=None, num_templates=25):
    model = DetectionModel(num_templates=num_templates)
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model"])
    return model


def write_results(dets, img_path, split):
    results_dir = "{0}_results".format(split)

    if not osp.exists(results_dir):
        os.makedirs(results_dir)

    filename = osp.join(results_dir, img_path.replace('jpg', 'txt'))
    file_dir = os.path.dirname(filename)
    if not osp.exists(file_dir):
        os.makedirs(file_dir)

    with open(filename, 'w') as f:
        f.write(img_path.split('/')[-1] + "\n")
        f.write(str(dets.shape[0]) + "\n")

        for x in dets:
            left, top = np.round(x[0]), np.round(x[1])
            width, height, score = np.round(x[2]-x[0]+1), np.round(x[3]-x[1]+1), x[4]
            d = "{0} {1} {2} {3} {4}\n".format(int(left), int(top),
                                               int(width), int(height), score)
            f.write(d)


def run(model, val_loader, templates, prob_thresh, nms_thresh, device, split, debug=False):
    num_images=len(val_loader)
    target_dets=np.zeros(shape=(num_images,2,100,4))
    target_scores=np.zeros(shape=(num_images,2,100,2))
    images=[]
    for idx, (img, filename) in enumerate(val_loader):
        print("idx",idx)
        dets = trainer.get_detections(model, img, templates, val_loader.dataset.rf,
                                      val_loader.dataset.transforms, prob_thresh,
                                      nms_thresh, device=device)
        

        
        #dets=dets[dets[:,-1].argsort()]
        dets=dets[dets[:,-1]>0.6]
        print("number of dets:",dets.shape[0])
        num_box=dets.shape[0]
        
        target_det=dets[:,0:4]
        target_score=dets[:,-1]

        target_dets[idx,1,:num_box,:]=target_det
        target_scores[idx,1,:num_box,1]=target_score
        target_scores[idx,1,:num_box,0]=1-target_score
#        if target_score.any():
#            target_scores[idx,1,:num_box,1]=target_score
#            target_scores[idx,1,:num_box,0]=np.max(1-target_score,0)
#        else:
#            target_scores[idx,1,:num_box,1]=0
#            target_scores[idx,1,:num_box,0]=1
        
        if idx!=0:
            target_dets[idx,0,:,:]=target_dets[idx-1,1,:,:]
            target_scores[idx,0,:,1]=target_scores[idx-1,1,:,1]
            target_scores[idx,0,:,0]=target_scores[idx-1,1,:,0]  
        else:
            target_dets[idx,0,:,:]=target_dets[idx,1,:,:]
            target_scores[idx,0,:,1]=target_scores[idx,1,:,0]
            target_scores[idx,0,:,0]=target_scores[idx,1,:,1]  
            
        if debug:
            print(img.shape)
            mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=img.device)
            std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=img.device)
            
            img=(img[0]).permute((1, 2, 0))
            print(img.shape)
            img=std*img+mean
            img=np.clip(img.numpy(),0,1)
            im = Image.fromarray((img*255).astype('uint8'), 'RGB')
            images.append(im)
            #visualize.visualize_bboxes(im, dets)

        #write_results(dets, filename[0], split)
    return images,target_dets,target_scores

def convert_frames_to_video(video_dataset, fps):
    print("Converting...")

    # define save dir
    output_dir = "/home/fengy/Documents/tiny-faces-pytorch/images"
    pathIn=output_dir+'/'
    pathOut=os.path.join(output_dir, 'test.avi')
    

    
    frame_array = []
    files = sorted(glob.glob(os.path.join(pathIn, "*.png")))
    
    for i in range(len(files)):
        filename=files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    
def visualize_with_paths(video_dataset, video_post_proc,imagenet_vid_classes):

    print("Visualizing...")

    # define save dir
    
    output_dir ="/home/fengy/Documents/tiny-faces-pytorch/images"
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    det_classes = imagenet_vid_classes

    num_frames = len(video_dataset)

    for i_frame in range(num_frames):
        print('frame: {}/{}'.format(i_frame, num_frames))
        fig, ax = plt.subplots(figsize=(12, 12))
        disp_image = video_dataset[i_frame]
        for i_pth, cls_ind in enumerate(video_post_proc.path_labels): # iterate over path labels
            cls_ind = int(cls_ind)
            ax.imshow(disp_image, aspect='equal')
            class_name = det_classes[cls_ind]
            path_starts =  video_post_proc.path_starts[i_pth]
            path_ends = video_post_proc.path_ends[i_pth]
            if i_frame >= path_starts and i_frame <= path_ends: # is this frame in the current path
                # bboxes for this class path
                bbox = video_post_proc.path_boxes[i_pth][i_frame-path_starts].cpu().numpy() 
                # scores for this class path
                score = video_post_proc.path_scores[i_pth][i_frame-path_starts].cpu().numpy() 
                
                ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor=COLOR_WHEEL[cls_ind], linewidth=3.5)
                        )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(class_name, score[0]),
                        bbox=dict(facecolor=COLOR_WHEEL[cls_ind], alpha=0.5),
                        fontsize=14, color='white')

        plt.axis('off')
        plt.tight_layout()
        #plt.show()
        im_save_name = os.path.join(output_dir,"%#09d.png" % (i_frame))
        print('Image with bboxes saved to {}'.format(im_save_name))
        plt.savefig(im_save_name)
        plt.clf()
        plt.close('all')

def main():
    args = arguments()
    images=[]
    dets=[]
    imagenet_vid_classes = ['__background__','person']
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    val_loader, templates = dataloader(args)
    num_templates = templates.shape[0]
    print("dataset", args.dataset)
    model = get_model(args.checkpoint, num_templates=num_templates)
    

    with torch.no_grad():
        # run model on val/test set and generate results files
        images, dets,scores=run(model, val_loader, templates, args.prob_thresh, args.nms_thresh, device, args.split,
            debug=args.debug)
        
        
    print("bbox of image",scores)
    vid_pred_boxes=torch.FloatTensor(dets)
    vid_scores=torch.FloatTensor(scores)
    vid_post_proc = VideoPostProcessor(vid_pred_boxes, vid_scores, imagenet_vid_classes)
    paths = vid_post_proc.class_paths(path_score_thresh=0.0)

    if vid_post_proc.path_total_score.numel() > 0:
           	  visualize_with_paths(images, vid_post_proc,imagenet_vid_classes)
           	  fps = 1
           	  convert_frames_to_video(images,fps)
    else:
           print("No object had been deteced!")

if __name__ == "__main__":
    main()
