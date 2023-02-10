import cv2 
import numpy as np 
import os 
import imageio 
from torchvision import transforms as tf 
import torch as t 
import fire
import torchsnooper

class Video(object):
    def __init__(self,height,width):
        self.name = None
        self.height = height
        self.width = width
        self.blk_num_w = int(self.width/160) #1 if self.width=160
        self.blk_num_h = int(self.height/160) #2 if self.height=160
        self.input_patch_numel = None
        self.input_patch_size = None
        self.input_frame_numel = None 
        self.input_frame_size = None 
    
    def frame_unfold(self,frame_imgs_t,block,n=0):
        #input data type torch tensor
        #input data shape [frame_num,height,width]
        #output data type torch tensor
        #output data size [frame_num,block_num_h,block_num_w,block_width,block_width
        #print(f"frame_imgs_t.shape -> {frame_imgs_t.shape}")
        self.input_frame_size = frame_imgs_t.size() #tensor (180, 160, 160)
        self.input_frame_numel = frame_imgs_t.numel() #25600 = 160 * 160
        self.frame_h = frame_imgs_t.size(1) # 160
        self.frame_w = frame_imgs_t.size(2) # 160
        #print(self.frame_h)
        #print(self.frame_w)
        '''
        if((self.frame_h-block)%(block-n)!=0 or (self.frame_w-block)%(block-n)!=0):
            print("frame size is",self.input_frame_size,"error:size mismatch!")
            return 0
        else:
        '''
        #print(frame_imgs_t.unfold(1,block,block-n).shape)
        
        #output_ = frame_imgs_t.unfold(1,block,block-n).unfold(2,block,block-n) #(180, 19, 19, 16, 16)
        output_ = frame_imgs_t.unfold(1,block,block).unfold(2,block,block) #(180, 10, 10, 16, 16)
        #print(f"output_ -> {output_.shape}")
        output = output_.contiguous() # (180, 19, 19, 16, 16)
        #print(output.shape)
        self.input_patch_numel = output.numel()
        self.input_patch_size = output.size()
        return output

    #@torchsnooper.snoop()
    def frame_fold(self,input_patches,block,n=0):
        #input data type torch tensor
        #input data shape [frame_num,block_num_h,block_num_w,block_width,block_width]
        #output data type torch tensor
        #output data shape [frame_num,height,width]
        input_patches = input_patches.float()
        idx = t.zeros(self.input_frame_numel).long().to('cuda')
        t.arange(0,self.input_frame_numel,out=idx)
        idx = idx.view(self.input_frame_size)
        
        #idx_unfold = idx.unfold(1,block,block-n).unfold(2,block,block-n)
        idx_unfold = idx.unfold(1,block,block).unfold(2,block,block)
        idx_unfold = idx_unfold.contiguous().view(-1)

        video = t.zeros(self.input_frame_size).view(-1).to('cuda')
        video_ones = t.zeros(self.input_frame_size).view(-1).to('cuda')
        patches_ones = (t.zeros_like(input_patches)+1).view(-1).to('cuda')

        input_patches = input_patches.contiguous().view(-1)
        video.index_add_(0,idx_unfold,input_patches)
        video_ones.index_add_(0,idx_unfold,patches_ones)
        output = (video/video_ones).view(self.input_frame_size)
        return output

    def video2array(self,filename,frame_num):
        self.name = filename.split(".")[-2].split("/")[-1]

        self.blk_num_w = int(self.width/160) #1 if self.width=160
        self.blk_num_h = int(self.height/160) #2 if self.height=160
        blk_frames = np.zeros((self.blk_num_h,self.blk_num_w,frame_num,160,160))

        vc = cv2.VideoCapture(filename)
        frame_num_cv2=vc.get(7) # 127
        frame_index = 0
        
        if vc.isOpened():
            rval,frame = vc.read()
        else:
            rval = False

        if frame_num > frame_num_cv2:
            frame_num = frame_num_cv2-1

        while (rval and frame_index<frame_num):
            rval, frame = vc.read()
            frame_yt = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #print(frame_yt.shape) # (240, 320) 
            for j in range(self.blk_num_h):
                for k in range(self.blk_num_w):
                    #print(f"k -> {k}, j -> {j}")
                    #print(f"self.blk_num_h -> {self.blk_num_h}, self.blk_num_w -> {self.blk_num_w}") # (4, 8)
                    blk_frames[j,k,frame_index,:,:] = frame_yt[j*160:(j+1)*160,k*160:(k+1)*160]
            frame_index = frame_index + 1
            cv2.waitKey(1)
        vc.release()

        print("job done!")
        return blk_frames #(1, 2, 180, 160, 160)
    #@torchsnooper.snoop()
    def array2video(self,data,saveroot):
        data = data.cpu()
        if not os.path.exists(saveroot):
            os.makedirs(saveroot)
        frames_num = data.size(2)
        #frames_num = data.size(0)
        frames = t.zeros((frames_num,1*160,1*160))
        frames_np = np.zeros((frames_num,1*160,1*160))
        '''
        for i in range(self.blk_num_h):
            for j in range(self.blk_num_w):
                frames[:,i*160:(i+1)*160,j*160:(j+1)*160] = data[i,j,:,:,:]
        '''
        frames[:,:,:] = data[:,:,:]
        for i in range(frames.size(0)):
            frames_t = tf.ToPILImage()(frames[i])
            frames_np[i] = np.asarray(frames_t)
        frames_np = frames_np.astype(np.uint8)
        
        #savename = saveroot + "/" + self.name + ".avi"
        #print("save video path is:",savename)
        #imageio.mimwrite(savename,frames_np,format='avi',fps=30)
        return None

    
    