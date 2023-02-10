'''
Hadamard行列-Interpolationのモデルの評価用のコード
'''
from config_interpolation import opt
import torch as t 
import numpy as np 
import os 
from models.MCNet import MCNet
from models.MCNet import IPM
import utils 
import math 
from skimage.measure import compare_psnr 
from skimage.measure import compare_ssim
import time
import torchsnooper
import cv2
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import torchvision

@t.no_grad()
def test(**kwargs):
    opt._parse(kwargs)
    
    #変えなければ行けない所
    # {32:0.125, 64:0.25, 96:0.375, 128:0.50, 160:0.625, 192:0.75}
    cr = 0.375 #here
    reconstructed_videos_pth = "reconstructed_frames/0-375/0-375.mat" #here
    M = int(opt.cr*opt.blk_size*opt.blk_size)
    MM = int(cr*opt.blk_size*opt.blk_size)
    
    save_test_root = opt.save_test_root
    if not os.path.exists(save_test_root):
        os.makedirs(save_test_root)
    log_file = open(save_test_root+"/result.txt",mode='w') # 指定したファイルを書き込みモードで読み込み

    os.environ['CUDA_VISIBLE_DEVICES'] = '1' #GPUをセットする。

    bernoulli_weights = scipy.io.loadmat('weights/hadamard.mat')
    bernoulli_weights = bernoulli_weights['x']
    bernoulli_weights =  bernoulli_weights[:int(opt.blk_size*opt.blk_size*opt.cr),:] #0.50 (if SR=0.50)
    bernoulli_weights = t.from_numpy(bernoulli_weights).float().to(opt.device)
    
    bernoulli_weights2 = scipy.io.loadmat('weights/hadamard.mat')
    bernoulli_weights2 = bernoulli_weights2['x']
    bernoulli_weights2 = t.from_numpy(bernoulli_weights2).float().to(opt.device)
    weights = bernoulli_weights2[:192,:]
    kernels = t.zeros(192, 1, opt.blk_size, opt.blk_size).to(opt.device)
    for i in range(192):
        kernel = weights[i,:]
        kernel = kernel.reshape(opt.blk_size, opt.blk_size)
        kernels[i,:,:,:] = kernel
    
    ipm = IPM()
    model = MCNet(bernoulli_weights,opt.cr,opt.blk_size,opt.ref_size).eval()
    if opt.load_ipm_model_path:
        ipm.load(opt.load_ipm_model_path)
        print(f"finish loading model_path : {opt.load_model_path}")
    if opt.load_model_path:
        model.load(opt.load_model_path)
        print(f"finish loading model_path : {opt.load_model_path}")
    ipm.to(opt.device)
    model.to(opt.device)
    ipm.eval() 
    model.eval()
    
    videos = [os.path.join(opt.test_data_root,video) for video in os.listdir(opt.test_data_root)]
    video_num = len(videos)
    print("total test video number:",video_num) #5

    end = time.time()
    psnr_av = 0
    ssim_av = 0
    time_av = 0
    count = 0
    iiii = 0
    jjjj = 0
    for item in videos:
        if (item.split(".")[-1]!='avi'):
            continue
        print("now is processing:",item)
        reconstructed_videos = scipy.io.loadmat(reconstructed_videos_pth)
        if count==0:
            reconstructed_frames = reconstructed_videos['v_Typing_g01_c01']
        elif count==1:
            reconstructed_frames = reconstructed_videos['v_ApplyEyeMakeup_g03_c03']
        elif count==2:
            reconstructed_frames = reconstructed_videos['v_Archery_g21_c02']
        elif count==3:
            reconstructed_frames = reconstructed_videos['v_Knitting_g01_c01']
        else:
            reconstructed_frames = reconstructed_videos['v_PlayingCello_g05_c01']
        
        log_file.write("%s"%item) #item -> "./data/test/v_Typing_g01_c01.avi"
        log_file.write("\n")

        uv = utils.Video(opt.height,opt.width) #(height, width) = (160, 160)
        test_data = uv.video2array(item,opt.frame_num) #item -> "./data/test/v_Typing_g01_c01.avi", opt.frame_num -> 180

        test_data_t = t.from_numpy(test_data).float().to(opt.device)
        result_data_t = t.zeros_like(test_data_t).cuda()

        psnr_total = 0
        ssim_total = 0
        frame_cnt = 0
        
        #do test on every video
        for i in range(test_data_t.size(0)):
            for j in range(test_data_t.size(1)):
                frames = test_data_t[i,j,:,:,:] #torch(180, 160, 160)
                frames_num = frames.size(0) #180
                result_frame = t.ones(1,frames[0].size(0),frames[0].size(1)).float().to(opt.device) #(1, 160, 160)
                result_frames = t.zeros(frames_num,frames[0].size(0),frames[0].size(1)).to(opt.device) #(180, 160, 160)
                frames_t = frames #torch(180, 160, 160)
                #blk_size -> 16
                x_b = uv.frame_unfold(frames_t,opt.blk_size,int(opt.blk_size)).to(opt.device)
                #CNNに入力できる状態にする
                #x_b tensor(frame_num, block_num_h, block_num_w, block_height, block_width)
                #print(x_b.shape)
                #(180, 10, 10, 16, 16)
                blk_num_h = x_b.size(1) #10
                blk_num_w = x_b.size(2) #10
                #print(x_b.size(3)) #16
                #print(x_b.size(4)) #16
        
                for ii in range(frames_num):                    
                    x_ref_b = uv.frame_unfold(result_frame,opt.ref_size,int(opt.ref_size/2)) #(1, 5, 5, 32, 32)
                    result_b = t.zeros_like(x_b[0].unsqueeze_(0)) #(1, 10, 10, 16, 16)
                    
                    reconstructed_frame = reconstructed_frames[:,:,:,ii]
                    reconstructed_frame = reconstructed_frame.reshape(1, 1, 160, 160)
                    reconstructed_frame = t.from_numpy(reconstructed_frame[:,:,:,:]/255.0).float().to(opt.device)
                    
                    input_ = (x_b[ii,:,:,:,:]/255.0).float().to(opt.device)
                    input_imgs = t.zeros((1, 1, 160, 160)).to(opt.device)
                    for iii in range(input_.size(0)):
                        for jjj in range(input_.size(1)):
                            input_imgs[0, 0, iii*opt.blk_size: (iii+1)*opt.blk_size, jjj*opt.blk_size: (jjj+1)*opt.blk_size] = input_[iii, jjj, :, :]
                    '''
                    if(ii==50 and count==3):
                        input_pil = torchvision.transforms.functional.to_pil_image(input_imgs.view(160,160))
                        print("aaaaaaaaaaaa")
                        input_pil.save("input_img.png")
                    '''
                    
                    ref = x_ref_b.repeat(1,2,1,1,1) #(1, 10, 5, 32, 32)
                    ref[:,[0,1,2,3,4,5,6,7,8,9],:,:,:] = ref[:,[0,9,1,2,3,4,5,6,7,8],:,:,:]
                    ref = ref.repeat(1,1,2,1,1)
                    ref[:,:,[0,1,2,3,4,5,6,7,8,9],:,:] = ref[:,:,[0,9,1,2,3,4,5,6,7,8],:,:]

                    #print(ref.shape)
                    ref = ref.view(1*blk_num_h*blk_num_w,opt.ref_size,opt.ref_size)
                    
                    imgs_lower = F.conv2d(input_imgs, kernels, stride=16, padding=0)
                    imgs_upper = ipm(reconstructed_frame) #input_imgsをReconstructed imgsに変更すれば良い。
                    interpolated_imgs = t.zeros((1, int(opt.blk_size*opt.blk_size*opt.cr), 10, 10))
                    interpolated_imgs[0, :MM, :, :] = imgs_lower[0,:MM,:,:]
                    interpolated_imgs[0,MM:, :, :] = imgs_upper[0,MM:M,:,:]
                    #print(interpolated_imgs.shape)  #(1, 128, 10 10)             
                    
                    
                    #imgs -> measurement(100, M)に変換 (today)
                    #1. interpolated_imgs(1, 128, 10, 10) -> (1, 128, 100)
                    trans_interpolated_imgs = interpolated_imgs.view((1,int(opt.blk_size*opt.blk_size*opt.cr),int(interpolated_imgs.size(2)*interpolated_imgs.size(2))))
                    interpolated_m = t.zeros((int(interpolated_imgs.size(2)*interpolated_imgs.size(2)), int(opt.blk_size*opt.blk_size*opt.cr))).to(opt.device)
                    
                    for iii in range(int(interpolated_imgs.size(2)*interpolated_imgs.size(2))):
                        for jjj in range(int(opt.blk_size*opt.blk_size*opt.cr)):
                            interpolated_m[iii, jjj] = trans_interpolated_imgs[0, jjj, iii]
                    
                    if(opt.noise_snr>0):
                        interpolated_m = add_noise(interpolated_m,opt.noise_snr,10)
                    if ii==1:
                        print(interpolated_m[1,:])
                        #print(interpolated_imgs[0, 0, :,:])
                        #print(interpolated_imgs[0, 1, :,:])
                        
                    output,_ = model(interpolated_m,ref,interpolated_m)
                    result_b = output.view(1,blk_num_h,blk_num_w,opt.blk_size,opt.blk_size)
                    #(1, 10, 10, 16, 16)
                    #print(result_b.shape)

                    frame_cnt = frame_cnt + 1
                    result_frame = uv.frame_fold(result_b,opt.blk_size,int(opt.blk_size/2))
                    
                    
                    #print(f"result_frame -> {result_frame.shape}") # (1, 160, 160)
                 
                    result_frames[ii] = result_frame
                    psnr = compare_psnr((frames_t[ii].unsqueeze(0)).cpu().numpy(),(result_frame*255).cpu().numpy(),data_range=255)
                    ssim = compare_ssim(frames_t[ii].cpu().numpy(),(result_frame*255).squeeze(0).cpu().numpy())
                    if(ii==30 and count==3):
                        output_name = str(cr) + "_" + str(opt.cr) + "kitty" + ".png"
                        output_pil = torchvision.transforms.functional.to_pil_image(result_frame.view(160,160))
                        output_pil.save(output_name)
                        print("#################################")
                        print(f"PSNR -> {psnr}")
                        print(f"SSIM -> {ssim}")
                        print("#################################")
                    
                    psnr_total = psnr_total + psnr
                    ssim_total = ssim_total + ssim
                
                result_data_t[i,j,:,:,:] = result_frames

        uv.array2video(result_data_t,opt.save_test_root)

        #get log information
        video_time = time.time() - end
        info = str(psnr_total/frame_cnt)+"\n"
        log_file.write("%s"%info)
        info = str(ssim_total/frame_cnt)+"\n"
        log_file.write("%s"%info)
        info = str(video_time/frame_cnt)+"\n"
        log_file.write("%s"%info)
        end = time.time()

        print("PSNR is:",psnr_total/frame_cnt,"SSIM is:",ssim_total/frame_cnt,"Time per frame is:",video_time/frame_cnt)
        psnr_av = psnr_av + psnr_total/frame_cnt
        ssim_av = ssim_av + ssim_total/frame_cnt
        time_av = time_av + video_time/frame_cnt
        count+=1
    log_file.close()
    print("Average PSNR is:",psnr_av/video_num,"Average SSIM is:",ssim_av/video_num,"Average Time per frame is:",time_av/video_num)

def add_noise(input,SNR,seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True

    input_np = input.cpu().numpy()
    noise = np.random.randn(input_np.shape[0],input_np.shape[1]).astype(np.float32)
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(input_np)**2/(input_np.size)
    noise_power = signal_power/np.power(10,(SNR/10))
    noise = (np.sqrt(noise_power)/np.std(noise))*noise

    y = input_np + noise 
    y = t.from_numpy(y).cuda()
    return y

if __name__=='__main__':
    import fire
    fire.Fire()