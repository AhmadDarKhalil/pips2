import time
import numpy as np
import saverloader
from nets.pips2 import Pips
import utils.improc
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys
import cv2
from pathlib import Path
from PIL import Image

def read_mp4(fn):
    vidcap = cv2.VideoCapture(fn)
    frames = []
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        frames.append(frame)
    vidcap.release()
    return frames

def compute_pairwise_distances(tensor1, tensor2):
    # Ensure that both input tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"

    # Get the number of frames and points
    _,num_frames, num_points, _ = tensor1.shape

    # Initialize an empty tensor to store distances
    distances = torch.zeros(num_frames, num_points)

    for i in range(num_frames):
        for j in range(num_points):
            # Compute the Euclidean distance between points at the same index in both tensors
            distance = torch.norm(tensor1[0,i, j] - tensor2[0,i, j], dim=-1)
            distances[i, j] = distance

    return distances

def run_model_forward_backward(model, rgbs, S_max=128, N=64, iters=16, sw=None):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B == 1)

    # Initialize points at the middle frame
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
    _, S, C, H, W = rgbs.shape

    # Initialize trajectories at the middle frame
    trajs_e_forward = xy0.unsqueeze(1).repeat(1, S, 1, 1)

    iter_start_time = time.time()

    # Forward tracking
    preds_forward, _, _, _ = model(trajs_e_forward, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e_forward = preds_forward[-1]

    # Reverse the iteration direction and use the final poses to track points backward
    preds_backward, _, _, _ = model(trajs_e_forward.flip(1), rgbs.flip(1), iters=iters, feat_init=None, beautify=True)
    trajs_e_backward = preds_backward[-1].flip(1)

    iter_time = time.time()-iter_start_time
    
    dist = compute_pairwise_distances(trajs_e_forward, trajs_e_backward)
    m = dist.mean(0) #N, average distence per point
    #m = dist.max(0)[0].shape
    print('average distence: ', dist.mean())
    keep_point = m < 2 #dist.mean() # keep the point if it's less the average
    
    
    trajs_e = trajs_e_forward[:,:,keep_point]

    print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S/iter_time))
    
    if sw is not None and sw.save_this:
        #rgbs_prep = utils.improc.preprocess_color(rgbs)
        #sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
        
        linewidth = 2    
        # visualize the input
        o1 = sw.summ_rgbs('inputs/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        # visualize the trajs overlaid on the rgbs
        o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=linewidth)
        # visualize the trajs alone
        o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1])*-0.5, cmap='spring', linewidth=linewidth)
        # concat these for a synced wide vis
        wide_cat = torch.cat([o1, o2, o3], dim=-1)
        sw.summ_rgbs('outputs/wide_cat', wide_cat.unbind(1))

        # write to disk, in case that's more convenient
        wide_list = list(wide_cat.unbind(1))
        wide_list = [wide[0].permute(1,2,0).cpu().numpy() for wide in wide_list]
        wide_list = [Image.fromarray(wide) for wide in wide_list]
        out_fn = './out_%d.gif' % sw.global_step
        wide_list[0].save(out_fn, save_all=True, append_images=wide_list[1:])
        print('saved %s' % out_fn)

    
    
    return trajs_e

    #return trajs_e_forward, trajs_e_backward


def run_model(model, rgbs, S_max=128, N=64, iters=16, sw=None):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B==1)

    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
    _, S, C, H, W = rgbs.shape

    # zero-vel init
    trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)

    iter_start_time = time.time()
    
    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None,beautify=True)
    trajs_e = preds[-1]

    iter_time = time.time()-iter_start_time
    print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S/iter_time))

    if sw is not None and sw.save_this:
        #rgbs_prep = utils.improc.preprocess_color(rgbs)
        #sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
        
        linewidth = 2    
        # visualize the input
        o1 = sw.summ_rgbs('inputs/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        # visualize the trajs overlaid on the rgbs
        o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=linewidth)
        # visualize the trajs alone
        o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1])*-0.5, cmap='spring', linewidth=linewidth)
        # concat these for a synced wide vis
        wide_cat = torch.cat([o1, o2, o3], dim=-1)
        sw.summ_rgbs('outputs/wide_cat', wide_cat.unbind(1))

        # write to disk, in case that's more convenient
        wide_list = list(wide_cat.unbind(1))
        wide_list = [wide[0].permute(1,2,0).cpu().numpy() for wide in wide_list]
        wide_list = [Image.fromarray(wide) for wide in wide_list]
        out_fn = './out_%d.gif' % sw.global_step
        wide_list[0].save(out_fn, save_all=True, append_images=wide_list[1:])
        print('saved %s' % out_fn)

    
    
    return trajs_e


def main(
        filename='./stock_videos/P11_102_from_1-52_to_1-55.MP4',
        S=48, # seqlen
        N=1024, # number of points per clip
        stride=8, # spatial stride of the model
        timestride=1, # temporal stride of the model
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        max_iters=4, # number of clips to run
        shuffle=False, # dataset shuffling
        log_freq=1, # how often to make image summaries
        log_dir='./logs_demo',
        init_dir='./reference_model',
        device_ids=[0],
):

    # the idea in this file is to run the model on a demo video,
    # and return some visualizations
    
    exp_name = 'de00' # copy from dev repo

    print('filename', filename)
    name = Path(filename).stem
    print('name', name)
    
    rgbs = read_mp4(filename)

    rgbs = np.stack(rgbs, axis=0) # S,H,W,3
    
    rgbs = rgbs[:,:,:,::-1].copy() # BGR->RGB
    rgbs = rgbs[::timestride]
    S_here,H,W,C = rgbs.shape
    print('rgbs', rgbs.shape)

    # autogen a name
    model_name = "%s_%d_%d_%s" % (name, S, N, exp_name)
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    log_dir = 'logs_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Pips(stride=8).cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()

    idx = list(range(0, max(S_here-S,1), S))
    if max_iters:
        idx = idx[:max_iters]
    
    for si in idx:
        global_step += 1
        
        iter_start_time = time.time()

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=6,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        rgb_seq = rgbs[si:si+S]
        rgb_seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
        rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W
        
        with torch.no_grad():
            trajs_e = run_model_forward_backward(model, rgb_seq, S_max=S, N=N, iters=iters, sw=sw_t)
            print(trajs_e.shape)

        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, max_iters, iter_time))
        
            
    writer_t.close()

if __name__ == '__main__':
    Fire(main)
