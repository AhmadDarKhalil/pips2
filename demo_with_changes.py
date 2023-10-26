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
import argparse


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


def run_model(model, rgbs, S_max=128, N=64, iters=16, sw=None, pass_on_trajs=None):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B==1)

    if pass_on_trajs is None:
        # pick N points to track; we'll use a uniform grid
        N_ = np.sqrt(N).round().astype(np.int32)
        grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
        grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
        grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
        xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
        _, S, C, H, W = rgbs.shape

        # zero-vel init
        trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)
    else:
        trajs_e = pass_on_trajs

    iter_start_time = time.time()
    
    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e = preds[-1]

    iter_time = time.time()-iter_start_time
    print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S/iter_time))

    if sw is not None and sw.save_this:
        annotate_whole_video(rgbs, trajs_e, sw, file_suffix=sw.global_step)    
    
    return trajs_e


def annotate_whole_video(rgbs, frame_points, sw, file_suffix="ALL"):
    linewidth = 2
    # visualize the input
    o1 = sw.summ_rgbs('inputs/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
    # visualize the trajs overlaid on the rgbs
    o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', frame_points[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=linewidth)
    # visualize the trajs alone
    o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', frame_points[0:1], torch.ones_like(rgbs[0:1])*-0.5, cmap='spring', linewidth=linewidth)
    # concat these for a synced wide vis
    wide_cat = torch.cat([o1, o2, o3], dim=-1)
    sw.summ_rgbs('outputs/wide_cat', wide_cat.unbind(1))

    # write to disk, in case that's more convenient
    wide_list = list(wide_cat.unbind(1))
    wide_list = [wide[0].permute(1,2,0).cpu().numpy() for wide in wide_list]
    out_fn = f"./example_outputs/NO_REINIT_{filename_for_demo.split('.MP4')[0]}_out_{file_suffix}.mp4"
    video_writer = cv2.VideoWriter("TEST_MP4_SAVE_CONVERT_RGB.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 24.0, (2688,512))
    for wide in wide_list:
        video_writer.write(cv2.cvtColor(wide, cv2.COLOR_RGB2BGR))
    video_writer.release()
    print(f"Saved {out_fn}")


def main(
        filename='./stock_videos/P02_102_from_2-44_to_2-47.MP4',
        S=48, # seqlen
        N=512, # number of points per clip
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
    filename = f"./stock_videos/{filename_for_demo}"
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
    
    pass_on_trajs = None
    all_frame_points = torch.tensor([])
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
            trajs_e = run_model(model, rgb_seq, S_max=S, N=N, iters=iters, sw=sw_t, pass_on_trajs=pass_on_trajs)
        pass_on_trajs = trajs_e[0, -1, :, :].repeat(1, S, 1, 1)
        all_frame_points = torch.cat([all_frame_points, trajs_e.detach().cpu()], dim=1)

        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, max_iters, iter_time))
    print("### Annotating whole video ###")
    # Use all points on whole video
    rgb_seq_full = torch.from_numpy(rgbs[0:si+S]).permute(0, 3, 1, 2).to(torch.float32)
    rgb_seq_full = F.interpolate(rgb_seq_full, image_size, mode='bilinear').unsqueeze(0)
    annotate_whole_video(rgb_seq_full, all_frame_points, sw_t, file_suffix="ALL")
    print("### Done ###")
            
    writer_t.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp4_filename", action="store", dest="mp4_filename", default="P01_101_from_0-42_to_0-47.MP4")
    args = parser.parse_args()
    filename_for_demo = args.mp4_filename
    Fire(main)
