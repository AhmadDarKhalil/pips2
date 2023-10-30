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


def load_synthetic_tracks(path):
    points = np.load(path)
    points = points["track_g"]
    return points[:, :, :2]

def annotate_video_with_dots(rgbs, window_points, sw, file_prefix="GROUND_TRUTH"):
    linewidth = 2
    print(f"Number of windows = {len(window_points)}")
    out_fn = f"{file_prefix}_synthetic_{args.sample_idx}.mp4"
    video_writer = cv2.VideoWriter(out_fn, cv2.VideoWriter_fourcc(*'MP4V'), 12.0, (2688,512))
    for window_idx, frame_points in  enumerate(window_points):
        print(f"Annotating window: {window_idx+1}")
        # visualize the input
        o1 = sw.summ_rgbs('inputs/rgbs', utils.improc.preprocess_color(rgbs[window_idx][0:1]).unbind(1))
        # visualize the trajs overlaid on the rgbs
        o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', frame_points[0:1], utils.improc.preprocess_color(rgbs[window_idx][0:1]), cmap='spring', linewidth=linewidth)
        # visualize the trajs alone
        o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', frame_points[0:1], torch.ones_like(rgbs[window_idx][0:1])*-0.5, cmap='spring', linewidth=linewidth)
        # concat these for a synced wide vis
        wide_cat = torch.cat([o1, o2, o3], dim=-1)
        sw.summ_rgbs('outputs/wide_cat', wide_cat.unbind(1))

        # write to disk, in case that's more convenient
        wide_list = list(wide_cat.unbind(1))
        #wide_list = [wide[0].permute(1,2,0).cpu().numpy() for wide in wide_list]
        for wide in wide_list:
            video_writer.write(cv2.cvtColor(wide[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Saved {out_fn}")


def visualise_track_ground_truths(image_size, rgbs, track_path, sw_t):
    rgb_seq = torch.from_numpy(rgbs).permute(0,3,1,2).to(torch.float32) # S,3,H,W
    rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W

    trajs_g = load_synthetic_tracks(track_path)

    annotate_video_with_dots([rgb_seq], [torch.from_numpy(trajs_g).unsqueeze(0)], sw_t, "GROUND_TRUTH")

def visualise_track_predictions(model_name, image_size, rgbs, track_path, init_dir, S_here, S, iters, max_iters, writer_t, log_freq, N):
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

    gt_tracks = torch.from_numpy(load_synthetic_tracks(track_path)).unsqueeze(0)
    pass_on_trajs = gt_tracks[0, -1, :, :].repeat(1, S_here if S_here < S else S, 1, 1).cuda()
    window_points = []
    rgb_seq_full = []
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
        rgb_seq_full.append(rgb_seq)

        with torch.no_grad():
            trajs_e = run_model(model, rgb_seq, pass_on_trajs, S_max=S, N=N, iters=iters, sw=sw_t)
            print(trajs_e.shape)
        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, max_iters, iter_time))

        if trajs_e.size(2) == 0:
            print("Exiting early because no points left...")
            break
        pass_on_trajs = trajs_e[0, -1, :, :].repeat(1, S, 1, 1)
        window_points.append(trajs_e.detach().cpu())
    annotate_video_with_dots(rgb_seq_full, window_points, sw_t, file_prefix="PREDS")


def run_model(model, rgbs, init_trajs, S_max=128, N=64, iters=16, sw=None):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B==1)

    # Initialise points in window
    trajs_e = init_trajs

    iter_start_time = time.time()
    print("######")
    print(trajs_e.size(), rgbs.size())
    print("######")

    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None,beautify=True)
    trajs_e = preds[-1]

    iter_time = time.time()-iter_start_time
    print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S/iter_time))
    return trajs_e


def run_model_forward_backward():
    pass


def main(
    sample_idx="000000",
    vis_track_type="gt",
    S=48,
    N=1024,
    stride=8,
    timestride=1, # temporal stride of the model
    iters=16, # inference steps of the model
    image_size=(512,896), # input resolution
    max_iters=4, # number of clips to run
    shuffle=False, # dataset shuffling
    log_freq=1, # how often to make image summaries
    log_dir='./logs_demo',
    init_dir='/home/deepthought/Ahmad/pips2/reference_model',
    device_ids=[0],
):
    filename = f"/media/deepthought/DATA/Ahmad/pointodyssey/epic/ae_24_96_384x512/{sample_idx}/rgb.mp4"
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

    track_path = f"{filename.split('rgb.mp4')[0]}track.npz"

    if vis_track_type == "gt":
        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=0,
            log_freq=log_freq,
            fps=6,
            scalar_freq=int(log_freq/2),
            just_gif=True
        )
        visualise_track_ground_truths(image_size, rgbs, track_path, sw_t)
    elif vis_track_type == "pred":
        visualise_track_predictions(
            model_name, image_size, rgbs, track_path, init_dir,
            S_here, S, iters, max_iters, writer_t, log_freq, N
        )
    else:
        print("Invalid vis_track_type value!")
    
    print("DONE!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", action="store", dest="sample_idx", default="000000")
    parser.add_argument("--vis_track_type", action="store", dest="vis_track_type", default="gt")
    #parser.add_argument("--keep_good_points", action="store_true", dest="keep_good_points")
    parser.set_defaults(keep_good_points=False)
    args = parser.parse_args()
    #keep_good_points = args.keep_good_points
    #print(f"### Keeping {'GOOD' if keep_good_points else 'BAD'} Points ###")
    Fire(main(sample_idx=args.sample_idx, vis_track_type=args.vis_track_type))
