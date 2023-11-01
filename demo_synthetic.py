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
import os

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
    return points[:, :, :2], points[:, :, 3]

def annotate_video_with_dots(rgbs, window_points, sw, sample_idx_name, file_prefix="GROUND_TRUTH", valids=None):
    linewidth = 2
    print(f"Number of windows = {len(window_points)}")
    out_fn = f"./synthetic_outputs/TEST_{file_prefix}_synthetic_{sample_idx_name}.mp4"
    video_writer = cv2.VideoWriter(out_fn, cv2.VideoWriter_fourcc(*'MP4V'), 12.0, (1536,384))
    for window_idx, frame_points in  enumerate(window_points):
        print(f"Annotating window: {window_idx+1}")
        # visualize the input
        o1 = sw.summ_rgbs('inputs/rgbs', utils.improc.preprocess_color(rgbs[window_idx][0:1]).unbind(1))
        # visualize the trajs overlaid on the rgbs
        o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', frame_points[0:1], utils.improc.preprocess_color(rgbs[window_idx][0:1]), cmap='spring', linewidth=linewidth, valids=valids)
        # visualize the trajs alone
        o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', frame_points[0:1], torch.ones_like(rgbs[window_idx][0:1])*-0.5, cmap='spring', linewidth=linewidth, valids=valids)
        # concat these for a synced wide vis
        wide_cat = torch.cat([o1, o2, o3], dim=-1)
        sw.summ_rgbs('outputs/wide_cat', wide_cat.unbind(1))

        # write to disk, in case that's more convenient
        wide_list = list(wide_cat.unbind(1))
        for wide in wide_list:
            video_writer.write(cv2.cvtColor(wide[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Saved {out_fn}")


def visualise_track_ground_truths(image_size, rgbs, track_path, sw_t, sample_idx_name):
    rgb_seq = torch.from_numpy(rgbs).permute(0,3,1,2).to(torch.float32) # S,3,H,W
    rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W
    print(rgb_seq.size())

    trajs_g, valids_g = load_synthetic_tracks(track_path)

    annotate_video_with_dots(
        [rgb_seq],
        [torch.from_numpy(trajs_g).unsqueeze(0)],
        sw_t,
        sample_idx_name,
        "GROUND_TRUTH",
        valids=torch.from_numpy(valids_g).unsqueeze(0)
    )


def visualise_track_predictions(
    model_name, image_size, rgbs, track_path, init_dir,
    S_here, S, iters, max_iters, writer_t, log_freq, N,
    sample_idx_name, save_vis, model_type
):
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

    gt_tracks, _ = load_synthetic_tracks(track_path)
    gt_tracks = torch.from_numpy(gt_tracks).unsqueeze(0)
    pass_on_trajs = gt_tracks[0, 0, :, :].repeat(1, S_here if S_here < S else S, 1, 1).cuda()
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

    if save_vis:
        annotate_video_with_dots(rgb_seq_full, window_points, sw_t, sample_idx_name, file_prefix=f"{model_type}_PREDS")

    avg_error, error_matrix = calculate_ground_truth_error(gt_tracks, window_points)
    return avg_error, torch.mean(error_matrix, dim=0)


def calculate_ground_truth_error(gt_tracks, window_points):
    s = 0
    window_track_diffs = []
    for frame_points in window_points:
        window_seq_len = frame_points.size(1)
        window_gt = gt_tracks[:, s:s+window_seq_len, :, :]
        distances = compute_pairwise_distances(window_gt, frame_points)
        s += window_seq_len
        window_track_diffs.append(distances)
    error_matrix = torch.cat(window_track_diffs)
    print(f"Error Matrix Size = {error_matrix.size()}")
    return torch.mean(error_matrix).item(), error_matrix


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


def compare_gt_po_epic(
    video_path, track_path, sample_idx, save_vis, model_name,
    image_size, S, iters, max_iters, log_freq, N, log_dir, timestride
):
    # Create log writer for GT, PO and EPIC
    gt_writer_t = SummaryWriter(f"{log_dir}/{model_name}/gt/t", max_queue=10, flush_secs=60)
    po_writer_t = SummaryWriter(f"{log_dir}/{model_name}/po/t", max_queue=10, flush_secs=60)
    epic_writer_t = SummaryWriter(f"{log_dir}/{model_name}/epic/t", max_queue=10, flush_secs=60)

    # Load video into tensor
    rgbs = read_mp4(video_path)
    print(len(rgbs))
    rgbs = np.stack(rgbs, axis=0) # S,H,W,3
    rgbs = rgbs[:,:,:,::-1].copy() # BGR->RGB
    rgbs = rgbs[::timestride]
    S_here,H,W,C = rgbs.shape
    print('rgbs', rgbs.shape)

    # Compute ground truth visualisations
    if save_vis:
        sw_t = utils.improc.Summ_writer(
            writer=gt_writer_t,
            global_step=0,
            log_freq=log_freq,
            fps=6,
            scalar_freq=int(log_freq/2),
            just_gif=True
        )
        visualise_track_ground_truths(image_size, rgbs, track_path, sw_t, sample_idx)

    # Compute P.O. model error and visualisations
    po_avg_error, po_track_avg_errors = visualise_track_predictions(
        model_name, image_size, rgbs, track_path,
        "/home/deepthought/Ahmad/pips2/reference_model",
        S_here, S, iters, max_iters, po_writer_t,
        log_freq, N, sample_idx, save_vis, "po"
    )

    # Computer EPIC model error and visualistions
    epic_avg_error, epic_track_avg_errors = visualise_track_predictions(
        model_name, image_size, rgbs, track_path,
        "/home/deepthought/Ahmad/pips2/reference_model_epic",
        S_here, S, iters, max_iters, epic_writer_t,
        log_freq, N, sample_idx, save_vis, "epic"
    )
    return po_avg_error, po_track_avg_errors, epic_avg_error, epic_track_avg_errors


def main(
    sample_idx="000000",
    save_vis=False,
    aggregate_all=False,
    on_val=False,
    S=48,
    N=1024,
    stride=8,
    timestride=1, # temporal stride of the model
    iters=16, # inference steps of the model
    image_size=(384,512),#(512,896), #(480,854), # input resolution
    max_iters=4, # number of clips to run
    shuffle=False, # dataset shuffling
    log_freq=1, # how often to make image summaries
    log_dir='./logs_demo',
    device_ids=[0],
):
    exp_name = 'de00' # copy from dev repo
    if on_val:
        dataset_split_dir = "ae_24_96_384x512_val"
    else:
        dataset_split_dir = "ae_24_96_384x512"

    if aggregate_all:
        num_vids_better, total_vids = 0, 0
        num_tracks_better, total_tracks = 0, 0
        #for i in range(0, 100):
        counter, sample_num = 0, 0
        hundred_vids = False
        while not hundred_vids:
            sample_idx = str(sample_num).zfill(6)
            track_dir = f"/media/deepthought/DATA/Ahmad/pointodyssey/epic/{dataset_split_dir}/{sample_idx}"
            filename = f"{track_dir}/rgb.mp4"
            if not os.path.exists(filename):
                print("Path does not exist, skipping...")
                sample_num += 1
                continue

            print('filename', filename)
            name = Path(filename).stem
            print('name', name)

            # autogen a name
            model_name = "%s_%s_%d_%d_%s" % (sample_idx, name, S, N, exp_name)
            import datetime
            model_date = datetime.datetime.now().strftime('%H:%M:%S')
            model_name = model_name + '_' + model_date
            print('model_name', model_name)

            track_path = f"{track_dir}/track.npz"

            po_avg_error, po_track_avg_errors, epic_avg_error, epic_track_avg_errors = compare_gt_po_epic(
                filename, track_path, sample_idx,save_vis, model_name, image_size,
                S, iters, max_iters, log_freq, N, "logs_demo", timestride
            )
            # Count videos better on EPIC model
            total_vids += 1
            counter += 1
            sample_num += 1
            if epic_avg_error < po_avg_error:
                num_vids_better += 1

            # Count tracks better on EPIC model
            num_tracks_better += torch.sum((epic_track_avg_errors < po_track_avg_errors).int(), dim=0).item()
            total_tracks += po_track_avg_errors.size(0)

            if counter == 100:
                hundred_vids = True
        print(f"Percentage of Videos Better on EPIC -> {(num_vids_better/total_vids)*100.0}% ({num_vids_better}/{total_vids})")
        print(f"Percentage of Tracks Better on EPIC -> {(num_tracks_better/total_tracks)*100.0}% ({num_tracks_better}/{total_tracks})")
    else:
        track_dir = f"/media/deepthought/DATA/Ahmad/pointodyssey/epic/{dataset_split_dir}/{sample_idx}"
        filename = f"{track_dir}/rgb.mp4"

        print('filename', filename)
        name = Path(filename).stem
        print('name', name)

        # autogen a name
        model_name = "%s_%s_%d_%d_%s" % (sample_idx, name, S, N, exp_name)
        import datetime
        model_date = datetime.datetime.now().strftime('%H:%M:%S')
        model_name = model_name + '_' + model_date
        print('model_name', model_name)

        track_path = f"{track_dir}/track.npz"

        po_avg_error, po_track_avg_errors, epic_avg_error, epic_track_avg_errors = compare_gt_po_epic(
            filename, track_path, sample_idx,save_vis, model_name, image_size,
            S, iters, max_iters, log_freq, N, "logs_demo", timestride
        )
        print(f"P.O. Avg. Error = {po_avg_error}, EPIC Avg. Error = {epic_avg_error}")
        print(f"Number of Tracks Better on EPIC -> {torch.sum((epic_track_avg_errors < po_track_avg_errors).int(), dim=0)}/{po_track_avg_errors.size(0)}")
    
    print("DONE!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_idx", action="store", dest="sample_idx", default="000000",
        help="6 digit code for folder containing rgb.mp4 and track.npz files."
    )
    parser.add_argument(
        "--save_vis", action="store_true", dest="save_vis",
        help="True=Save visualisation, False=Skip visualisation creation and save."
    )
    parser.add_argument(
        "--aggregate_all", action="store_true", dest="aggregate_all",
        help="True=Ignore sample_idx and compute % better for 100 train/val, False=Compute difference of errors for one video"
    )
    parser.add_argument(
        "--on_val", action="store_true", dest="on_val",
        help="True=Use validation videos, False=Use training videos"
    )
    parser.set_defaults(save_vis=False)
    parser.set_defaults(aggregate_all=False)
    parser.set_defaults(on_val=False)
    args = parser.parse_args()
    print(f"save_vis={args.save_vis}")
    print(f"aggregate_all={args.aggregate_all}")
    print(f"on_val={args.on_val}")

    Fire(main(
        sample_idx=args.sample_idx,
        save_vis=args.save_vis,
        aggregate_all=args.aggregate_all,
        on_val=args.on_val
    ))
