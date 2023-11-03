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
import matplotlib.pyplot as plt
import math


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


def annotate_model_comparison(
    po_rgbs, po_window_points, epic_rgbs, epic_window_points, writer_t,
    log_freq, sample_idx_name, two_colour_map=None, gt_window_points=None
):
    sw = utils.improc.Summ_writer(
        writer=writer_t,
        global_step=0,
        log_freq=log_freq,
        fps=6,
        scalar_freq=int(log_freq/2),
        just_gif=True
    )
    linewidth = 2
    out_fn = f"./synthetic_outputs/BETTER_TRACKS_10_po_v_epic_synthetic_{sample_idx_name}{'' if two_colour_map is None else '_with_CC'}.mp4"
    video_res = (1536,384) if gt_window_points is not None else (1024,384)
    video_writer = cv2.VideoWriter(out_fn, cv2.VideoWriter_fourcc(*'MP4V'), 12.0, video_res)

    for window_idx, (po_frame_points, epic_frame_points) in enumerate(zip(po_window_points, epic_window_points)):
        print(f"Annotating window: {window_idx+1}")
        o_gt = torch.tensor([])
        if gt_window_points is not None:
            # Visualise the tracks overload on the rgbs for GT
            o_gt = sw.summ_traj2ds_on_rgbs(
                'outputs/gt_trajs_on_rgbs', gt_window_points[window_idx][0:1],
                utils.improc.preprocess_color(po_rgbs[window_idx][0:1]),
                cmap='spring', linewidth=linewidth,
                two_colour_map=None
            )
        # Visualise the tracks overlaid on the rgbs for PO
        o1 = sw.summ_traj2ds_on_rgbs(
            'outputs/po_trajs_on_rgbs', po_frame_points[0:1],
            utils.improc.preprocess_color(po_rgbs[window_idx][0:1]),
            cmap='spring', linewidth=linewidth,
            two_colour_map=two_colour_map
        )
        # Visualise the tracks overload on the rgbs for EPIC
        o2 = sw.summ_traj2ds_on_rgbs(
            'outputs/epic_trajs_on_rgbs', epic_frame_points[0:1],
            utils.improc.preprocess_color(epic_rgbs[window_idx][0:1]),
            cmap='spring', linewidth=linewidth,
            two_colour_map=two_colour_map
        )
        # Concat views
        wide_cat = torch.cat([o_gt, o1, o2], dim=-1)
        sw.summ_rgbs('outputs/model_comp_wide_cat', wide_cat.unbind(1))

        # write to disk, in case that's more convenient
        wide_list = list(wide_cat.unbind(1))
        for wide in wide_list:
            video_writer.write(cv2.cvtColor(wide[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    video_writer.release()
    print(f"Saved model comparison {out_fn}.")


def annotate_video_with_dots(rgbs, window_points, sw, sample_idx_name, file_prefix="GROUND_TRUTH", valids=None):
    linewidth = 2
    print(f"Number of windows = {len(window_points)}")
    out_fn = f"./synthetic_outputs/{file_prefix}_synthetic_{sample_idx_name}.mp4"
    video_writer = cv2.VideoWriter(out_fn, cv2.VideoWriter_fourcc(*'MP4V'), 12.0, (1536,384))
    for window_idx, frame_points in enumerate(window_points):
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
    sample_idx_name, save_vis, model_type, cc_error=False
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
            if cc_error:
                track_cc_error, trajs_e_forward, trajs_e_backward = calculate_cc_error(model, rgb_seq, pass_on_trajs, S_max=S, N=N, iters=iters, sw=sw_t)
                print("TODO")
            print(trajs_e.shape)
        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, max_iters, iter_time))

        if trajs_e.size(2) == 0:
            print("Exiting early because no points left...")
            break
        pass_on_trajs = trajs_e[0, -1, :, :].repeat(1, S, 1, 1)
        window_points.append(trajs_e.detach().cpu())

    avg_error, error_matrix = calculate_ground_truth_error(gt_tracks, window_points)
    return_obj = {
        "global_l2_error": avg_error,
        "track_l2_error": torch.mean(error_matrix, dim=0)
    }
    if cc_error:
        return_obj["cc_error"] = track_cc_error
        return_obj["cc_forward"] = trajs_e_forward
        return_obj["cc_backward"] = trajs_e_backward
    if save_vis:
        return_obj["trajs"] = window_points
        return_obj["rgb_seq_full"] = rgb_seq_full
    return return_obj


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


def calculate_cc_error(model, rgbs, init_trajs, S_max=128, N=64, iters=16, sw=None):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B == 1)
    trajs_e_forward = init_trajs 
    iter_start_time = time.time()

    # Forward tracking
    preds_forward, _, _, _ = model(trajs_e_forward, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e_forward = preds_forward[-1]

    # Reverse the iteration direction and use the final poses to track points backward
    preds_backward, _, _, _ = model(trajs_e_forward.flip(1), rgbs.flip(1), iters=iters, feat_init=None, beautify=True)
    trajs_e_backward = preds_backward[-1].flip(1)

    iter_time = time.time()-iter_start_time
    
    dist = compute_pairwise_distances(trajs_e_forward, trajs_e_backward)
    return dist, trajs_e_forward.unsqueeze(0), trajs_e_backward.unsqueeze(0)
 

def compare_gt_po_epic(
    video_path, track_path, sample_idx, save_vis, save_model_diff_vis, model_name,
    image_size, S, iters, max_iters, log_freq, N, log_dir, timestride, cc_error
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

    # Compute P.O. model error and visualisations
    po_error_obj = visualise_track_predictions(
        model_name, image_size, rgbs, track_path,
        "/home/deepthought/Ahmad/pips2/reference_model",
        S_here, S, iters, max_iters, po_writer_t,
        log_freq, N, sample_idx,
        True if save_vis or save_model_diff_vis else False,
        "po", cc_error
    )
    # Computer EPIC model error and visualistions
    epic_error_obj = visualise_track_predictions(
        model_name, image_size, rgbs, track_path,
        "/home/deepthought/Ahmad/pips2/reference_model_epic",
        S_here, S, iters, max_iters, epic_writer_t,
        log_freq, N, sample_idx,
        True if save_vis or save_model_diff_vis else False,
        "epic", cc_error
    )

    if save_vis:
        # Only compute ground truth if individual visualisations needed
        sw_t = utils.improc.Summ_writer(
            writer=gt_writer_t,
            global_step=0,
            log_freq=log_freq,
            fps=6,
            scalar_freq=int(log_freq/2),
            just_gif=True
        )
        visualise_track_ground_truths(image_size, rgbs, track_path, sw_t, sample_idx)
        po_sw_t = utils.improc.Summ_writer(
            writer=po_writer_t,
            global_step=0,
            log_freq=log_freq,
            fps=6,
            scalar_freq=int(log_freq/2),
            just_gif=True
        )
        annotate_video_with_dots(po_error_obj['rgb_seq_full'], po_error_obj['trajs'], po_sw_t, sample_idx_name, file_prefix=f"po_PREDS")
        epic_sw_t = utils.improc.Summ_writer(
            writer=epic_writer_t,
            global_step=0,
            log_freq=log_freq,
            fps=6,
            scalar_freq=int(log_freq/2),
            just_gif=True
        )
        annotate_video_with_dots(epic_error_obj['rgb_seq_full'], epic_error_obj['trajs'], epic_sw_t, sample_idx_name, file_prefix=f"epic_PREDS")

    return po_error_obj, epic_error_obj


def plot_error_diff(vids_diffs, tracks_diffs, split, flip_negatives=False):
    new_min_vids = -(math.floor(abs(min(vids_diffs))/0.5) * 0.5)
    new_max_vids = math.floor(abs(max(vids_diffs))/0.5) * 0.5

    new_min_tracks = -(math.floor(abs(min(tracks_diffs))/1.0) * 1.0)
    new_max_tracks = math.floor(abs(max(tracks_diffs))/1.0) * 1.0

    vids_bins = np.concatenate([np.array([min(vids_diffs)]), np.arange(new_min_vids, 0, 0.5), np.arange(0, new_max_vids, 0.5), np.array([max(vids_diffs)])])
    tracks_bins = np.concatenate([np.array([min(tracks_diffs)]), np.arange(new_min_tracks, 0, 1.0), np.arange(0, new_max_tracks, 1.0), np.array([max(tracks_diffs)])])
    vids_hist, vids_bin_edges = np.histogram(vids_diffs, bins=vids_bins)
    tracks_hist, tracks_bin_edges = np.histogram(tracks_diffs, bins=tracks_bins)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(f"Difference in Error Between {split} Samples on P.O. & EPIC")

    # Plot bars with positive or negative heights based on bin values
    for i in range(len(vids_hist)):
        if i == 0:
            width = round(abs(min(vids_diffs)) - abs(new_min_vids), 2)
        elif i == len(vids_hist)-1:
            width = round(abs(max(vids_diffs)) - abs(new_max_vids), 2)
        else:
            width = 0.5
        if vids_bin_edges[i] < 0:
            axs[0].bar(vids_bin_edges[i], -vids_hist[i] if flip_negatives else vids_hist[i], width=width, color='red', align='edge')
        else:
            axs[0].bar(vids_bin_edges[i], vids_hist[i], width=width, color='blue', align='edge')

    for i in range(len(tracks_hist)):
        if i == 0:
            width = round(abs(min(tracks_diffs)) - abs(new_min_tracks), 2)
        elif i == len(vids_hist)-1:
            width = round(abs(max(tracks_diffs)) - abs(new_max_tracks), 2)
        else:
            width = 1.0
        if tracks_bin_edges[i] < 0:
            axs[1].bar(tracks_bin_edges[i], -tracks_hist[i] if flip_negatives else tracks_hist[i], width=width, color='red', align='edge')
        else:
            axs[1].bar(tracks_bin_edges[i], tracks_hist[i], width=width, color='blue', align='edge')

    axs[0].set_title("Videos")
    axs[0].set_xlabel("L2 Difference")
    axs[1].set_title("Tracks")
    axs[1].set_xlabel("L2 Difference")
    axs[1].set_xlim([-10.0, 10.0])
    fig.tight_layout()
    fig.savefig(f"{split}_samples_error_diff_hists.png")


def main(
    sample_idx="000000",
    save_vis=False,
    save_model_diff_vis=False,
    aggregate_all=False,
    on_val=False,
    plot_hists=False,
    cc_error=False,
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
        vids_diffs = []
        num_tracks_better, total_tracks = 0, 0
        tracks_diffs = []
        num_cc_vids_worse, total_cc_vids = 0, 0
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

            po_error_obj, epic_error_obj = compare_gt_po_epic(
                filename, track_path, sample_idx, save_vis, save_model_diff_vis, model_name, image_size,
                S, iters, max_iters, log_freq, N, "logs_demo", timestride, cc_error
            )
            po_avg_error = po_error_obj["global_l2_error"]
            po_track_avg_errors = po_error_obj["track_l2_error"]
            epic_avg_error = epic_error_obj["global_l2_error"]
            epic_track_avg_errors = epic_error_obj["track_l2_error"]

            # Count videos better on EPIC model
            total_vids += 1
            counter += 1
            sample_num += 1
            vids_diffs.append(po_avg_error - epic_avg_error)
            if epic_avg_error < po_avg_error:
                num_vids_better += 1
                total_cc_vids += 1
                if cc_error:
                    num_cc_vids_worse += 1 if epic_error_obj["cc_error"].mean().item() > po_error_obj["cc_error"].mean().item() else 0

            # Count tracks better on EPIC model
            num_tracks_better += torch.sum((epic_track_avg_errors < po_track_avg_errors).int(), dim=0).item()

            if save_model_diff_vis:
                model_comp_writer = epic_writer_t = SummaryWriter(f"{log_dir}/{model_name}/epic_v_po/t", max_queue=10, flush_secs=60)
                threshold = 10.0
                # Pick the best three points
                show_points = 3
                # ####
                diff_bool_idx = ((po_track_avg_errors - epic_track_avg_errors) > threshold)
                temp_num_tracks_better = torch.sum(diff_bool_idx.int(), dim=0).item()
                print(f"@@@ Sample ID={sample_idx} -> Number of Tracks Better on EPIC by {threshold} = {temp_num_tracks_better}")
                if temp_num_tracks_better > 0:
                    if cc_error:
                        # TODO - Enable this for multiple windows
                        chosen_po_trajs = [torch.cat([
                            po_error_obj["cc_forward"][0][:, :, diff_bool_idx, :],
                            po_error_obj["cc_backward"][0][:, :, diff_bool_idx, :]
                        ], dim=2)]
                        chosen_epic_trajs = [torch.cat([
                            epic_error_obj["cc_forward"][0][:, :, diff_bool_idx, :],
                            epic_error_obj["cc_backward"][0][:, :, diff_bool_idx, :]
                        ], dim=2)]
                        two_colour_map = np.concatenate([np.zeros(temp_num_tracks_better), np.ones(temp_num_tracks_better)])
                    else:
                        chosen_po_trajs = [window_trajs[:, :, diff_bool_idx, :] for window_trajs in po_error_obj['trajs']]
                        chosen_epic_trajs = [window_trajs[:, :, diff_bool_idx, :] for window_trajs in epic_error_obj['trajs']]
                        two_colour_map = None

                    gt_tracks, _ = load_synthetic_tracks(track_path)
                    annotate_model_comparison(
                        po_error_obj['rgb_seq_full'],
                        chosen_po_trajs,
                        epic_error_obj['rgb_seq_full'],
                        chosen_epic_trajs,
                        model_comp_writer,
                        log_freq,
                        sample_idx,
                        two_colour_map=two_colour_map,
                        # TODO - Enable this for multiple windows
                        gt_window_points=[torch.from_numpy(gt_tracks)[:, diff_bool_idx, :].unsqueeze(0)]
                    )
            if counter == 10:
                sys.exit(0)
            total_tracks += po_track_avg_errors.size(0)
            temp_diffs = po_track_avg_errors - epic_track_avg_errors
            tracks_diffs.extend(list(temp_diffs.numpy()))

            if counter == 100:
                hundred_vids = True
        print(f"Percentage of Videos Better on EPIC -> {(num_vids_better/total_vids)*100.0}% ({num_vids_better}/{total_vids})")
        print(f"Percentage of Tracks Better on EPIC -> {(num_tracks_better/total_tracks)*100.0}% ({num_tracks_better}/{total_tracks})")
        if cc_error:
            print(f"Videos which are better on EPIC = {total_cc_vids}, Number of Which are Worse for CC Measure = {num_cc_vids_worse}")
        # Create histograms of difference in better vids/tracks
        if plot_hists:
            plot_error_diff(
                vids_diffs,
                tracks_diffs,
                "val" if on_val else "train"
            )
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

        po_error_obj, epic_error_obj = compare_gt_po_epic(
            filename, track_path, sample_idx, save_vis, save_model_diff_vis, model_name, image_size,
            S, iters, max_iters, log_freq, N, "logs_demo", timestride
        )
        print(f"P.O. Avg. Error = {po_error_obj['global_l2_error']}, EPIC Avg. Error = {epic_error_obj['global_l2_error']}")
        print(f"Number of Tracks Better on EPIC -> {torch.sum((epic_error_obj['track_l2_error'] < po_error_obj['track_l2_error']).int(), dim=0)}/{po_error_obj['track_l2_error'].size(0)}")
    
    print("DONE!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_idx", action="store", dest="sample_idx", default="000000",
        help="6 digit code for folder containing rgb.mp4 and track.npz files."
    )
    parser.add_argument(
        "--save_vis", action="store_true", dest="save_vis",
        help="True=Save separate visualisations of each model, False=Skip visualisation creation and save."
    )
    parser.add_argument(
        "--save_model_diff_vis", action="store_true", dest="save_model_diff_vis",
        help="True=Save visualisations of tracks better on EPIC, False=Do not save"
    )
    parser.add_argument(
        "--aggregate_all", action="store_true", dest="aggregate_all",
        help="True=Ignore sample_idx and compute % better for 100 train/val, False=Compute difference of errors for one video"
    )
    parser.add_argument(
        "--on_val", action="store_true", dest="on_val",
        help="True=Use validation videos, False=Use training videos"
    )
    parser.add_argument(
        "--plot_hists", action="store_true", dest="plot_hists"
    )
    parser.add_argument("--cc_error", action="store_true", dest="cc_error")
    parser.set_defaults(save_vis=False)
    parser.set_defaults(save_model_diff_vis=False)
    parser.set_defaults(aggregate_all=False)
    parser.set_defaults(on_val=False)
    parser.set_defaults(plot_hists=False)
    parser.set_defaults(cc_error=False)
    args = parser.parse_args()
    print(f"save_vis={args.save_vis}")
    print(f"save_model_diff_vis={args.save_model_diff_vis}")
    print(f"aggregate_all={args.aggregate_all}")
    print(f"on_val={args.on_val}")
    print(f"plot_hists={args.plot_hists}")
    print(f"cc_error={args.cc_error}")

    Fire(main(
        sample_idx=args.sample_idx,
        save_vis=args.save_vis,
        save_model_diff_vis=args.save_model_diff_vis,
        aggregate_all=args.aggregate_all,
        on_val=args.on_val,
        plot_hists=args.plot_hists,
        cc_error=args.cc_error
    ))
