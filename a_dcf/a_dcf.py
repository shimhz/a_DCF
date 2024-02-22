import sys
import os
import matplotlib.pyplot as plt
from typing import List
from dataclasses import dataclass

import numpy as np

@dataclass
class CostModel:
    "Class describing SASV-DCF's relevant costs"
    Pspf: float = 0.05
    Pnontrg: float = 0.05
    Ptrg: float = 0.9
    Cmiss: float = 1
    Cfa_asv: float = 10
    Cfa_cm: float = 20


def calculate_a_dcf_eers(
    sasv_score_dir: str,
    output_dir: str = None,
    cost_model: CostModel = CostModel(),
    printres: bool = True,
    ):
    if output_dir is None and not printres:
        raise KeyError("At least one output, either printing or writing to a file is required.")
    if output_dir is not None:
        f_out = open(output_dir, "w")

    data = np.genfromtxt(sasv_score_dir, dtype=str, delimiter=" ")
    scores = data[:, 2].astype(np.float)
    keys = data[:, 3]

    # Extract target, nontarget, and spoof scores from the ASV scores
    trg = scores[keys == 'target']
    nontrg = scores[keys == 'nontarget']
    spf = scores[keys == 'spoof']

    # three EERs: SASV-, SV-, SPF- EER
    sasv_eer, sasv_eer_thresh = compute_eer(trg, np.concatenate([nontrg, spf]))
    sv_eer, sv_eer_thresh = compute_eer(trg, nontrg)
    spf_eer, spf_eer_thresh = compute_eer(trg, spf)

    eer_msg = f"SASV-EER: {sasv_eer*100:3.2f}, SV-EER: {sv_eer*100:3.2f}, SPF-EER:{spf_eer*100:3.2f}"
    eer_thresh_msg = f"[Thresholds] SASV-EER: {sasv_eer_thresh:.5f}, SV-EER: {sv_eer_thresh:.5f}, SPF-EER:{spf_eer_thresh:.5f}"
    if printres:
        print(eer_msg)
        print(eer_thresh_msg)
    if output_dir is not None:
        f_out.write(eer_msg + "\n")
        f_out.write(eer_thresh_msg + "\n")

    far_asvs, far_cms, frrs, sasv_dcf_thresh = compute_sasv_det_curve(trg, nontrg, spf)

    sasv_dcfs = np.array([cost_model.Cmiss * cost_model.Ptrg]) * np.array(frrs) + \
        np.array([cost_model.Cfa_asv * cost_model.Pnontrg]) * np.array(far_asvs) + \
        np.array([cost_model.Cfa_cm * cost_model.Pspf]) * np.array(far_cms)

    sasv_dcfs_normed = normalize(sasv_dcfs, cost_model)


    min_sasv_dcf_idx = np.argmin(sasv_dcfs_normed)
    min_sasv_dcf = sasv_dcfs_normed[min_sasv_dcf_idx]
    min_sasv_dcf_thresh = sasv_dcf_thresh[min_sasv_dcf_idx]
    x_axis = np.arange(len(sasv_dcfs_normed))

    plt.plot(x_axis, sasv_dcfs_normed, color="b", label="sasv-dcf")
    # plt.plot(x_axis, sasv_dcf_thresh, color="r", label="threshold") # also needs to be normalized
    plt.plot(min_sasv_dcf_idx, min_sasv_dcf, marker="x")
    plt.annotate(f'({min_sasv_dcf_idx}, {min_sasv_dcf:.4f})', (min_sasv_dcf_idx, min_sasv_dcf), ha='center')
    plt.legend()
    plt.savefig('plot.png')

    dcf_msg = f"SASV-DCF: {min_sasv_dcf:.5f}, threshold: {min_sasv_dcf_thresh:.5f}"

    if printres:
        print(dcf_msg)
        print(cost_model)
    if output_dir is not None:
        f_out.write(dcf_msg + "\n")
        f_out.write(f"{cost_model}\n")

    if output_dir is not None:
        f_out.close()

    return {
        "sasv_eer": sasv_eer,
        "sasv_eer_thresh": sasv_eer_thresh,
        "sv_eer": sv_eer,
        "sv_eer_thresh": sv_eer_thresh,
        "spf_eer": spf_eer,
        "spf_eer_thresh": spf_eer_thresh,
        "min_sasv_dcf": min_sasv_dcf,
        "min_sasv_dcf_thresh": min_sasv_dcf_thresh,
    }


def normalize(sasv_dcfs: np.ndarray, cost_model: CostModel) -> np.ndarray:
    """
    print(f"tmp, {sasv_dcfs - np.minimum(c_all_accept, c_all_reject)}")
    """
    sasv_dcf_all_accept = np.array([cost_model.Cfa_asv * cost_model.Pnontrg + \
        cost_model.Cfa_cm * cost_model.Pspf])
    sasv_dcf_all_reject = np.array([cost_model.Cmiss * cost_model.Ptrg])
    print(f"sasv_dcf_all_accept: {sasv_dcf_all_accept}, sasv_dcf_all_reject: {sasv_dcf_all_reject}")

    sasv_dcfs_normed = sasv_dcfs / min(sasv_dcf_all_accept, sasv_dcf_all_reject)

    print(f"Before normalization: {sasv_dcfs}")
    print(f"After normalization: {sasv_dcfs_normed}")
    return sasv_dcfs_normed

def compute_det_curve(target_scores: List, nontarget_scores: List) -> List[np.ndarray]:

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds

def compute_sasv_det_curve(trg_scores: np.ndarray, nontrg_scores: np.ndarray, spf_scores: np.ndarray) -> List[List]:

    all_scores = np.concatenate((trg_scores, nontrg_scores, spf_scores))
    labels = np.concatenate(
        (np.ones_like(trg_scores), np.zeros_like(nontrg_scores), np.ones_like(spf_scores) + 1))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    scores_sorted = all_scores[indices]

    fp_nontrg, fp_spf, fn = len(nontrg_scores), len(spf_scores), 0
    far_asvs, far_cms, frrs, sasv_dcf_thresh = [1.], [1.], [0.], [float(np.min(scores_sorted))-1e-8]
    for sco, lab in zip(scores_sorted, labels):
        if lab == 0: # non-target
            fp_nontrg -= 1 # false alarm for accepting nontarget trial
        elif lab == 1: # target
            fn += 1 # miss
        elif lab == 2: # spoof
            fp_spf -= 1 # false alarm for accepting spof trial
        else:
            raise ValueError ("Label should be one of (0, 1, 2).")
        far_asvs.append(fp_nontrg / len(nontrg_scores))
        far_cms.append(fp_spf / len(spf_scores))
        frrs.append(fn / len(trg_scores))
        sasv_dcf_thresh.append(sco)

    return far_asvs, far_cms, frrs, sasv_dcf_thresh


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> List[np.float64]:
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]



if __name__ == "__main__":
    # default sasv-dcf cost func
    if len(sys.argv) == 2:
        sys.exit(
            calculate_sasv_dcf_eers(
                sys.argv[1])
        )
    else:
        costmodel = CostModel(
            Pspf=float(sys.argv[2]),
            Pnontrg=float(sys.argv[3]),
            Ptrg=float(sys.argv[4]),
            Cmiss=float(sys.argv[5]),
            Cfa_asv=float(sys.argv[6]),
            Cfa_cm=float(sys.argv[7]),
        )
        sys.exit(calculate_sasv_dcf_eers(sys.argv[1], cost_model = costmodel))

