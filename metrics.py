"""
All the metrics assume similar input y_pred and y:
* both y_pred and y are binary masks
* numpy.ndarray
* of the same dimensionality: [H, W, D]
* function returns a dictionary in a form dict(zip(metrics_names, metrics_values))

Update:
metrics that are computed separately for CL and WML lesions (while the segmentation was 1 classes)
should take as additional input `cl_mask: np.ndarray, wml_mask: np.ndarray` that satisfy same conditions as y_pred and y,
and have the same shape as y and y_pred
"""
import numpy as np
from joblib import Parallel, delayed
from functools import partial
from scipy import ndimage
from collections import Counter
from monai.metrics import compute_average_surface_distance


def check_inputs(y_pred, y):
    def check_binary_mask(mask):
        unique = np.unique(mask)
        if np.sum(np.isin(unique, test_elements=[0.0, 1.0], invert=True)) != 0.0:
            return False
        return True

    instance = bool(isinstance(y_pred, np.ndarray) * isinstance(y, np.ndarray))

    binary_mask = bool(check_binary_mask(y_pred) * check_binary_mask(y))

    dimensionality = bool((y_pred.shape == y.shape) * (len(y_pred.shape) == 3))

    if not instance * binary_mask * dimensionality:
        raise ValueError(f"Inconsistent input to metric function. Failed in instance: {instance},"
                         f"binary mask: {binary_mask}, dimensionality: {dimensionality}.")


def DSC_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False):
    if check: check_inputs(y_pred, y)

    if np.sum(y_pred) + np.sum(y) > 0:
        return {'DSC': 2 * (y_pred * y).sum() / (y_pred + y).sum()}
    return {'DSC': 1.0}


def nDSC_metric(y_pred: np.ndarray, y: np.ndarray, r: float = 0.001, check: bool = False):
    if check: check_inputs(y_pred, y)

    if np.sum(y_pred) + np.sum(y) > 0:
        scaling_factor = 1.0 if np.sum(y) == 0 else (1 - r) * np.sum(y) / (r * (len(y.flatten()) - np.sum(y)))
        tp = np.sum(y_pred[y == 1])
        fp = np.sum(y_pred[y == 0])
        fn = np.sum(y[y_pred == 0])
        fp_scaled = scaling_factor * fp
        return {'nDSC': 2 * tp / (fp_scaled + 2 * tp + fn)}
    return {'nDSC': 1.0}


def voxel_rates_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False):
    if check: check_inputs(y_pred, y)

    if np.sum(y_pred) + np.sum(y) > 0:
        tp = np.sum(y_pred[y == 1])
        fp = np.sum(y_pred[y == 0])
        fn = np.sum(y[y_pred == 0])
        tn = np.sum(y[y_pred == 0] == 0)

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        fdr = fp / (fp + tp)
        return {'TPRvox': tpr, 'FPRvox': fpr, 'FDRvox': fdr}
    return {'TPRvox': 1.0, 'FPRvox': 0.0, 'FDRvox': 0.0}


def IoU_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False):
    if check: check_inputs(y_pred, y)
    return {'IoU': np.sum(y_pred * y) / np.sum(y_pred + y - y_pred * y)}

def IoU_adjusted_metric(cc_pred: np.ndarray, y_pred:np.ndarray=None, y: np.ndarray=None,
                        y_pred_multi: np.ndarray=None, y_multi: np.ndarray = None,
                        check: bool = False):
    if (y_pred is not None and y is not None) or (y_pred_multi is not None and y_multi is not None):
        if y_pred_multi is None and y_multi is None:
            if check:
                check_inputs(y_pred, y)
            struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
            y_pred_multi, n_les_pred = ndimage.label(y_pred - cc_pred, structure=struct_el)
            y_multi, n_les_gt = ndimage.label(y, structure=struct_el)
        # labels of the gt lesions that have an overlap with the predicted lesion
        K_prime_labels: list = np.unique(y_multi * cc_pred).tolist()
        K_prime_labels.remove(0)
        if K_prime_labels:
            K_prime = np.isin(y_multi, test_elements=K_prime_labels).astype(float)
            # labels of the predicted lesions that have an overlap with the K' lesions, except from the predicted lesion
            Q_labels: list = np.unique(K_prime * y_pred_multi).tolist()
            Q_labels.remove(0)
            Q = np.isin(y_pred_multi, test_elements=Q_labels)
            nominator = np.sum(cc_pred * K_prime)
            denominator = np.sum(cc_pred + K_prime - K_prime * Q > 0.0)
            return {'IoUadj': nominator / denominator}
        return {'IoUadj': 0.0}
    else:
        raise ValueError("Either `y_pred` and `y` or `y_pred_multi` and `y_multi` must be not none. "
                         f"Got `y_pred`: {type(y_pred)}, `y`: {type(y)}, `y_pred_multi`: {type(y_pred_multi)}, `y_multi`: {type(y_multi)}")

def F1_lesion_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False, IoU_threshold: float = 0.25,
                     n_jobs: int = None):
    def intersection_over_union(mask1, mask2):
        return np.sum(mask1 * mask2) / np.sum(mask1 + mask2 - mask1 * mask2)

    def get_tp_fp(label_pred, _mask_multi_pred, _mask_multi_gt, threshold):
        lesion_pred = (_mask_multi_pred == label_pred).astype(int)
        all_iou = [0.0]
        # iterate only intersections
        for int_label_gt in np.unique(_mask_multi_gt * lesion_pred):
            if int_label_gt != 0.0:
                lesion_gt = (_mask_multi_gt == int_label_gt).astype(int)
                all_iou.append(intersection_over_union(
                    lesion_pred, lesion_gt))
        if max(all_iou) >= threshold:
            return 'tp'
        return 'fp'

    def get_fn(label_gt, _mask_bin_pred, _mask_multi_gt):
        lesion_gt = (_mask_multi_gt == label_gt).astype(int)
        iou = intersection_over_union(lesion_gt, _mask_bin_pred)
        if iou == 0:
            return 1
        return 0

    if check: check_inputs(y_pred, y)

    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    mask_multi_pred, n_les_pred = ndimage.label(y_pred, structure=struct_el)
    mask_multi_gt, n_les_gt = ndimage.label(y, structure=struct_el)

    process_fp_tp = partial(get_tp_fp, _mask_multi_pred=mask_multi_pred, _mask_multi_gt=mask_multi_gt,
                            threshold=IoU_threshold)
    process_fn = partial(get_fn, _mask_bin_pred=y_pred, _mask_multi_gt=mask_multi_gt)

    with Parallel(n_jobs=n_jobs) as parallel:
        tp_fp = parallel(delayed(process_fp_tp)(l) for l in range(1, n_les_pred + 1))
        fn = parallel(delayed(process_fn)(l) for l in range(1, n_les_gt + 1))
    # else:
    #     tp_fp = [process_fp_tp(label_pred) for label_pred in range(1, n_les_pred + 1)]
    #     fn = [process_fn(label_gt) for label_gt in range(1, n_les_gt + 1)]

    counter = Counter(tp_fp)
    tp = float(counter['tp'])
    fp = float(counter['fp'])
    fn = float(np.sum(fn))

    tpr = tp / (tp + fn)
    fnr = fn / (tp + fn)
    fdr = fp / (tp + fp)
    f1 = 1.0 if tp + 0.5 * (fp + fn) == 0.0 else tp / (tp + 0.5 * (fp + fn))

    return {'F1les': f1, 'TPRles': tpr, 'FDRles': fdr, 'FNRles': fnr}


def ASSD_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False):
    """ Average surface symmetric distance """
    if check: check_inputs(y_pred, y)

    assd = compute_average_surface_distance(y_pred=np.expand_dims(y_pred, axis=0), y=np.expand_dims(y, axis=0),
                                            include_background=True, symmetric=True, distance_metric='euclidean')

    return {'ASSD': assd}


def ASSD_TP_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False, IoU_threshold: float = 0.25,
                   n_jobs: int = None):
    """ Average surface symmetric distance averaged across TP lesions """

    def intersection_over_union(mask1, mask2):
        return np.sum(mask1 * mask2) / np.sum(mask1 + mask2 - mask1 * mask2)

    def lesion_assd(label_pred, _mask_multi_pred, _mask_multi_gt, threshold):
        lesion_pred = (_mask_multi_pred == label_pred).astype(int)
        OHEbin = lambda x: np.expand_dims(np.stack([1 - x, x]), axis=0)
        max_iou = 0.0
        max_lesion_gt = None
        for int_label_gt in np.unique(_mask_multi_gt * lesion_pred):
            if int_label_gt != 0.0:
                lesion_gt = (_mask_multi_gt == int_label_gt).astype(int)
                iou = intersection_over_union(lesion_pred, lesion_gt)
                if iou > max_iou:
                    max_iou = iou
                    max_lesion_gt = lesion_gt
        if max_iou >= threshold:
            return compute_average_surface_distance(y_pred=OHEbin(lesion_pred), y=OHEbin(max_lesion_gt),
                                                    include_background=False, symmetric=True,
                                                    distance_metric='euclidean').item()
        return None

    if check: check_inputs(y_pred, y)

    mask_multi_pred, n_les_pred = ndimage.label(y_pred)
    mask_multi_gt, n_les_gt = ndimage.label(y)

    process_assd = partial(lesion_assd, _mask_multi_pred=mask_multi_pred, _mask_multi_gt=mask_multi_gt,
                           threshold=IoU_threshold)

    with Parallel(n_jobs=n_jobs) as parallel:
        assd_list = parallel(delayed(process_assd)(l) for l in range(1, n_les_pred))

    assd = [_ for _ in assd_list if _ is not None]

    return {'ASSD_TP_mean': np.mean(assd), 'ASSD_TP_std': np.std(assd)}


def lesTPR_metric_clwml(y_pred: np.ndarray, y: np.ndarray, cl_mask: np.ndarray, wml_mask: np.ndarray,
                        check: bool = False,
                        IoU_threshold: float = 0.25, n_jobs: int = None):
    def intersection_over_union(mask1, mask2):
        return np.sum(mask1 * mask2) / np.sum(mask1 + mask2 - mask1 * mask2)

    def get_tp_fp(label_gt, _mask_multi_pred, _mask_multi_gt, threshold):
        lesion_gt = (_mask_multi_gt == label_gt).astype(int)  # cl or wml lesion
        all_iou = [0.0]
        for int_label_pred in np.unique(_mask_multi_pred * lesion_gt):
            if int_label_pred != 0.0:
                lesion_pred = (_mask_multi_pred == int_label_pred).astype(int)
                all_iou.append(intersection_over_union(lesion_gt, lesion_pred))
        if max(all_iou) >= threshold:
            return 'tp'
        elif max(all_iou) == 0.0:
            return 'fn'
        else:
            return 'fp'

    def get_rates(types):
        counter = Counter(types)
        tp, fn, fp = float(counter['tp']), float(counter['fn']), float(counter['fp'])
        return {"TPRles": tp / (tp + fn) if tp + fn > 0.0 else 1.0,
                "FNRles": fn / (fn + tp) if tp + fn > 0.0 else 0.0,
                "UDRles": fp / (fp + tp) if fp + tp > 0.0 else 0.0}

    def update_rates(rates_glob, rates_l, type_l):
        for k, v in rates_l.items():
            rates_glob[k + '-' + type_l] = v
        return rates_glob

    if check:
        check_inputs(y_pred, y)
        check_inputs(cl_mask, wml_mask)
        if not (cl_mask.shape == y_pred.shape):
            raise ValueError(f"Mask have different shape from predictions: {cl_mask.shape}")

    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    mask_multi_pred, n_les_pred = ndimage.label(y_pred, structure=struct_el)
    mask_multi_gt_cl, n_les_gt_cl = ndimage.label(cl_mask, structure=struct_el)
    mask_multi_gt_wml, n_les_gt_wml = ndimage.label(wml_mask, structure=struct_el)

    process_fp_tp_cl = partial(get_tp_fp, _mask_multi_pred=mask_multi_pred, _mask_multi_gt=mask_multi_gt_cl,
                               threshold=IoU_threshold)
    process_fp_tp_wml = partial(get_tp_fp, _mask_multi_pred=mask_multi_pred, _mask_multi_gt=mask_multi_gt_wml,
                                threshold=IoU_threshold)

    with Parallel(n_jobs=n_jobs) as parallel:
        types_cl = parallel(delayed(process_fp_tp_cl)(l) for l in range(1, n_les_gt_cl + 1))
        types_wml = parallel(delayed(process_fp_tp_wml)(l) for l in range(1, n_les_gt_wml + 1))

    rates = update_rates(dict(), get_rates(types_cl), 'CL')
    rates = update_rates(rates, get_rates(types_wml), 'WML')

    return rates


def ASSD_TP_metric_clwml(y_pred: np.ndarray, y: np.ndarray, cl_mask: np.ndarray, wml_mask: np.ndarray,
                         check: bool = False,
                         IoU_threshold: float = 0.25, n_jobs: int = None):
    def update_rates(rates_glob, rates_l, type_l):
        for k, v in rates_l.items():
            rates_glob[k + '-' + type_l] = v
        return rates_glob

    if check:
        check_inputs(y_pred, y)
        check_inputs(cl_mask, wml_mask)
        if not (cl_mask.shape == y_pred.shape):
            raise ValueError(f"Mask have different shape from predictions: {cl_mask.shape}")

    rates_cl = ASSD_TP_metric(y_pred=y_pred, y=cl_mask, check=False, IoU_threshold=IoU_threshold, n_jobs=n_jobs)
    rates_wml = ASSD_TP_metric(y_pred=y_pred, y=wml_mask, check=False, IoU_threshold=IoU_threshold, n_jobs=n_jobs)

    rates = update_rates(dict(), rates_cl, 'CL')
    rates = update_rates(rates, rates_wml, 'WML')

    return rates


def lesion_scale_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False, IoU_threshold: float = 0.25,
                        n_jobs: int = None):
    def intersection_over_union(mask1, mask2):
        return np.sum(mask1 * mask2) / np.sum(mask1 + mask2 - mask1 * mask2)

    def get_tp_fp(label_pred, _mask_multi_pred, _mask_multi_gt, threshold):
        lesion_pred = (_mask_multi_pred == label_pred).astype(int)
        OHEbin = lambda x: np.expand_dims(np.stack([1 - x, x]), axis=0)
        max_iou = 0.0
        max_lesion_gt = None
        for int_label_gt in np.unique(_mask_multi_gt * lesion_pred):
            if int_label_gt != 0.0:
                lesion_gt = (_mask_multi_gt == int_label_gt).astype(int)
                iou = intersection_over_union(lesion_pred, lesion_gt)
                if iou > max_iou:
                    max_iou = iou
                    max_lesion_gt = lesion_gt
        assd = compute_average_surface_distance(y_pred=OHEbin(lesion_pred), y=OHEbin(max_lesion_gt),
                                                include_background=False, symmetric=True,
                                                distance_metric='euclidean').item() if max_lesion_gt is not None else None
        if max_iou >= threshold:
            return 'tp', assd, max_iou
        return 'fp', assd, max_iou

    def get_fn(label_gt, _mask_bin_pred, _mask_multi_gt):
        lesion_gt = (_mask_multi_gt == label_gt).astype(int)
        iou = intersection_over_union(lesion_gt, _mask_bin_pred)
        if iou == 0:
            return 1
        return 0

    if check: check_inputs(y_pred, y)

    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    mask_multi_pred, n_les_pred = ndimage.label(y_pred, structure=struct_el)
    mask_multi_gt, n_les_gt = ndimage.label(y, structure=struct_el)

    process_fp_tp = partial(get_tp_fp, _mask_multi_pred=mask_multi_pred, _mask_multi_gt=mask_multi_gt,
                            threshold=IoU_threshold)
    process_fn = partial(get_fn, _mask_bin_pred=y_pred, _mask_multi_gt=mask_multi_gt)

    with Parallel(n_jobs=n_jobs) as parallel:
        tp_fp_count = parallel(delayed(process_fp_tp)(l) for l in range(1, n_les_pred + 1))
        fn_count = parallel(delayed(process_fn)(l) for l in range(1, n_les_gt + 1))

    tp_fp_count, tp_fp_assd, tp_fp_iou = list(zip(*tp_fp_count))

    counter = Counter(tp_fp_count)
    tp = float(counter['tp'])
    fp = float(counter['fp'])
    fn = float(np.sum(fn_count))

    tpr = tp / (tp + fn) if tp + fn > 0.0 else 1.0
    fnr = fn / (tp + fn) if tp + fn > 0.0 else 0.0
    fdr = fp / (tp + fp) if tp + fp > 0.0 else 0.0
    f1 = 1.0 if tp + 0.5 * (fp + fn) == 0.0 else tp / (tp + 0.5 * (fp + fn))

    tp_fp_assd = [_ for _ in tp_fp_assd if _ is not None]
    tp_assd = [_ for j, _ in enumerate(tp_fp_assd) if tp_fp_count[j] == 'tp']
    fp_assd = [_ for j, _ in enumerate(tp_fp_assd) if tp_fp_count[j] == 'fp']

    tp_iou = [_ for j, _ in enumerate(tp_fp_iou) if tp_fp_count[j] == 'tp']
    fp_iou = [_ for j, _ in enumerate(tp_fp_iou) if tp_fp_count[j] == 'fp']

    return {'F1les': f1, 'TPRles': tpr, 'FDRles': fdr, 'FNRles': fnr,
            'ASSDpredles': np.mean(tp_fp_assd), 'ASSDtples': np.mean(tp_assd), 'ASSDfples': np.mean(fp_assd),
            'IoU': np.mean(tp_fp_iou), 'IoUtples': np.mean(tp_iou), 'IoUfples': np.mean(fp_iou)}


def lesion_scale_metric_clwml(y_pred: np.ndarray, y: np.ndarray, cl_mask: np.ndarray, wml_mask: np.ndarray,
                              check: bool = False, IoU_threshold: float = 0.25, n_jobs: int = None):
    def intersection_over_union(mask1, mask2):
        return np.sum(mask1 * mask2) / np.sum(mask1 + mask2 - mask1 * mask2)

    def get_tp_fp_rates(label_gt, _mask_multi_pred, _mask_multi_gt, threshold):
        lesion_gt = (_mask_multi_gt == label_gt).astype(int)
        OHEbin = lambda x: np.expand_dims(np.stack([1 - x, x]), axis=0)
        max_iou = 0.0
        max_lesion_pred = None
        for label_pred in np.unique(_mask_multi_pred * lesion_gt):
            if label_pred != 0.0:
                lesion_pred = (_mask_multi_pred == label_pred).astype(int)
                iou = intersection_over_union(lesion_pred, lesion_gt)
                if iou > max_iou:
                    max_iou = iou
                    max_lesion_pred = lesion_gt
        assd = compute_average_surface_distance(y_pred=OHEbin(max_lesion_pred), y=OHEbin(lesion_gt),
                                                include_background=False, symmetric=True,
                                                distance_metric='euclidean').item() if max_lesion_pred is not None else None
        if max_iou >= threshold:
            return 'tp', assd, max_iou
        elif max_iou == 0.0:
            return 'fn', assd, max_iou
        else:
            return 'fp', assd, max_iou

    def get_rates(res):
        if res:
            tp_fp_count, tp_fp_assd, tp_fp_iou = list(zip(*res))

            counter = Counter(tp_fp_count)
            tp = float(counter['tp'])
            fp = float(counter['fp'])
            fn = float(counter['fn'])
            tpr = tp / (tp + fn) if tp + fn > 0.0 else 1.0
            fdr = fp / (tp + fp) if tp + fp > 0.0 else 0.0
            fnr = fn / (tp + fn) if tp + fn > 0.0 else 0.0
            f1 = 1.0 if tp + 0.5 * (fp + fn) == 0.0 else tp / (tp + 0.5 * (fp + fn))

            tp_fp_assd = [_ for _ in tp_fp_assd if _ is not None]
            tp_assd = [_ for j, _ in enumerate(tp_fp_assd) if tp_fp_count[j] == 'tp']
            fp_assd = [_ for j, _ in enumerate(tp_fp_assd) if tp_fp_count[j] == 'fp']

            tp_iou = [_ for j, _ in enumerate(tp_fp_iou) if tp_fp_count[j] == 'tp']
            fp_iou = [_ for j, _ in enumerate(tp_fp_iou) if tp_fp_count[j] == 'fp']

            return {'F1les': f1, 'TPRles': tpr, 'FDRles': fdr, 'FNRles': fnr,
                    'ASSDpredles': np.mean(tp_fp_assd), 'ASSDtples': np.mean(tp_assd), 'ASSDfples': np.mean(fp_assd),
                    'IoU': np.mean(tp_fp_iou), 'IoUtples': np.mean(tp_iou), 'IoUfples': np.mean(fp_iou)}
        else:
            return {'F1les': np.nan, 'TPRles': np.nan, 'FDRles': np.nan, 'FNRles': np.nan,
                    'ASSDpredles': np.nan, 'ASSDtples': np.nan, 'ASSDfples': np.nan,
                    'IoU': np.nan, 'IoUtples': np.nan, 'IoUfples': np.nan}

    def update_rates(rates_glob, rates_l, type_l):
        for k, v in rates_l.items():
            rates_glob[k + '-' + type_l] = v
        return rates_glob

    if check: check_inputs(y_pred, y)

    if check:
        check_inputs(y_pred, y)
        check_inputs(cl_mask, wml_mask)
        if not (cl_mask.shape == y_pred.shape):
            raise ValueError(f"Mask have different shape from predictions: {cl_mask.shape}")

    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    mask_multi_pred, n_les_pred = ndimage.label(y_pred, structure=struct_el)
    mask_multi_gt_cl, n_les_gt_cl = ndimage.label(cl_mask, structure=struct_el)
    mask_multi_gt_wml, n_les_gt_wml = ndimage.label(wml_mask, structure=struct_el)

    process_cl = partial(get_tp_fp_rates,
                         _mask_multi_pred=mask_multi_pred,
                         _mask_multi_gt=mask_multi_gt_cl,
                         threshold=IoU_threshold)
    process_wml = partial(get_tp_fp_rates,
                          _mask_multi_pred=mask_multi_pred,
                          _mask_multi_gt=mask_multi_gt_wml,
                          threshold=IoU_threshold)

    with Parallel(n_jobs=n_jobs) as parallel:
        res_cl = parallel(delayed(process_cl)(l) for l in range(1, n_les_gt_cl + 1))
        res_wml = parallel(delayed(process_wml)(l) for l in range(1, n_les_gt_wml + 1))

    rates = update_rates(dict(), get_rates(res_cl), 'CL')
    rates = update_rates(rates, get_rates(res_wml), 'WML')

    return rates


def shitty_metric(y_pred: np.ndarray, y: np.ndarray, cl_mask: np.ndarray = None, wml_mask: np.ndarray = None,
                  check: bool = False, n_jobs: int = None):
    def get_tp_fp(label_pred, _mask_multi_pred, _mask_bin_gt):
        """ If a predicted lesion has at least one voxel overlap with the gt lesion - TP lesion
        If no overlap - FP lesion
        """
        lesion_pred = (_mask_multi_pred == label_pred).astype(int)
        if (lesion_pred * _mask_bin_gt).sum() > 0:
            return 'tp'
        else:
            return 'fp'

    def get_fn(label_gt, _mask_bin_pred, _mask_multi_gt):
        """If gt lesion has NO overlap with the predicted map - FN lesion """
        lesion_gt = (_mask_multi_gt == label_gt).astype(int)
        if (lesion_gt * _mask_bin_pred).sum() == 0:
            return 1
        return 0

    def get_tp_fn_clwml(label_gt, _mask_bin_pred, _mask_multi_gt):
        lesion_gt = (_mask_multi_gt == label_gt).astype(int)
        if (lesion_gt * _mask_bin_pred).sum() > .0:
            return 'tp'
        else:
            return 'fn'

    if check:
        check_inputs(y_pred, y)
        check_inputs(y_pred, y)
        check_inputs(cl_mask, wml_mask)
        if not (cl_mask.shape == y_pred.shape):
            raise ValueError(f"Mask have different shape from predictions: {cl_mask.shape}")

    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    mask_multi_pred, n_les_pred = ndimage.label(y_pred, structure=struct_el)
    mask_multi_gt, n_les_gt = ndimage.label(y, structure=struct_el)
    mask_multi_gt_cl, n_les_gt_cl = ndimage.label(cl_mask, structure=struct_el)
    mask_multi_gt_wml, n_les_gt_wml = ndimage.label(wml_mask, structure=struct_el)

    process_fp_tp = partial(get_tp_fp, _mask_multi_pred=mask_multi_pred, _mask_bin_gt=y)
    process_fn = partial(get_fn, _mask_bin_pred=y_pred, _mask_multi_gt=mask_multi_gt)
    process_tp_fn_cl = partial(get_tp_fn_clwml, _mask_bin_pred=y_pred, _mask_multi_gt=mask_multi_gt_cl)
    process_tp_fn_wml = partial(get_tp_fn_clwml, _mask_bin_pred=y_pred, _mask_multi_gt=mask_multi_gt_wml)

    with Parallel(n_jobs=n_jobs) as parallel:
        tp_fp = parallel(delayed(process_fp_tp)(l) for l in range(1, n_les_pred + 1))
        fn = parallel(delayed(process_fn)(l) for l in range(1, n_les_gt + 1))
        tp_fn_cl = parallel(delayed(process_tp_fn_cl)(l) for l in range(1, n_les_gt_cl))
        tp_fn_wml = parallel(delayed(process_tp_fn_wml)(l) for l in range(1, n_les_gt_wml))

    counter = Counter(tp_fp)
    tp, fp, fn = float(counter['tp']), float(counter['fp']), float(np.sum(fn))

    counter = Counter(tp_fn_cl)
    tp_cl, fn_cl = float(counter['tp']), float(counter['fn'])

    counter = Counter(tp_fn_wml)
    tp_wml, fn_wml = float(counter['tp']), float(counter['fn'])

    tpr = tp / (tp + fn)
    fnr = fn / (tp + fn)
    fdr = fp / (tp + fp)
    f1 = 1.0 if tp + 0.5 * (fp + fn) == 0.0 else tp / (tp + 0.5 * (fp + fn))
    tpr_cl = tp_cl / (tp_cl + fn_cl) if tp_cl + fn_cl > 0.0 else 1.0
    tpr_wml = tp_wml / (tp_wml + fn_wml) if tp_wml + fn_wml > 0.0 else 1.0

    return {'sF1les': f1,
            'sTPRles': tpr, 'sFDRles': fdr,  'sFNRles': fnr,
            'sTPRles-CL': tpr_cl, 'sFNRles-CL': 1-tpr_cl,
            'sTPRles-WML': tpr_wml, 'sFNRles-WML': 1-tpr_wml
            }

def shitty_metric_simple(y_pred: np.ndarray, y: np.ndarray,
                  check: bool = False, n_jobs: int = None):
    def get_tp_fp(label_pred, _mask_multi_pred, _mask_bin_gt):
        """ If a predicted lesion has at least one voxel overlap with the gt lesion - TP lesion
        If no overlap - FP lesion
        """
        lesion_pred = (_mask_multi_pred == label_pred).astype(int)
        if (lesion_pred * _mask_bin_gt).sum() > 0:
            return 'tp'
        else:
            return 'fp'

    def get_fn(label_gt, _mask_bin_pred, _mask_multi_gt):
        """If gt lesion has NO overlap with the predicted map - FN lesion """
        lesion_gt = (_mask_multi_gt == label_gt).astype(int)
        if (lesion_gt * _mask_bin_pred).sum() == 0:
            return 1
        return 0

    if check:
        check_inputs(y_pred, y)
        check_inputs(y_pred, y)

    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    mask_multi_pred, n_les_pred = ndimage.label(y_pred, structure=struct_el)
    mask_multi_gt, n_les_gt = ndimage.label(y, structure=struct_el)

    process_fp_tp = partial(get_tp_fp, _mask_multi_pred=mask_multi_pred, _mask_bin_gt=y)
    process_fn = partial(get_fn, _mask_bin_pred=y_pred, _mask_multi_gt=mask_multi_gt)

    with Parallel(n_jobs=n_jobs) as parallel:
        tp_fp = parallel(delayed(process_fp_tp)(l) for l in range(1, n_les_pred + 1))
        fn = parallel(delayed(process_fn)(l) for l in range(1, n_les_gt + 1))

    counter = Counter(tp_fp)
    tp, fp, fn = float(counter['tp']), float(counter['fp']), float(np.sum(fn))

    tpr = tp / (tp + fn)
    fnr = fn / (tp + fn)
    fdr = fp / (tp + fp)
    f1 = 1.0 if tp + 0.5 * (fp + fn) == 0.0 else tp / (tp + 0.5 * (fp + fn))

    return {'sF1les': f1,
            'sTPRles': tpr, 'sFDRles': fdr,  'sFNRles': fnr
            }
