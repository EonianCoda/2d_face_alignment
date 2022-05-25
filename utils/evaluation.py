import torch

def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    # batchsize * numPoints * 2
    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(2)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def accuracy(output, target, idxs, thr=0.08):
    ''' Calculate accuracy according to NME, but uses ground truth heatmap rather than x,y locations
    First value to be returned is accuracy calculated based on overall 'idxs'
    followed by individual accuracies
    '''
    preds = get_preds(output)
    gts = get_preds(target)
    # B * 2
    norm = torch.ones(preds.size(0))
    for i, gt in enumerate(gts):
        norm[i] = _get_bboxsize(gt)

    dists = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs) + 1)
    avg_acc = 0
    cnt = 0

    mean_dists = torch.mean(dists, 0)
    acc[0] = mean_dists.le(thr).sum() * 1.0 / preds.size(0)
    # for i in range(len(idxs)):
    #     acc[i+1] = dist_acc(dists[idxs[i]-1], thr=thr)
    #     if acc[i+1] >= 0:
    #         avg_acc = avg_acc + acc[i+1]
    #         cnt += 1

    # if cnt != 0:
    #     acc[0] = avg_acc / cnt
    return acc, dists