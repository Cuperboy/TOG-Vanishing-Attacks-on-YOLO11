from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace
import torch
import numpy as np


def Compute_grad(im):
    """ Computing vanishing grad """
    img = im.clone()
    detector = YOLO("yolo11n.pt")
    core_model = detector.model
    core_model.eval()
    core_model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    train_batch = {'cls': torch.zeros((0, 1)), 'bboxes': torch.zeros((0, 4)), 'batch_idx': torch.zeros(0)}
    img.requires_grad_()
    loss, _ = core_model.loss(train_batch, core_model(img))
    loss.backward()
    loss.grad = None
    return img.grad

def tog_vanishing(x_query, n_iter=10, eps=4/255., eps_iter=1/255.):
    """
    TOG-Vanishing Attack

    Args:
            x_query (torch.tensor): Input image tensor
            n_iter (int): number of attack iterations
            eps (float): noise average difference
            eps_iter(float): momentum factor

    Output:
            tuple(
            torch.tensor: image with noise
            torch.tensor: TOG-Vanishing noise
            )
    """
    x_query = x_query.to("cpu")
    eta = torch.from_numpy(np.random.uniform(-eps, eps, size=x_query.size())).float()
    for _ in range(n_iter):
        x_adv = torch.clip(x_query + eta, 0.0, 1.0).float()
        grad = Compute_grad(x_adv)
        signed_grad = torch.sign(grad)
        eta = torch.clip(eta - eps_iter * signed_grad, -eps, eps)
    x_adv = torch.clip(x_query + eta, 0.0, 1.0).float()
    return x_adv, eta