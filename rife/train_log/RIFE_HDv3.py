import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from ..model.loss import *
from .IFNet_HDv3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.version = 4.25
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}
            else:
                return param

        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load(path)), False)
            else:
                self.flownet.load_state_dict(
                    convert(torch.load(path, map_location="cpu")),
                    False,
                )

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), "{}/flownet.pkl".format(path))

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group["lr"] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [16, 8, 4, 2, 1]
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        loss_l1 = (merged[-1] - gt).abs().mean()
        loss_smooth = self.sobel(flow[-1], flow[-1] * 0).mean()
        # loss_vgg = self.vgg(merged[-1], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_cons + loss_smooth * 0.1  # noqa: F405
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]  # noqa: F841
        return merged[-1], {
            "mask": mask,
            "flow": flow[-1][:, :2],
            "loss_l1": loss_l1,
            "loss_cons": loss_cons,  # noqa
            "loss_smooth": loss_smooth,
        }
