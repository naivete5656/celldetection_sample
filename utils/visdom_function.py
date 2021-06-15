import torch
import visdom


class VisdomClass(object):
    def create_vis_show(self, batch_size):
        return self.vis.images(
            torch.ones((batch_size, 1, 256, 256)), batch_size
        )

    def update_vis_show(self, images, window1, batch_size):
        self.vis.images(images, batch_size, win=window1)

    def create_vis_plot(self, _xlabel, _ylabel, _title, _legend):
        return self.vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(xlabel=_xlabel, ylabel=_ylabel, title=_title, legend=_legend),
        )

    def update_vis_plot(self, iteration, loss, window1, update_type):
        self.vis.line(
            X=torch.ones((1)).cpu() * iteration,
            Y=torch.Tensor(loss).unsqueeze(0).cpu(),
            win=window1,
            update=update_type,
        )

    def vis_show_result(self, iteration, loss, mask_preds, imgs, true_masks, batch_size):
        self.update_vis_plot(iteration, [loss.item()], self.iter_plot, "append")
        mask_preds = (mask_preds - mask_preds.min()) / (mask_preds.max() - mask_preds.min())
        self.update_vis_show(imgs, self.ori_view, batch_size)
        self.update_vis_show(mask_preds[:, :1], self.pred_view, batch_size)
        self.update_vis_show(mask_preds[:, 1:], self.pred_view2, batch_size)
        self.update_vis_show(true_masks[:, :1], self.gt_view, batch_size)
        self.update_vis_show(true_masks[:, 1:], self.gt_view2, batch_size)

    def vis_init(self, env, barch_size):
        HOSTNAME = "localhost"
        PORT = 8097

        self.vis = visdom.Visdom(port=PORT, server=HOSTNAME, env=env)

        vis_title = "ctc"
        vis_legend = ["Loss"]
        vis_epoch_legend = ["Loss", "Val Loss"]

        self.iter_plot = self.create_vis_plot(
            "Iteration", "Loss", vis_title, vis_legend
        )
        self.epoch_plot = self.create_vis_plot(
            "Epoch", "Loss", vis_title, vis_epoch_legend
        )
        self.ori_view = self.create_vis_show(barch_size)
        self.gt_view = self.create_vis_show(barch_size)
        self.gt_view2 = self.create_vis_show(barch_size)
        self.pred_view = self.create_vis_show(barch_size)
        self.pred_view2 = self.create_vis_show(barch_size)
        self.bg_mask_view = self.create_vis_show(barch_size)
