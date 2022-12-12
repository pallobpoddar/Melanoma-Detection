import datetime
import torch
from tqdm import tqdm
from ..utils import AverageMeter

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    _xla_available = True
except ImportError:
    _xla_available = False
try:
    from apex import amp

    _apex_available = True
except ImportError:
    _apex_available = False


def reduce_fn(vals):
    return sum(vals) / len(vals)


class Engine:
    def __init__(
        self,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1,
        use_tpu=False,
        tpu_print=10,
        fp16=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
        self.use_tpu = use_tpu
        self.tpu_print = tpu_print
        self.fp16 = fp16

        if self.use_tpu and not _xla_available:
            raise Exception(
                "You want to use TPUs but you dont have pytorch_xla installed"
            )
        if self.fp16 and not _apex_available:
            raise Exception("You want to use fp16 but you dont have apex installed")
        if self.fp16 and use_tpu:
            raise Exception("Apex fp16 is not available when using TPUs")
        if self.fp16:
            self.accumulation_steps = 1

    def train(self, data_loader):
        losses = AverageMeter()
        self.model.train()
        print_idx = int(len(data_loader) * self.tpu_print / 100)
        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()
        if self.use_tpu:
            para_loader = pl.ParallelLoader(data_loader, [self.device])
            tk0 = para_loader.per_device_loader(self.device)
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))

        for b_idx, data in enumerate(tk0):
            if self.accumulation_steps == 1 and b_idx == 0:
                self.optimizer.zero_grad()
            
            if self.model_fn is None:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                _, loss = self.model(**data)
                
            else:
                if self.fp16:
                    with amp.autocast():
                        loss = self.model_fn(data, self.device, self.model)
                else:
                    loss = self.model_fn(data, self.device, self.model)

            if not self.use_tpu:
                with torch.set_grad_enabled(True):
                    if self.use_mean_loss:
                        loss = loss.mean()
                        
                    if self.fp16:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                        
                    if (b_idx + 1) % self.accumulation_steps == 0:
                        if self.fp16:
                            self.scaler.step(self.optimizer)
                        else:
                            self.optimizer.step()
                            
                        if self.scheduler is not None:
                            self.scheduler.step()
                        if b_idx > 0:
                            self.optimizer.zero_grad()
            else:
                loss.backward()
                xm.optimizer_step(self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                if b_idx > 0:
                    self.optimizer.zero_grad()
            if self.use_tpu:
                reduced_loss = xm.mesh_reduce("loss_reduce", loss, reduce_fn)
                losses.update(reduced_loss.item(), data_loader.batch_size)
            else:
                losses.update(loss.item(), data_loader.batch_size)

            if not self.use_tpu:
                tk0.set_postfix(loss=losses.avg)
            else:
                if b_idx % print_idx == 0:
                    xm.master_print(
                        f"{datetime.datetime.now()}: Batch {b_idx} / {len(data_loader)}, loss={losses.avg}"
                    )
        if not self.use_tpu:
            tk0.close()
        return losses.avg

    def evaluate(self, data_loader):
        losses = AverageMeter()
        final_predictions = []
        print_idx = int(len(data_loader) * self.tpu_print / 100)
        self.model.eval()
        with torch.no_grad():
            if self.use_tpu:
                para_loader = pl.ParallelLoader(data_loader, [self.device])
                tk0 = para_loader.per_device_loader(self.device)
            else:
                tk0 = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(self.device)
                predictions, loss = self.model(**data)
                predictions = predictions.cpu()
                if self.use_tpu:
                    reduced_loss = xm.mesh_reduce("loss_reduce", loss, reduce_fn)
                    losses.update(reduced_loss.item(), data_loader.batch_size)
                else:
                    losses.update(loss.item(), data_loader.batch_size)
                final_predictions.append(predictions)
                if not self.use_tpu:
                    tk0.set_postfix(loss=losses.avg)
                else:
                    if b_idx % print_idx == 0:
                        xm.master_print(
                            f"{datetime.datetime.now()}: Batch {b_idx} / {len(data_loader)}, loss={losses.avg}"
                        )
            if not self.use_tpu:
                tk0.close()
        return final_predictions, losses.avg

    def predict(self, data_loader):
        self.model.eval()
        final_predictions = []
        if self.use_tpu:
            raise Exception("TPU not available for predict yet!")
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for data in tk0:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                predictions, _ = self.model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions
