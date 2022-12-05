import torch
from tqdm import tqdm
from ..utils import AverageMeter

try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False

try:
    from apex import amp

    _apex_available = True
except ImportError:
    _apex_available = False


class RCNNEngine:
    @staticmethod
    def train(
        data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1,
        use_tpu=False,
        fp16=False,
    ):
        if use_tpu and not _xla_available:
            raise Exception(
                "You want to use TPUs but you dont have pytorch_xla installed"
            )
        if fp16 and not _apex_available:
            raise Exception("You want to use fp16 but you dont have apex installed")
        if fp16 and use_tpu:
            raise Exception("Apex fp16 is not available when using TPUs")
        if fp16:
            accumulation_steps = 1
        losses = AverageMeter()
        predictions = []
        model.train()
        if accumulation_steps > 1:
            optimizer.zero_grad()
        tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
        for b_idx, data in enumerate(tk0):
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if accumulation_steps == 1 and b_idx == 0:
                optimizer.zero_grad()
            loss = model(images, targets)

            if not use_tpu:
                with torch.set_grad_enabled(True):
                    if fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    if (b_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        if b_idx > 0:
                            optimizer.zero_grad()
            else:
                loss.backward()
                xm.optimizer_step(optimizer)
                if scheduler is not None:
                    scheduler.step()
                if b_idx > 0:
                    optimizer.zero_grad()

            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)
        return losses.avg

    @staticmethod
    def predict(data_loader, model, device, use_tpu=False):
        model.eval()
        final_predictions = []
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            for b_idx, data in enumerate(tk0):
                images, targets = data
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images, targets)
                predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]
                final_predictions.append(predictions)
        final_predictions = [item for sublist in final_predictions for item in sublist]
        return final_predictions
