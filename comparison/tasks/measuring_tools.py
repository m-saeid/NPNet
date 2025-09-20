import torch
import argparse
from fvcore.nn import FlopCountAnalysis
from torch.autograd import Variable
import numpy as np

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('eval')
    parser.add_argument('--model', type=str, default='npnet',
                        help='model: [npnet, pointnn, pointgn]')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--workers', default=6, type=int, help='workers')
    return parser.parse_args()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y

def params(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}")
    return trainable


def gflops(model: torch.nn.Module, inputs) -> float:
    """
    Compute GFLOPs using fvcore.
    """
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, inputs)
        total = flops.total()
    gflops = total / 1e9
    print(f"Total FLOPs: {gflops:.3f} GFLOPs")
    return gflops


def memory(model: torch.nn.Module, inputs) -> float:
    """
    Measure peak GPU memory during a single forward pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if isinstance(inputs, tuple):
        args = tuple(inp.to(device) for inp in inputs)
    else:
        args = (inputs.to(device),)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(*args) if len(args) > 1 else model(args[0])

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak GPU memory: {peak_mb:.2f} MB")
    return peak_mb



def inference_time(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    bs,
    model_name,
    num_warmup_batches: int = 5,
    max_batches: int = None,
    ) -> float:
    """
    Measures average inference time per batch over a DataLoader.
    Benchmark with different batch sizes: Data loading time grows with batch size, so test what works best for your hardware
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Warm-up phase to stabilize performance
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_warmup_batches:
                break
            if model_name in ['npnet', 'pointgn']:
                data = data.to(device)
            else:
                data = (data.to(device).permute(0,2,1), )
            #_ = model(data[0].permute(0,2,1)) if model_name in ['npnet', 'pointnn', 'pointgn'] else model(*data)
            _ = model(data) if model_name in ['npnet', 'pointgn'] else model(*data)

    timings = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            if model_name in ['npnet', 'pointgn']:
                data = data.to(device)
            else:
                data = (data.to(device).permute(0,2,1), )

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(data) if model_name in ['npnet', 'pointgn'] else model(*data)
            end.record()
            torch.cuda.synchronize()
            delta = start.elapsed_time(end)

            timings.append(delta)

    avg_ms = sum(timings) / (len(timings)*bs)

    print(f"Avg inference time over {len(timings)} runs, each with {bs} batches: {avg_ms:.3f} ms per sample")
    return avg_ms



def batch_prep(data, model_name):  # data [B, N, 3]

    device = 'cuda'

    if model_name in ['npnet', 'pointnn']:
        return data[0].permute(0,2,1).to(device)



def inference_time_shapenet(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    bs,
    model_name,
    num_warmup_batches: int = 5,
    max_batches: int = None,
    ) -> float:
    """
    Measures average inference time per batch over a DataLoader.
    Benchmark with different batch sizes: Data loading time grows with batch size, so test what works best for your hardware
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Warm-up phase to stabilize performance
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_warmup_batches:  # <-- add this
                break
            data = batch_prep(data, model_name)   
            model(data) if model_name in ['npnet', 'pointnn'] else model(*data)

    timings = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            data = batch_prep(data, model_name)    
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(data) if model_name in ['npnet', 'pointnn'] else model(*data)
            end.record()
            torch.cuda.synchronize()
            delta = start.elapsed_time(end)
            timings.append(delta)
            
    avg_ms = sum(timings) / (len(timings)*bs)

    print(f"Avg inference time over {len(timings)} runs, each with {bs} batches: {avg_ms:.3f} ms per sample")
    return avg_ms
