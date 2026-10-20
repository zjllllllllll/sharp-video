import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gaussian_lifecycle import read_lifecycle_animation
from .data import prepare, epoch_batches, batches_per_epoch
from .model import AttributeDecoder, FORMAT, GROUPS, default_config, group_losses, physical_attributes, load_package
from .state import State, atomic_torch_save


def train(run_dir):
    run_dir = Path(run_dir)
    output = run_dir / "trajectory-codec"
    state = State(output)
    config = default_config()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.set_num_threads(8)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    device = torch.device("cuda")
    source = run_dir / "tracks-lifecycle" / "animation.4dgs"
    prepared_path = output / "prepared.pt"
    if not prepared_path.exists():
        state.update(status="running", stage="preparing", config=config, currentFrame=0)
        prepared = prepare(source, lambda n, total: state.update(currentFrame=n, frameCount=total), state.stopped)
        atomic_torch_save(prepared, prepared_path)
    else:
        prepared = torch.load(prepared_path, weights_only=True)
    header = prepared["header"]
    life = prepared["lifecycles"].numpy()
    count = len(life) * header["layerCount"]
    network = AttributeDecoder(config).to(device)
    codes = nn.Embedding(count, config["latent_dim"], sparse=True, device=device)
    nn.init.normal_(codes.weight, std=1 / math.sqrt(config["latent_dim"]))
    optimizer = torch.optim.Adam(network.parameters(), lr=config["network_learning_rate"])
    code_optimizer = torch.optim.SparseAdam(codes.parameters(), lr=config["code_learning_rate"])
    mean, std = prepared["mean"].to(device), prepared["std"].to(device)
    checkpoint_path = output / "checkpoint.pt"
    steps = batches_per_epoch(header, config["batch_size"])
    total_steps = steps * config["epochs"]
    step, previous_seconds = 0, 0.
    micro = config["micro_batch_size"]
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu", mmap=True)
        if checkpoint["config"] != config:
            raise ValueError("Resume configuration differs from the saved experiment")
        network.load_state_dict(checkpoint["network"])
        with torch.no_grad():
            codes.weight.copy_(checkpoint["codes"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        code_optimizer.load_state_dict(checkpoint["code_optimizer"])
        step, previous_seconds = checkpoint["step"], checkpoint["trainingSeconds"]
        micro = checkpoint["micro_batch_size"]
        torch.set_rng_state(checkpoint["rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        del checkpoint
        gc.collect()
    started = time.perf_counter()
    state.update(status="running", stage="training", config=config, globalStep=step, totalSteps=total_steps,
                 epoch=step // steps, epochs=config["epochs"], sampleCount=prepared["sample_count"],
                 gaussianTrackCount=count, networkParameters=sum(p.numel() for p in network.parameters()),
                 latentParameters=count * config["latent_dim"], microBatchSize=micro,
                 trainingSeconds=previous_seconds, frameCount=header["frameCount"], sourceBytes=source.stat().st_size)

    def seconds():
        return previous_seconds + time.perf_counter() - started

    def save_checkpoint():
        state.update(stage="checkpoint", globalStep=step, trainingSeconds=seconds())
        atomic_torch_save(dict(config=config, network=network.state_dict(), codes=codes.weight.detach(),
                               optimizer=optimizer.state_dict(), code_optimizer=code_optimizer.state_dict(),
                               step=step, trainingSeconds=seconds(), micro_batch_size=micro,
                               rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), checkpoint_path)
        state.update(checkpointStep=step, checkpointBytes=checkpoint_path.stat().st_size,
                     resumable=True, stage="training")

    _, frames, lifetimes = read_lifecycle_animation(source)
    log = output / "loss.jsonl"
    # On crash recovery, discard log entries newer than the durable checkpoint.
    if log.exists():
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        log.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= step))
    sums = np.zeros(5, dtype=np.float64)
    logged_batches = 0
    try:
        for epoch in range(step // steps, config["epochs"]):
            skip = step % steps
            for batch_index, frame, identities, tau, attributes in epoch_batches(header, frames, life, config, epoch, skip):
                if state.stopped():
                    save_checkpoint()
                    state.update(status="stopped", trainingSeconds=seconds(), globalStep=step)
                    return
                # Retrying a microbatch OOM never advances the optimizer or sampler.
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    code_optimizer.zero_grad(set_to_none=True)
                    batch_losses = torch.zeros(5, device=device)
                    try:
                        for begin in range(0, len(identities), micro):
                            end = min(begin + micro, len(identities))
                            identity = torch.from_numpy(identities[begin:end]).to(device)
                            t = torch.from_numpy(tau[begin:end]).to(device)
                            target = torch.from_numpy(attributes[begin:end]).to(device)
                            prediction = network(codes(identity), t)
                            losses = group_losses(prediction, target, mean, std)
                            fraction = (end - begin) / len(identities)
                            (losses.sum() * fraction).backward()
                            batch_losses += losses.detach() * fraction
                        break
                    except torch.cuda.OutOfMemoryError:
                        if micro == 1:
                            raise
                        optimizer.zero_grad(set_to_none=True)
                        code_optimizer.zero_grad(set_to_none=True)
                        prediction = losses = target = identity = t = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        micro = max(1, micro // 2)
                        state.update(microBatchSize=micro)
                if not torch.isfinite(batch_losses).all():
                    raise FloatingPointError(f"Nonfinite reconstruction loss at step {step}")
                optimizer.step()
                code_optimizer.step()
                step += 1
                sums += batch_losses.detach().cpu().numpy()
                logged_batches += 1
                if step % 20 == 0 or step == 1 or step % steps == 0:
                    values = dict(zip(GROUPS, (sums / logged_batches).tolist()))
                    row = dict(step=step, epoch=step / steps, loss=sum(values.values()),
                               groups=values, trainingSeconds=seconds())
                    with log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row) + "\n")
                    state.update(globalStep=step, epoch=step // steps, epochFraction=step / steps,
                                 currentFrame=frame, loss=row["loss"], losses=values, trainingSeconds=seconds(),
                                 microBatchSize=micro, samplesVisited=step * config["batch_size"],
                                 peakGpuBytes=torch.cuda.max_memory_allocated())
                    sums[:] = 0
                    logged_batches = 0
                if step % (steps * config["checkpoint_epochs"]) == 0:
                    save_checkpoint()
        state.update(stage="exporting", globalStep=step, epoch=config["epochs"], trainingSeconds=seconds())
        network.eval()
        with torch.no_grad():
            probe_ids = torch.arange(min(count, 32), device=device)
            probe_t = torch.linspace(-1, 1, len(probe_ids), device=device)[:, None]
            expected = physical_attributes(network(codes(probe_ids), probe_t), mean, std).cpu()
        package = dict(format=FORMAT, config=config, network={k: v.detach().cpu() for k, v in network.state_dict().items()},
                       codes=codes.weight.detach().cpu(), lifecycles=prepared["lifecycles"],
                       mean=prepared["mean"], std=prepared["std"], header=header,
                       provenance={"siren-pytorch": "0.1.7", "architecture": "Modulated SIREN", "training": "auto-decoder"})
        final_path = output / "trajectory-codec.pt"
        atomic_torch_save(package, final_path)
        del package
        reloaded, restored = load_package(final_path, device)
        with torch.no_grad():
            actual = physical_attributes(restored(reloaded["codes"][:len(probe_ids)].to(device), probe_t), mean, std).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        state.update(exportComplete=True, exportReloadMaxError=float((actual - expected).abs().max()),
                     modelFile="trajectory-codec.pt", modelBytes=final_path.stat().st_size,
                     codeBytes=count * config["latent_dim"] * 4,
                     networkBytes=sum(p.numel() * p.element_size() for p in network.parameters()),
                     lifecycleBytes=prepared["lifecycles"].numel() * prepared["lifecycles"].element_size(),
                     compressionRatio=source.stat().st_size / final_path.stat().st_size,
                     trainingSeconds=seconds(), samplesVisited=prepared["sample_count"] * config["epochs"],
                     stage="exported")
    finally:
        frames._mmap.close()
        lifetimes._mmap.close()
