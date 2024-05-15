from torchaudio.pipelines import SQUIM_OBJECTIVE
import torch 
import torchaudio

model = None

def squim_apply(batch, rank=None, audio_column_name="audio"):
    global model
    if model is None:
        model = SQUIM_OBJECTIVE.get_model()
    if rank is not None:
        # move the model to the right GPU if not there already
        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        # move to device and create pipeline here because the pipeline moves to the first GPU it finds anyway
        model.to(device)
    else:
        device = "cpu"

    if isinstance(batch[audio_column_name], list):  
        sdr = []
        pesq = []
        stoi = []
        for sample in batch[audio_column_name]:
            waveform = torchaudio.functional.resample(torch.tensor(sample["array"][None, :]).to(device).float(), sample["sampling_rate"], SQUIM_OBJECTIVE.sample_rate)
            with torch.no_grad():
                stoi_sample, pesq_sample, sdr_sample = model(waveform)
            sdr.append(sdr_sample.cpu())
            pesq.append(pesq_sample.cpu())
            stoi.append(stoi_sample.cpu())

        batch["sdr"] = sdr
        batch["pesq"] = pesq
        batch["stoi"] = stoi
    else:
    
        waveform = torchaudio.functional.resample(torch.tensor(batch[audio_column_name]["array"][None, :]).to(device).float(), batch[audio_column_name]["sampling_rate"], SQUIM_OBJECTIVE.sample_rate)
        with torch.no_grad():
            stoi_sample, pesq_sample, sdr_sample = model(waveform)
        batch["sdr"] = sdr_sample
        batch["pesq"] = pesq_sample
        batch["stoi"] = stoi_sample
        # TODO
    return batch

