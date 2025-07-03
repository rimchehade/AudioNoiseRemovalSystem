import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from model import Generator
import csv

def emphasis(signal, emph_coeff=0.95, pre=True):
    if pre:
        first = signal[:, :, :1]
        diff  = signal[:, :, 1:] - emph_coeff * signal[:, :, :-1]
        return np.concatenate((first, diff), axis=2)
    else:
        output = signal.copy()
        for t in range(1, signal.shape[-1]):
            output[:, :, t] = signal[:, :, t] + emph_coeff * output[:, :, t-1]
        return output

class TestDataset(Dataset):
    def __init__(self, serialized_test_folder):
        self.files = [
            os.path.join(serialized_test_folder, f)
            for f in os.listdir(serialized_test_folder)
            if f.endswith('.npy')
        ]

    def __getitem__(self, idx):
        clean, noisy = np.load(self.files[idx])
        clean = clean[np.newaxis, :] 
        noisy = noisy[np.newaxis, :]  
        noisy_emph = noisy[np.newaxis, :, :]
        noisy_emph = emphasis(noisy_emph).reshape(1, -1)
        return (
            torch.from_numpy(clean).float(),
            torch.from_numpy(noisy_emph).float()
        )

    def __len__(self):
        return len(self.files)

def compute_snr(clean, enh, eps=1e-10):
    noise = clean - enh
    return 10 * np.log10((np.sum(clean**2) + eps) / (np.sum(noise**2) + eps))

def test_epoch(epoch, generator, device, sample_rate, serialized_test_folder, batch_size=50):
    model_path = f'C:/Users/timan/segan/pytorch_segan/epochs/generator-{epoch}.pkl'
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    generator.to(device).eval()

    loader = DataLoader(
        TestDataset(serialized_test_folder),
        batch_size=batch_size, shuffle=False
    )

    all_pesq, all_stoi, all_snr = [], [], []

    for clean_b, noisy_b in tqdm(loader, desc=f"Eval epoch {epoch}"):
        clean_b = clean_b.to(device)   
        noisy_b = noisy_b.to(device)
        z = torch.randn(clean_b.size(0), 1024, 8, device=device)

        with torch.no_grad():
            enh_b = generator(noisy_b, z).cpu().numpy() 

        # de-emphasis
        enh_b = emphasis(enh_b, pre=False).reshape(-1, enh_b.shape[-1])
        cln  = clean_b.cpu().numpy().reshape(-1, clean_b.shape[-1])

        for c, e in zip(cln, enh_b):
            try:   all_pesq.append(pesq(sample_rate, c, e, 'wb'))
            except: all_pesq.append(np.nan)
            try:   all_stoi.append(stoi(c, e, sample_rate, extended=False))
            except: all_stoi.append(np.nan)
            all_snr.append(compute_snr(c, e))

    return {
        'pesq': np.nanmean(all_pesq),
        'stoi': np.nanmean(all_stoi),
        'snr':  np.nanmean(all_snr)
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator()
    sample_rate = 16000  
    serialized_test_folder = "C:/Users/timan/segan/pytorch_segan/serialized_test_data"  
    num_epochs = 50 

    results = []

    for epoch in range(1, num_epochs + 1):
        try:
            avg_metrics = test_epoch(
                epoch=epoch,
                generator=generator,
                device=device,
                sample_rate=sample_rate,
                serialized_test_folder=serialized_test_folder
            )
            print(f"Epoch {epoch}:")
            for metric, value in avg_metrics.items():
                print(f"{metric.upper()}: {value:.4f}")

            results.append({
                'epoch': epoch,
                'pesq': avg_metrics['pesq'],
                'stoi': avg_metrics['stoi'],
                'snr': avg_metrics['snr']
            })

        except Exception as e:
            print(f"Error evaluating epoch {epoch}: {e}")
            results.append({
                'epoch': epoch,
                'pesq': None,
                'stoi': None,
                'snr': None
            })

    results_path = "C:/Users/timan/segan/pytorch_segan/test_metrics.csv"  
    with open(results_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'pesq', 'stoi', 'snr'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {results_path}")