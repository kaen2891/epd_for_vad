import torch
import torchaudio


input_wav = torch.rand(1, 320000)

fbank = torchaudio.compliance.kaldi.fbank(input_wav, frame_length=25.0, frame_shift=10.0, num_mel_bins=n_mels, sample_frequency=sr, use_log_fbank=True)


fbank = torchaudio.compliance.kaldi.fbank(input_wav, blackman_coeff=0.42, channel=-1, dither=1.0, energy_floor=0.0, frame_length=25.0, frame_shift=10.0, high_freq=0.0, htk_compat=True, low_freq=20.0, min_duration=0.0, num_mel_bins=40, preemphasis_coefficient=0.97, raw_energy=True, remove_dc_offset=True, round_to_power_of_two=True, sample_frequency=16000.0, snip_edges=True, subtract_mean=False, use_energy=False, use_log_fbank=True,use_power=True, vtln_high=-500.0, vtln_low=100.0, vtln_warp=1.0, window_type='hamming')