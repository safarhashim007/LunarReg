import torch

from romav2 import RoMaV2
from romav2.device import device


# RoMa v2 requires this
torch.set_float32_matmul_precision("highest")


CHANDRAYAAN = "data/test_pairs/pair_002/chandrayaan.png"
LRO = "data/test_pairs/pair_002/lro.png"


print("===== RoMa v2 Lunar Test =====")
print("Device:", device)

print()
print("Loading model...")

model = RoMaV2()

# First test = lower memory / faster.
model.apply_setting("fast")

print("Model loaded.")
print("Setting: fast")
print("Low resolution:", model.H_lr, "x", model.W_lr)
print("High resolution:", model.H_hr, model.W_hr)
print("Bidirectional:", model.bidirectional)

print()
print("Running dense matching...")

preds = model.match(
    CHANDRAYAAN,
    LRO
)

print()
print("Matching completed ✅")

print()
print("Prediction tensors:")

for key, value in preds.items():

    if value is None:
        print(key, ": None")

    else:
        print(
            key,
            ":",
            tuple(value.shape),
            value.device
        )


overlap = preds["overlap_AB"]

print()
print("Overlap confidence:")
print("Minimum:", overlap.min().item())
print("Maximum:", overlap.max().item())
print("Mean   :", overlap.mean().item())

print()
print("============================")