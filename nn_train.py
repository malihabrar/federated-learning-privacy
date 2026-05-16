# %%
import shelve
import argparse
import os

from src.prepare import prepare
from src.train_v2 import FederatedLearning
import src.config as config

# Parse input arguments
parser = argparse.ArgumentParser(description="Run federated learning experiments with PLGP.")
parser.add_argument("--alpha", type=lambda x: round(float(x), 2), default=1,
                    help="Alpha value for PLGP (floating point with 2 decimals). Default is None.")
parser.add_argument("--seed", type=int, default=10,
                    help="Seed for experiments. Default is 10.")
parser.add_argument("--method", type=str, default='prune',
                    help="Pruning method")
parser.add_argument("--model", type=str, default='lenet',
                    help="Model architecture: 'lenet' or 'vit'")
parser.add_argument("--max_rounds", type=int, default=10,
                    help="Number of federated learning rounds. Default is 10.")


args = parser.parse_args()
alpha = args.alpha
seed = args.seed
method = args.method

# store the arguments in config
config.alpha = args.alpha
config.seed = args.seed
config.method = args.method
config.model_name = args.model
config.reinit_model()

print(f"Running experiment with alpha: {alpha} and seed: {seed}")
if not os.path.exists("results/bin"):
    os.makedirs("results/bin")


idlg_prep_exp = False
if alpha == 1 and method == "prune":
    # idlg only needs to be setup once per seed, since it is the same for all other values,
    # the randomness is in the data loaders, and the model is stored as checkpoints
    idlg_prep_exp = True


# %%
iid_client_train_loader, device, criterion, validation_loader, train_loader, clients_dataset, global_model = prepare(seed=seed)

# ViT is a large pretrained model — use a much smaller learning rate and fewer local epochs
lr = 1e-4 if args.model == "vit" else 5e-3
num_local_epochs = 3 if args.model == "vit" else 10

federated_learning = FederatedLearning(
    model=global_model,
    clients_dataloaders=iid_client_train_loader,
    num_clients_per_round=2,
    num_local_epochs=num_local_epochs,
    lr=lr,
    max_rounds=args.max_rounds,
    device=device,
    criterion=criterion,
    test_dataloader=validation_loader,
    train_dataloader=train_loader,
    filtered_train_dataset=clients_dataset,
    alpha=alpha,
    prune_method=method,
)

train_accuracy, val_accuracy, psnr, ssim = federated_learning.federated_learning_experiment(idlg_prep_exp=idlg_prep_exp)

shelve_file = f"results/bin/pruning_exp_model_{args.model}_seed_{seed}_alpha_{alpha:.2f}_method_{method}"
print(f"Using shelve file: {shelve_file}")

d = shelve.open(shelve_file)
d["seed"] = seed
d["alpha"] = alpha
d["method"] = method

d["train_accuracy"] = train_accuracy
d["val_accuracy"] = val_accuracy
d["psnr"] = psnr
d["ssim"] = ssim

d.close()
