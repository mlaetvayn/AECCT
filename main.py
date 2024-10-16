import numpy as np
import torch
import os
import torch
import argparse
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Optional

from models import ECC_Transformer, Config, Code, freeze_weights
from dataset import EbN0_to_std, ECC_Dataset, test, train, set_seed, Get_Generator_and_Parity, CODES_PATH


def test_model(args: Config, model: torch.nn.Module, device: str):
    set_seed(args.seed)
    code = args.code
    EbNo_range_test = range(4, 7)
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(std_test))]
    test(model, device, test_dataloader_list, EbNo_range_test)


def train_model(args: Config, model: Optional[torch.nn.Module] = None):
    code = args.code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"device: {device}")
    #################################
    if model is None:
        model = ECC_Transformer(args, dropout=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=args.eta_min)

    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    EbNo_range_test = range(4, 7)
    EbNo_range_train = range(2, 8)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    train_dataloader = DataLoader(ECC_Dataset(code, std_train, len=args.batch_size * 1000, zero_cw=True), batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=args.workers)
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(std_test))]
    #################################
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss, ber, fer = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0], config=args)
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            logging.info(f"saving model with loss {loss}")
            torch.save(model.state_dict(), os.path.join(args.path, 'best_model'))

        # if epoch % 200 == 0:
        #     test(model, device, test_dataloader_list, EbNo_range_test)
    return model


def aecct_training(config: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"configuration:\n {config}")

    # phase 1
    logging.info("First phase starts...")
    phase_one_model = train_model(config)
    best_model_path = os.path.join(config.path, 'best_model')
    logging.info(f"loading best checkpoint: {best_model_path}")
    phase_one_model.load_state_dict(torch.load(best_model_path))
    logging.info(f"Post first phase evaluation:")
    test_model(config, phase_one_model, device)

    # phase 2
    config.use_aap_linear_training = True
    phase_two_model = ECC_Transformer(config).to('cuda')
    phase_two_model.load_state_dict(phase_one_model.state_dict(), strict=False)
    logging.info("Phase two starts (QAT)...")
    phase_two_model = train_model(config, model=phase_two_model)
    best_model_path = os.path.join(config.path, "best_model")
    logging.info(f"loading best checkpoint: {best_model_path}")
    phase_two_model.load_state_dict(torch.load(best_model_path))
    
    # convert to inference model
    logging.info("converting to inference model")
    config.use_aap_linear_training = False
    config.use_aap_linear_inference = True
    inference_model = ECC_Transformer(config).to(device)
    inference_model.load_state_dict(phase_two_model.state_dict(), strict=True)
    logging.info(inference_model)
    
    logging.info("freeze weights..")
    freeze_weights(inference_model, config)

    logging.info("Inference evaluation...")
    config.workers = 12
    test_model(config, inference_model, device)

    logging.info(f"output directory: {config.path}")


def preapre_args(code_hint: str = None, results_folder_name: str = "Results_AECCT", standardize: bool = True):
    args = Config()
    args.standardize = standardize
    set_seed(args.seed)

    code_files = os.listdir(CODES_PATH)
    code_files = [f for f in code_files if code_hint in f][0]
    print(code_files)
    code_n = int(code_files.split('_')[1][1:])
    code_k = int(code_files.split('_')[-1][1:].split('.')[0])
    code_type = code_files.split('_')[0]

    code = Code(code_n, code_k, code_type)

    G, H = Get_Generator_and_Parity(code, standard_form=args.standardize)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code
    output_path = os.path.join('logs',
                             results_folder_name,
                            code.code_type + '__Code_n_' + str(
                                code.n) + '_k_' + str(
                                code.k) + '__' + datetime.now().strftime(
                                "%d_%m_%Y_%H_%M_%S"))


    os.makedirs(output_path, exist_ok=True)
    args.path = output_path

    handlers = [
        logging.FileHandler(os.path.join(output_path, 'logging.txt')),
        logging.StreamHandler()
    ]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)

    logging.info(f"Path to model/logs: {output_path}")
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple argument parser example")

    parser.add_argument('--code', type=str, help="Code to train on", required=True)
    parser.add_argument('--N_dec', type=int, help="The number of Transformer blocks", required=True)
    parser.add_argument('--d_model', type=int, help="The Embedding dimension", required=True)

    args = parser.parse_args()

    if "POLAR_N128" in args.code:
        standardize = False
    else:
        standardize = True

    config = preapre_args(args.code, standardize=standardize)
    config.N_dec = args.N_dec
    config.d_model = args.d_model
    if config.N_dec == 10:
        config.epochs = 1500

    aecct_training(config)