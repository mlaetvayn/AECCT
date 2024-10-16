import numpy as np
import torch
import os
import torch
import random
from torch.utils import data
import logging
import time
from tqdm import tqdm

from configuration import Config


CODES_PATH = "codes/"

def Read_pc_matrixrix_alist(fileName):
    with open(fileName, 'r') as file:
        lines = file.readlines()
        columnNum, rowNum = np.fromstring(
            lines[0].rstrip('\n'), dtype=int, sep=' ')
        H = np.zeros((rowNum, columnNum)).astype(int)
        for column in range(4, 4 + columnNum):
            nonZeroEntries = np.fromstring(
                lines[column].rstrip('\n'), dtype=int, sep=' ')
            for row in nonZeroEntries:
                if row > 0:
                    H[row - 1, column - 4] = 1
        return H
#############################################
def row_reduce(mat, ncols=None):
    assert mat.ndim == 2
    ncols = mat.shape[1] if ncols is None else ncols
    mat_row_reduced = mat.copy()
    p = 0
    for j in range(ncols):
        idxs = p + np.nonzero(mat_row_reduced[p:,j])[0]
        if idxs.size == 0:
            continue
        mat_row_reduced[[p,idxs[0]],:] = mat_row_reduced[[idxs[0],p],:]
        idxs = np.nonzero(mat_row_reduced[:,j])[0].tolist()
        idxs.remove(p)
        mat_row_reduced[idxs,:] = mat_row_reduced[idxs,:] ^ mat_row_reduced[p,:]
        p += 1
        if p == mat_row_reduced.shape[0]:
            break
    return mat_row_reduced, p

def get_generator(pc_matrix_):
    assert pc_matrix_.ndim == 2
    pc_matrix = pc_matrix_.copy().astype(bool).transpose()
    pc_matrix_I = np.concatenate((pc_matrix, np.eye(pc_matrix.shape[0], dtype=bool)), axis=-1)
    pc_matrix_I, p = row_reduce(pc_matrix_I, ncols=pc_matrix.shape[1])
    return row_reduce(pc_matrix_I[p:,pc_matrix.shape[1]:])[0]

def get_standard_form(pc_matrix_):
    pc_matrix = pc_matrix_.copy().astype(bool)
    next_col = min(pc_matrix.shape)
    for ii in range(min(pc_matrix.shape)):
        while True:
            rows_ones = ii + np.where(pc_matrix[ii:, ii])[0]
            if len(rows_ones) == 0:
                new_shift = np.arange(ii, min(pc_matrix.shape) - 1).tolist()+[min(pc_matrix.shape) - 1,next_col]
                old_shift = np.arange(ii + 1, min(pc_matrix.shape)).tolist()+[next_col, ii]
                pc_matrix[:, new_shift] = pc_matrix[:, old_shift]
                next_col += 1
            else:
                break
        pc_matrix[[ii, rows_ones[0]], :] = pc_matrix[[rows_ones[0], ii], :]
        other_rows = pc_matrix[:, ii].copy()
        other_rows[ii] = False
        pc_matrix[other_rows] = pc_matrix[other_rows] ^ pc_matrix[ii]
    return pc_matrix.astype(int)
#############################################

def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

#############################################
def Get_Generator_and_Parity(code, standard_form = False):
    n, k = code.n, code.k
    path_pc_mat = os.path.join(CODES_PATH, f'{code.code_type}_N{str(n)}_K{str(k)}')
    if code.code_type in ['POLAR', 'BCH']:
        ParityMatrix = np.loadtxt(path_pc_mat+'.txt')
    elif code.code_type in ['CCSDS', 'LDPC', 'MACKAY']:
        ParityMatrix = Read_pc_matrixrix_alist(path_pc_mat+'.alist')
    else:
        raise Exception(f'Wrong code {code.code_type}')
    if standard_form and code.code_type not in ['CCSDS', 'LDPC', 'MACKAY']:
        ParityMatrix = get_standard_form(ParityMatrix).astype(int)
        GeneratorMatrix = np.concatenate([np.mod(-ParityMatrix[:, min(ParityMatrix.shape):].transpose(),2),np.eye(k)],1).astype(int)
    else:
        GeneratorMatrix = get_generator(ParityMatrix)
    assert np.all(np.mod((np.matmul(GeneratorMatrix, ParityMatrix.transpose())), 2) == 0) and np.sum(GeneratorMatrix) > 0
    return GeneratorMatrix.astype(float), ParityMatrix.astype(float)



##################################################################
##################################################################

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

##################################################################


class ECC_Dataset(data.Dataset):
    def __init__(self, code, sigma, len, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

        self.zero_word = torch.zeros((self.code.k)).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.n)).long() if zero_cw else None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        z = torch.randn(self.code.n) * random.choice(self.sigma)
        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        return m.float(), x.float(), z.float(), y.float(), magnitude.float(), syndrome.float()


##################################################################
##################################################################

def train(model, device, train_loader, optimizer, epoch, LR, config: Config):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = cum_loss = 0.
    t = time.time()
    batch_idx = 0
    for m, x, z, y, magnitude, syndrome in tqdm(train_loader, position=0, leave=True, desc="Training"):
        z_mul = (y * bin_to_sign(x))
        z_pred = model(magnitude.to(device), syndrome.to(device))
        loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device))
        fer = FER(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]
        if batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
        batch_idx += 1
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


def test(model, device, test_loader_list, EbNo_range_test, min_FER=100):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.
            while True:
                (m, x, z, y, magnitude, syndrome) = next(iter(test_loader))
                z_mul = (y * bin_to_sign(x))
                z_pred = model(magnitude.to(device), syndrome.to(device))
                loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))

                test_loss += loss.item() * x.shape[0]

                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
                if (min_FER > 0 and test_fer > min_FER and cum_count > 1e5) or cum_count >= 1e9:
                    if cum_count >= 1e9:
                        logging.info(f'Number of samples threshold reached for EbN0:{EbNo_range_test[ii]}')
                    else:
                        logging.info(f'FER count threshold reached for EbN0:{EbNo_range_test[ii]}')
                    break
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list[-1]:.2e}')
        ###
        logging.info('\nTest Loss ' + ' '.join(
            ['{}: {:.4e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_list, EbNo_range_test))]))
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.4e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_fer_list, EbNo_range_test))]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.4e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
        logging.info('Test -ln(BER) ' + ' '.join(
            ['{}: {:.4e}'.format(ebno, -np.log(elem)) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))

    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_list, test_loss_ber_list, test_loss_fer_list
