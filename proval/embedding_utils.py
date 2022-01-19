import pickle, torch, biovec, esm, os
from CPCProt.tokenizer import Tokenizer
from CPCProt import CPCProtModel, CPCProtEmbedding
import numpy as np
from utils import is_symmetric
from sw_alignment import alignSmithWaterman


def swTrainTest(train, test, sw_alignment):
    if sw_alignment == "full":
        # Option 1: full alignment and the test / train split
        if not os.path.isfile("data/sw_alignment_all.pkl"):
            alignSmithWaterman(sw_alignment)
        sw_alignment_all = pickle.load(open("data/sw_alignment_all.pkl",
                                            "rb")).to_numpy(dtype=np.float)
        sw_alignment_all = 0.5 * (sw_alignment_all + sw_alignment_all.T)
        sw_alignment_train = sw_alignment_all[:len(train), :len(train)]
        sw_alignment_test = sw_alignment_all[len(train):, :len(train)]
    elif sw_alignment == "traintest":
        if not os.path.isfile(
                "data/sw_alignment_train.pkl") and not os.path.isfile(
                    "data/sw_alignment_test.pkl"):
            alignSmithWaterman(sw_alignment)
        # Option 2 (for completeness): alignment of all train sequences to all train sequences and all test sequences to all train sequences, respectively
        sw_alignment_train = pickle.load(
            open("data/sw_alignment_train.pkl", "rb"))
        assert sw_alignment_train.isnull().values.any() == False
        assert list(sw_alignment_train) == [_.id for _ in train]
        assert list(sw_alignment_train.index) == [_.id for _ in train]
        sw_alignment_train = sw_alignment_train.to_numpy(dtype=np.float)
        sw_alignment_train = 0.5 * (sw_alignment_train + sw_alignment_train.T)

        sw_alignment_test = pickle.load(
            open("data/sw_alignment_test.pkl", "rb"))
        assert sw_alignment_test.isnull().values.any() == False
        assert list(sw_alignment_test) == [_.id for _ in train]
        assert list(sw_alignment_test.index) == [_.id for _ in test]
        sw_alignment_test = sw_alignment_test.to_numpy(dtype=np.float)
    return sw_alignment_train, sw_alignment_test


def swDissimRep(X_Train, X_Test, train_fasta, test_fasta, d):
    (n, m) = X_Train.shape
    if n != m:
        raise Exception('The similarity matrix must be square.')
    if not is_symmetric(X_Train):
        raise Exception('The similarity matrix must be symmetric.')
    a, c = np.linalg.eigh(X_Train)
    embeddedEigenvalues = np.diag(np.lib.scimath.sqrt(np.abs(a[-d:])))
    embedded_X_Train = (embeddedEigenvalues @ c[-d:, :] @ X_Train.T).T
    embedded_X_Test = (embeddedEigenvalues @ c[-d:, :] @ X_Test.T).T
    embedded_X_Train_dict = {
        str(s.id): vec
        for s, vec in zip(train_fasta, embedded_X_Train)
    }
    embedded_X_Test_dict = {
        str(s.id): vec
        for s, vec in zip(test_fasta, embedded_X_Test)
    }
    return embedded_X_Train_dict, embedded_X_Test_dict


def swDissimRepComplex(X_Train, X_Test, train_fasta, test_fasta, d):
    n = X_Train.shape[0]
    r = np.random.choice(n, d)
    r.sort()
    X_Train_nm = X_Train[:, r]
    X_Train_mm = X_Train[np.ix_(r, r)]
    X_Test_nm = X_Test[:, r]

    w = np.linalg.pinv(X_Train_mm, hermitian=True)
    w = 0.5 * (w + w.T)
    a, c = np.linalg.eigh(w)
    embeddedEigenvalues = np.diag(np.lib.scimath.sqrt(a))
    embedded_X_Train = (embeddedEigenvalues @ c.T @ X_Train_nm.T).T
    embedded_X_Test = (embeddedEigenvalues @ c.T @ X_Test_nm.T).T
    embedded_X_Train_dict = {
        str(s.id): vec
        for s, vec in zip(train_fasta, embedded_X_Train)
    }
    embedded_X_Test_dict = {
        str(s.id): vec
        for s, vec in zip(test_fasta, embedded_X_Test)
    }
    return embedded_X_Train_dict, embedded_X_Test_dict


def protvec(train, test):
    pv = biovec.models.load_protvec(
        'pretrained_models/swissprot-reviewed-protvec.model')
    embedded_X_Train, embedded_X_Test, y_Train, y_Test = {}, {}, {}, {}
    for s in train:
        try:  # necessary to skip sequences, where ngrams are missing
            embedded_X_Train[str(s.id)] = sum(pv.to_vecs(str(s.seq)))
            y_Train[str(s.id)] = float(s.description)
        except Exception:
            pass
    for s in test:
        try:  # necessary to skip sequences, where ngrams are missing
            embedded_X_Test[str(s.id)] = sum(pv.to_vecs(str(s.seq)))
            y_Test[str(s.id)] = float(s.description)
        except Exception:
            pass

    return embedded_X_Train, y_Train, embedded_X_Test, y_Test


def esm1b(train, test):
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    embedded_X_Train, embedded_X_Test, y_Train, y_Test = {}, {}, {}, {}

    for s in train:
        if len(s.seq) <= 1022:
            _, _, sample_batch_tokens = batch_converter([(s.id, s.seq)])

            with torch.no_grad():
                results = model(sample_batch_tokens.cuda(),
                                repr_layers=[33],
                                return_contacts=True)
            token_representations = results["representations"][
                33]  #.detach().cpu()

            seq_rep = token_representations[0, 1:len(s.seq) + 1].mean(0)

            embedded_X_Train[str(s.id)] = np.squeeze(seq_rep.detach().cpu())
            y_Train[str(s.id)] = float(s.description)

    for s in test:
        if len(s.seq) <= 1022:
            _, _, sample_batch_tokens = batch_converter([(s.id, s.seq)])

            with torch.no_grad():
                results = model(sample_batch_tokens.cuda(),
                                repr_layers=[33],
                                return_contacts=True)
            token_representations = results["representations"][
                33]  #.detach().cpu()

            seq_rep = token_representations[0, 1:len(s.seq) + 1].mean(0)

            embedded_X_Test[str(s.id)] = np.squeeze(seq_rep.detach().cpu())
            y_Test[str(s.id)] = float(s.description)

    return embedded_X_Train, y_Train, embedded_X_Test, y_Test


def cpcprot(train, test, vec_type):
    ckpt_path = "pretrained_models/cpcprot_best.ckpt"  # Replace with actual path to CPCProt weights
    model = CPCProtModel()
    model.load_state_dict(torch.load(ckpt_path))
    embedder = CPCProtEmbedding(model)
    tokenizer = Tokenizer()

    embedded_X_Train, embedded_X_Test = {}, {}

    def pad_sequence(seq, token=0):
        s = list(seq)
        while len(s) < 11:
            s.append(token)
        return np.array(s)

    for s in train:
        tokens = tokenizer.encode(s.seq)
        if len(tokens) < 11:
            tokens = pad_sequence(tokens)
        input = torch.tensor([tokens])

        if vec_type == 'zmean':
            vec = embedder.get_z_mean(input)  # (1, 512)
        elif vec_type == 'cmean':
            vec = embedder.get_c_mean(input)  # (1, 512)
        elif vec_type == 'cfinal':
            vec = embedder.get_c_final(input)  # (1, 512)
        else:
            raise ValueError

        embedded_X_Train[str(s.id)] = np.squeeze(vec.detach().cpu())

    for s in test:
        tokens = tokenizer.encode(s.seq)
        if len(tokens) < 11:
            tokens = pad_sequence(tokens)
        input = torch.tensor([tokens])

        if vec_type == 'zmean':
            vec = embedder.get_z_mean(input)  # (1, 512)
        elif vec_type == 'cmean':
            vec = embedder.get_c_mean(input)  # (1, 512)
        elif vec_type == 'cfinal':
            vec = embedder.get_c_final(input)  # (1, 512)
        else:
            raise ValueError

        embedded_X_Test[str(s.id)] = np.squeeze(vec.detach().cpu())

    return embedded_X_Train, embedded_X_Test
