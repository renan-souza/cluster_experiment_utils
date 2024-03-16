# The code in this file is based on:
# https://blog.paperspace.com/build-a-language-model-using-pytorch/
import math
import os
import sys
from uuid import uuid4
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
from time import sleep
import json


# Define a function to batchify the data
def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i, bptt=35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def get_wiki_text(data_dir):
    # Load the WikiText2 dataset
    t0 = time()
    train_data = torch.load(os.path.join(data_dir, "train_data.tensor"))
    val_data = torch.load(os.path.join(data_dir, "val_data.tensor"))
    test_data = torch.load(os.path.join(data_dir, "test_data.tensor"))
    with open(os.path.join(data_dir, "ntokens.txt")) as f:
        ntokens = int(f.read())
    t1 = time()
    t_disk_load = t1 - t0

    try:
        device_type = "cpu"
        try:
            if torch.cuda.is_available():
                device_type = "cuda"
            elif torch.backends.mps.is_available():
                device_type = "mps"
        except:
            pass
        device = torch.device(device_type)
        t2 = time()
        t_device_available = t2 - t1        
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)
        t_gpu_load = time() - t2    
    except:
        raise Exception("Couldn't send data to device")
    
    
    if device_type == "cpu":
        raise Exception(
            "It is going to take forever because we coudln't send the data to a faster device"
        )

    print("Train data", train_data.shape)
    print("Validation data", val_data.shape)
    print("Test data", test_data.shape)
    print("ntokens", ntokens)
    return ntokens, train_data, val_data, test_data, t_disk_load, t_device_available, t_device_available, t_gpu_load


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken,
        d_model,
        nhead,
        d_hid,
        nlayers,
        dropout=0.5,
        pos_encoding_max_len=5000,
        parent_workflow_id=None,
    ):
        super(TransformerModel, self).__init__()
        
        TransformerEncoderLayer = nn.TransformerEncoderLayer
        TransformerEncoder = nn.TransformerEncoder
        Embedding = nn.Embedding
        Linear = nn.Linear

        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(
            d_model,
            dropout,
            max_len=pos_encoding_max_len,
            #workflow_id=self.workflow_id,
        )
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = Linear(d_model, ntoken)

    ##Generate a mask for the input sequence
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        ## Change all the zeros to negative infinity and all the ones to zeros as follows:
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    #@flowcept_task(args_handler=torch_args_handler)
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


# Define the PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, workflow_id=None):
        super(PositionalEncoding, self).__init__()
        #self.workflow_id = workflow_id
        # Dropout = register_modules(
        #     [
        #         nn.Dropout,
        #     ],
        #     workflow_id=self.workflow_id,
        # )
        Dropout = nn.Dropout
        
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    #@flowcept_task(args_handler=torch_args_handler)
    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def train_epoch(ntokens, model, train_data, criterion, optimizer, bptt=35):
    model.train()  # Set the model to training mode
    total_loss = 0.0  # Initialize the total loss to 0

    # Iterate through the mini-batches of data
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(
            train_data, i, bptt
        )  # Get the input data and targets for the current mini-batch
        optimizer.zero_grad()  # Reset the gradients to zero before the next backward pass
        output = model(
            data
        )  # Forward pass: compute the output of the model given the input data

        loss = criterion(
            output.view(-1, ntokens), targets
        )  # Calculate the loss between the model output and the targets
        loss.backward()  # Backward pass: compute the gradients of the loss with respect to the model parameters
        optimizer.step()  # Update the model parameters using the computed gradients
        total_loss += loss.item()  # Accumulate the total loss

    return total_loss / (batch + 1)  # Return the average loss per mini-batch


def evaluate(ntokens, model, data_source, criterion, bptt=35):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0  # Initialize the total loss to 0

    # Use torch.no_grad() to disable gradient calculation during evaluation
    with torch.no_grad():
        # Iterate through the mini-batches of data
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(
                data_source, i, bptt
            )  # Get the input data and targets for the current mini-batch
            output = model(
                data
            )  # Forward pass: compute the output of the model given the input data
            loss = criterion(
                output.view(-1, ntokens), targets
            )  # Calculate the loss between the model output and the targets
            total_loss += loss.item()  # Accumulate the total loss

    return total_loss / (i + 1)  # Return the average loss per mini-batch


#@model_profiler()
def model_train(
    batch_size,
    eval_batch_size,
    epochs,
    emsize,
    nhead,
    nhid,
    nlayers,
    dropout,
    lr,
    pos_encoding_max_len,
    rep_dir,
    input_data_dir,
    workflow_id=None,
):
    parameters_dict = {
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": epochs,
        "emsize": emsize,
        "nhead": nhead,
        "nhid": nhid,
        "nlayers": nlayers,
        "dropout": dropout,
        "lr": lr,
        "pos_encoding_max_len": pos_encoding_max_len,
        "rep_dir": rep_dir,
        "input_data_dir": input_data_dir,
    }
    t0 = time()
    init_time = t0
    print("Getting wikitext data...")    
    ntokens, train_data, val_data, test_data, t_disk_load, t_device_available, t_device_available, t_gpu_load = get_wiki_text(input_data_dir)
    print("Done. Now starting the workflow.")
    time_get_wiki_text = time() - t0
    # TODO :ml-refactor: save device type and random seed: https://pytorch.org/docs/stable/notes/randomness.html
    # TODO :base-interceptor-refactor: Can we do it better?
    
    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    device_type = "cpu"
    try:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
    except:
        pass

    if device_type == "cpu":
        raise Exception(
            "It is going to take forever because we coudln't send the data to a faster device"
        )
        
    device = torch.device(device_type)

    model = TransformerModel(
        ntokens,
        emsize,
        nhead,
        nhid,
        nlayers,
        dropout,
        pos_encoding_max_len,
        parent_workflow_id=workflow_id,
    ).to(device)
    
    transformer_model_wf_id = str(uuid4())
    # flowceptor = FlowceptConsumerAPI(flowcept.instrumentation.decorators.instrumentation_interceptor, bundle_exec_id=workflow_id, start_doc_inserter=False)
    # flowceptor.start()
    # flowceptor.logger.info(f"Flowcept with bundle id {id(flowceptor)} started!")
    

    
    # model_wf_ids_dir = os.path.join(rep_dir, "model_wf_ids")
    # os.makedirs(model_wf_ids_dir, exist_ok=True)
    # model_wf_id_path = os.path.join(model_wf_ids_dir, "model_wf_id" + transformer_model_wf_id + ".txt")
    # with open(model_wf_id_path, "w+") as f:
    #     f.write(transformer_model_wf_id)

    best_models_dir =  os.path.join(rep_dir, "best_models")
    os.makedirs(best_models_dir, exist_ok=True)
    best_model_path = os.path.join(
        best_models_dir, f"best_transformer_wikitext2_model_{transformer_model_wf_id}.pth"
    )        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")  # Initialize the best validation loss to infinity
    # best_m = None
    # Iterate through the epochs
    t0 = time()
    for epoch in range(1, epochs + 1):
        print(f"Starting training for epoch {epoch}/{epochs}")
        # Train the model on the training data and calculate the training loss
        train_loss = train_epoch(
            ntokens, model, train_data, criterion, optimizer, batch_size
        )

        # Evaluate the model on the validation data and calculate the validation loss
        val_loss = evaluate(ntokens, model, val_data, criterion, eval_batch_size)

        # Print the training and validation losses for the current epoch
        print(
            f"Epoch: {epoch}, Train loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}"
        )

        # If the validation loss has improved, save the model's state
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # best_m = model
            torch.save(model.state_dict(), best_model_path)
        
        #print(f"Model {transformer_model_wf_id} finished epoch {epoch}")
    training_time = time() - t0
    #print(f"Model {transformer_model_wf_id} finished training.")
    # Load the best model's state
    best_m = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(
        device
    )
    t0 = time()
    print("Loading model")
    torch_loaded = torch.load(best_model_path)
    best_m.load_state_dict(torch_loaded)
    t_load_best_model = time() - t0
    print("Evaluating")
    # Evaluate the best model on the test dataset
    t0 = time()
    test_loss = evaluate(ntokens, best_m, test_data, criterion, eval_batch_size)
    print(f"Test loss: {test_loss:.2f}")
    end_time = time()

    t_eval = end_time - t0
    task_output = {
        "time_total": end_time - init_time,
        "time_device_available": t_device_available,
        "time_disk_load": t_disk_load,
        "time_gpu_load": t_gpu_load,
        "time_get_wiki_text": time_get_wiki_text,
        "time_training": training_time,
        "time_eval": t_eval,
        "time_load_best_model": t_load_best_model,
        "device_type": device_type,
        "test_loss": test_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        #"model": model,
    }
    task_output.update(parameters_dict)
    with open(os.path.join(rep_dir, f"FINISHED_{transformer_model_wf_id}_model_result.json"), "w") as outfile:
        json.dump(task_output, outfile)

    #task_output["model"] = model  # for model_profiler

    # flowceptor.logger.info(f"All done for this model. Closing inner flowcept bundle {id(flowceptor)}")
    # flowceptor.stop()
    return task_output
