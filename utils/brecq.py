from utils.ptq_utils import *


def track_grad(self, grad_input, grad_output):
    grad = grad_output[0].detach().clone().cpu()
    if hasattr(self, "stored_grad"):
        self.stored_grad.append(grad.abs())
    else:
        self.stored_grad = [grad.abs()]

def track_input(self, inp, out):
    if hasattr(self, "stored_inp"):
        [inp_old.append(inp_new.detach().clone().cpu()) for inp_old, inp_new in zip(self.stored_inp, inp)] 
    else:
        self.stored_inp = [[i.detach().clone().cpu()] for i in inp]
        
def track_output(self, inp, out):
    if hasattr(self, "stored_out"):
        self.stored_out.append(out.detach().clone().cpu()) 
    else:
        self.stored_out = [out.detach().clone().cpu()]
        
def observe_input(m):
    handle = m.register_forward_hook(track_input)
    m.handle_input = handle
    return m    

def observe_output(m):
    handle = m.register_forward_hook(track_output)
    m.handle_output = handle
    return m    

def observe_grad(m):
    handle = m.register_backward_hook(track_grad)
    m.handle_grad = handle
    return m

def reset_hooks(mod):
    
    if hasattr(mod, "handle_input"):
        mod.handle_input.remove()
        del mod.handle_input
        
    if hasattr(mod, "handle_output"):
        mod.handle_output.remove()
        del mod.handle_output
        
    if hasattr(mod, "handle_grad"):
        mod.handle_grad.remove()  
        del mod.handle_grad
        
        
def delete_stored_tensors(mod):     

    if hasattr(mod, "stored_inp"):
        del mod.stored_inp

    if hasattr(mod, "stored_out"):
        del mod.stored_out
        
    if hasattr(mod, "stored_grad"):
        del mod.stored_grad
       
        
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def attach_observers_prog(generator, obs=observe_input):
    for i in range(6):
        generator.generator.progression[i] = obs(generator.generator.progression[i])

    generator.generator.to_rgb[5] = obs(generator.generator.to_rgb[5])
    return generator

def del_observers_prog(generator):
    for i in range(6):
        generator.generator.progression[i] = generator.generator.progression[i].m

    generator.generator.to_rgb[5] = generator.generator.to_rgb[5].m
    return generator

def attach_observers_style(generator, obs=observe_input):
    style = []
    style.append(obs(nn.Sequential(generator.style[0], generator.style[1],
                                   generator.style[2], generator.style[3])))
    for i in range(4, len(generator.style), 2):
        style.append(obs(nn.Sequential(generator.style[i], generator.style[i+1])))
    style = nn.Sequential(*style)
    generator.style = style
    return generator.style


def get_optimizers_and_schedulers(block, lr, lr_lsq, lq, iters, weight_mode):
    
    block.apply(enable_sigmoid_quant)
    parameters = []
    parameters_quant = []
    for name, para in block.named_parameters():
        if ("scale" in name) or ("zero_point" in name):
            parameters_quant.append(para) #LSQ params
        elif weight_mode == "STE":
            parameters.append(para) 
        elif weight_mode == "adaround":
            if ("weight_fake_quant.alpha" in name) or ("bias" in name):
                parameters.append(para) # adaround and biases
            else:
                para.requires_grad = False # no gradient to weights
        else:
            raise NotImplementedError 

    block.apply(torch.quantization.enable_observer)
    if lq:
        block.apply(enable_param_learning) 
    else:
        block.apply(enable_static_estimate)
        block.apply(enable_nearest_quant)

    print("Number of quant params", sum([i.numel() for i in parameters_quant]))
    print("Number of params", sum([i.numel() for i in parameters]))

    # Schedulers and optimizers
    optimizer_lq = torch.optim.Adam(parameters_quant, lr=lr_lsq) 
    lr_scheduler_lq = torch.optim.lr_scheduler.StepLR(optimizer_lq, int(iters/2.))     

    optimizer_p = torch.optim.Adam(parameters, lr=lr)
    lr_scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, int(iters/2.))
    
    return optimizer_p, lr_scheduler_p, optimizer_lq, lr_scheduler_lq
          
    
class SetupStoredTensors:
    
    def __init__(self, full_reference, full_quant, X, noise, batch_size=16, weighted=False):
        self.full_reference = full_reference
        self.full_quant = full_quant
        self.X = X
        self.noise = noise
        self.batch_size = batch_size
        self.weighted = weighted
        
    def __call__(self):
        self.full_reference.apply(delete_stored_tensors)
        self.full_quant.apply(delete_stored_tensors)

        self.full_reference.cuda() 
        self.full_quant.cuda()

        for i in tqdm(range(int(len(self.X)/self.batch_size)), desc="Setting up stored tensors"):
            inp = self.X[self.batch_size*i:self.batch_size*(i+1)].cuda()
            inp_noise = [n[self.batch_size*i:self.batch_size*(i+1)].cuda() for n in self.noise]

            with torch.no_grad():
                out_ref = self.full_reference(inp, noise=inp_noise)
            
            if self.weighted:
                out_q = self.full_quant(inp, noise=inp_noise)  
            else:
                with torch.no_grad():
                    out_q = self.full_quant(inp, noise=inp_noise)
            
            if self.weighted:
                loss = F.mse_loss(out_q, out_ref.detach().clone())
                loss.backward()
            del out_q
            del out_ref

        self.full_reference.cpu()
        self.full_quant.cpu()
        for p in self.full_quant.parameters():
            del p.grad
            p.grad = None
        for p in self.full_reference.parameters():
            del p.grad
            p.grad = None

def weighted_mse(y, y_hat, weights):
    return ((y - y_hat).pow(2) * weights).mean()  


def get_block_data(quant_block, ref_block, train_size, weighted, quantized_inp):
    
    if quantized_inp:
        stored_input = [torch.cat(x, axis=0).detach() for x in quant_block.stored_inp]
        del quant_block.stored_inp
    else:
        stored_input = [torch.cat(x, axis=0).detach() for x in ref_block.stored_inp]
        del ref_block.stored_inp
        
    stored_output = torch.cat(ref_block.stored_out, axis=0).detach()
    del ref_block.stored_out
    
    inp_train = [x[:train_size] for x in stored_input]
    out_train = stored_output[:train_size]
    inp_val = [x[train_size:] for x in stored_input]
    out_val = stored_output[train_size:]
    
    if weighted:
        stored_grad = torch.cat(quant_block.stored_grad, axis=0).detach()
        del quant_block.stored_grad
        
        weights_train = stored_grad[:train_size].pow(2)
        normalizer_train = weights_train.mean()
        weights_train /= normalizer_train
        
        weights_val = stored_grad[train_size:].pow(2)
       # normalizer_val = weights_val.mean([d for d in range(1, len(stored_grad.shape))], True)
        weights_val /= normalizer_train
        
    else:
        weights_train = None
        weights_val = None   
        
    return inp_train, inp_val, out_train, out_val, weights_train, weights_val
      
    
def get_batch(inp, out, batch_size, weights=None):
    permutation = torch.randperm(len(out))
    indices = permutation[:batch_size]
    X_b = [x[indices].cuda().clone() for x in inp]
    Y_b = out[indices].cuda().clone()
    if torch.is_tensor(weights):
        W_b = weights[indices].cuda().clone()
    else:
        W_b = 1
    return X_b, Y_b, W_b


def optimize_sequentially(reference, quant, data_size, train_size=1024, weighted=True, device="cuda:0",
                          batch_size=16, num_blocks=None, lr=1e-4, lr_lsq=1e-4, quantized_inp=False,
                          iters=2000, lq=True, plot=True, setup_tensors=None, weight_mode="STE"):
    
    val_size = data_size - train_size
    
    reference.cpu().train()
    quant.cpu().train()
    
    history_train = []
    history_val = []
    
    for block_id in range(num_blocks if num_blocks else len(quant)):

        losses = AverageMeter("loss")
        losses_val = AverageMeter("loss")
        loss_train = []
        loss_val = []
        optimizer_p, lr_scheduler_p, optimizer_lq, lr_scheduler_lq = get_optimizers_and_schedulers(quant[block_id], lr,
                                                                                                   lr_lsq, lq, iters,
                                                                                                   weight_mode)
        print("Layer", block_id)
        if weighted:
            observe_grad(quant[block_id])
        if quantized_inp:
            observe_input(quant[block_id])
        else:
            observe_input(reference[block_id])
        observe_output(reference[block_id])
        setup_tensors()
        #Setup train, val and weights
        inp_train, inp_val, out_train, out_val, weights_train, weights_val = get_block_data(quant[block_id],
                                                                                            reference[block_id],
                                                                                            train_size,
                                                                                            weighted,
                                                                                            quantized_inp)
            
        quant[block_id].apply(reset_hooks)
        reference[block_id].apply(reset_hooks)   

        quant[block_id].cuda()
        
        criterion = LossFunction(quant[block_id], weight_mode, max_count=iters)
        
        for j in range(iters):  
            #validation
            with torch.no_grad():
                if j % 10 == 0:
                    X_b, Y_b, W_b = get_batch(inp_val, out_val, batch_size, weights_val)
                    loss = weighted_mse(quant[block_id](*X_b), Y_b, W_b)
                    losses_val.update(loss.detach().data.cpu().item())            
            
            #Sample batch of data
            X_b, Y_b, W_b = get_batch(inp_train, out_train, batch_size, weights_train)
            optimizer_lq.zero_grad()
            optimizer_p.zero_grad()  
            loss = criterion(quant[block_id](*X_b), Y_b, W_b)
            loss.backward()
            optimizer_lq.step()
            optimizer_p.step()
            lr_scheduler_lq.step()
            lr_scheduler_p.step()
            losses.update(loss.detach().data.cpu().item())

            if (j % 100 == 0):
                #print(f"Iter{j} Avg train loss:", losses)
                loss_train.append(losses.avg)
                print("Val MSE:", losses_val)
                loss_val.append(losses_val.avg)
        
        history_train.append(loss_train)
        history_val.append(loss_val)
        
        if plot:
           # clear_output()
            plt.figure(figsize=(8, 4 * len(history_train)))
            for i in range(len(history_train)):
                plt.subplot(len(history_train), 1, i + 1)
                plt.plot(history_train[i], label="train")
                plt.plot(history_val[i], label="val")
                plt.title(f"Layer {i}")
                plt.legend()
            plt.show()
        
        del X_b, Y_b, W_b
        del optimizer_lq, optimizer_p, lr_scheduler_p, lr_scheduler_lq
        for p in quant[block_id].parameters():
            del p.grad
            p.grad = None
        quant[block_id] = quant[block_id].cpu()
        del inp_train, inp_val, out_train, out_val, weights_train, weights_val
        
            
        
        
def get_lbl_quant_model_stylegan(model_path, save_path="./q.pth", q_params=None, device="cuda:0",
                                 weighted=False, disable_observer=True, quantized_inp=False,
                                 plot=True, lq=True, lr=1e-4, lr_lsq=1e-4, iters=2000):
    
    weight_mode = "adaround" if q_params.get("adaround", False) else "STE"
    
    full_ref = StyledGenerator(512)
    full_ref.load_state_dict(torch.load(model_path, map_location="cpu"))
    full_ref.to(device)
    transform_style(full_ref)

    full_quant = get_static_quant_model(model_path, q_params, 500).cpu()
    transform_style(full_quant)

    data_size = 896
    batch_size = 16
    train_size = 864

    X = torch.randn(data_size, 512)
    noise = []

    for i in range(6):
        size = 4 * 2 ** i
        noise.append(torch.randn(data_size, 1, size, size))

    setuper = SetupStoredTensors(full_ref, full_quant, X, noise, batch_size, weighted)

    #optimize style network #full_quant.style # full_ref.style
    blocks_quant = nn.Sequential(*([full_quant.style[i] for i in range(len(full_quant.style))] + \
                                [full_quant.generator.progression[i] for i in range(0, 6)] + \
                                [full_quant.generator.to_rgb[5]]))
    
    blocks_ref = nn.Sequential(*([full_ref.style[i] for i in range(len(full_ref.style))] + \
                                [full_ref.generator.progression[i] for i in range(0, 6)] + \
                                [full_ref.generator.to_rgb[5]]))
    
    optimize_sequentially(blocks_ref, blocks_quant, setup_tensors=setuper, quantized_inp=quantized_inp,
                          data_size=data_size, weighted=weighted, plot=plot, lq=lq, lr=lr, lr_lsq=lr_lsq,
                          train_size=train_size, batch_size=batch_size, iters=iters, weight_mode=weight_mode)    
    
    
    torch.save(full_quant.state_dict(), save_path)
    
    if disable_observer:
        full_quant.apply(torch.quantization.disable_observer)
        full_quant.apply(enable_fixed_estimate)
        full_quant.apply(enable_hard_sigmoid)
    return full_quant
