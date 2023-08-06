'''
We only put some key components in demo.py, so it can not run directly. We contain train()
for training models, ML_Dataset() for loading data, set_mediator() for pre-training, and sync() for
synchronization.  AuxiliaryDeepGenerativeModel() is the generative model we use in this work.
'''

# some hyper-parameters
BATCH_SIZE = 50
LABELED_RATIO = 0.05
image_size = 28*28
n_labels = 11

DATASET, LEARNING_RATE, WEIGHT_DECAY = HyperParams[MODEL]
VAE_LEARNING_RATE = 0.001
INIT_LEARNING_RATE = VAE_LEARNING_RATE
MAX_ROUND = 3000
MAX_PRETRAIN_ROUND = 20





#train model, including VAE model and model
def train():
    labelled, unlabelled, validation = datasource.ML_Dataset(DATASET, WORLD_SIZE, RANK, BATCH_SIZE, LABELED_RATIO, D_ALPHA, IS_INDEPENDENT).get_dataloaders() #Load dataset
    model = dgm.AuxiliaryDeepGenerativeModel([28*28, n_labels, 100, 100, [500, 500]])
    vae_model = vae_pretrain.VAE_Mnist()
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=VAE_LEARNING_RATE)
    beta = DeterministicWarmup(n=4*len(unlabelled)*100)
    sampler = ImportanceWeightedSampler(mc=1, iw=1)
    elbo = SVI(model, likelihood=binary_cross_entropy, beta=beta, sampler=sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
    loss_func = torch.nn.CrossEntropyLoss() #loss function (supervised learning)

    logging('initial model parameters: ')
    logging('\n\n ----- start training -----')

    as_manager = AS_Manager(vae_model, model, MASTER_ADDRESS, WORLD_SIZE, RANK, INIT_SYNC_FREQ)


    iter_id = 0
    epoch_id = 0
    pretrain_epoch_id = 0
    pretrain_iter_id = 0
    best = 0.0
    alpha = 0.1 * len(unlabelled) / len(labelled)

    client_mu_mu = torch.zeros(30)
    client_sigma_mu = torch.zeros(30)
    client_mu_sigma = torch.zeros(30)
    client_sigma_sigma = torch.zeros(30)

    logging('\n\n ----- start pre-training -----')

    # Add before FL train
    while pretrain_epoch_id < MAX_PRETRAIN_ROUND:
        logging('\n\n--- start pretrain epoch '+ str(pretrain_epoch_id) + ' ---')

        for step, (b_x, b_y) in enumerate(unlabelled):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            #print('b_y: ' + str(b_y))
            b_x = b_x.view(-1, image_size)
            x_reconst, mu, log_var = vae_model(b_x) # Unsupervised learning

            # Compute reconstruction loss and kl divergence
            reconst_loss = F.mse_loss(x_reconst, b_x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Backprop and optimize
            loss = reconst_loss + kl_div
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()


        logging('pretrain Reconst Loss:' + str(reconst_loss.item()) + '; pretrain epoch id: ' + str(pretrain_epoch_id))
        as_manager.last_test_pretrain_epoch_id = pretrain_epoch_id

        pretrain_epoch_id += 1
        as_manager.pretrain_epoch_id = pretrain_epoch_id

    logging("local pre-train finishes.")

    logging("start calculate mu and sigma.")

    # Calculate mu for each client
    for step, (b_x, b_y) in enumerate(unlabelled):
        b_x = b_x.cuda()
        b_x = b_x.view(-1, image_size)
        with torch.no_grad():
            x_reconst, mu, log_var = vae_model(b_x) # Unsupervised learning
        log_sigma = 0.5 * log_var
        #print('pretrain iter id: ' + str(pretrain_iter_id))

        client_mu_mu += torch.sum(mu,dim=0).cpu() # Calculate client mu
        #print('client mu mu: ' + str(client_mu_mu))
        client_sigma_mu += torch.sum(log_sigma,dim=0).cpu()

        pretrain_iter_id += 1

    num_of_samples = pretrain_iter_id * BATCH_SIZE
    client_mu_mu = client_mu_mu / num_of_samples
    client_sigma_mu = client_sigma_mu / num_of_samples

    # Calculate sigma for each client
    for step, (b_x, b_y) in enumerate(unlabelled):
        b_x = b_x.cuda()
        b_x = b_x.view(-1, image_size)
        with torch.no_grad():
            x_reconst, mu, log_var = vae_model(b_x) # Unsupervised learning
        log_sigma = 0.5 * log_var

        for i in range(BATCH_SIZE):
            mu_mu_temp = mu[i].cpu()
            sigma_mu_temp = log_sigma[i].cpu()
            client_mu_sigma += (mu_mu_temp - client_mu_mu).pow(2)
            client_sigma_sigma += (sigma_mu_temp - client_sigma_mu).pow(2)

    client_mu_sigma = client_mu_sigma / (num_of_samples - 1) # Calculate client sigma
    client_sigma_sigma = client_sigma_sigma / (num_of_samples - 1)
    num_of_samples = torch.tensor(num_of_samples)
    total_num_samples = as_manager.total_samples(num_of_samples)

    logging("finishing calculating mu and sigma.")

    as_manager.sync_mu_and_sigma(client_mu_mu, client_mu_sigma, client_sigma_mu, client_sigma_sigma)

    logging('mediator num: ' + str(as_manager.mediators_num))


    torch.autograd.set_detect_anomaly(True)
    while epoch_id < MAX_ROUND:
        model.train()
        logging('\n\n--- start epoch '+ str(epoch_id) + ' ---')
        logging('\nrank id: ' + str(as_manager.rank))
        #count = [0]*200
        total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
        #for step, (b_x, b_y) in enumerate(train_loader):
        for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
            #mediator_id, rank_id, mediator_length = as_manager.sync_within_mediator(model, iter_id)

            #if NOT_MEDIATOR == True or as_manager.rank == rank_id[as_manager.current_transferred_client % mediator_length]:
            '''
                #if as_manager.rank == 0 or as_manager.rank == 1:
                size_original = b_x[int(b_x.size()[0]*0.8):b_x.size()[0]].size() # Supervised learning size
                #size_original = b_x.size()
                b_y = b_y[int(b_y.size()[0]*0.8):b_y.size()[0]] # Only use 10 percent samples for supervised learning
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                b_x = b_x.view(-1, image_size)
                x_reconst, mu, log_var = vae_model(b_x) # Unsupervised learning


                # Compute reconstruction loss and kl divergence
                # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
                reconst_loss = F.mse_loss(x_reconst, b_x, size_average=False)
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                # Backprop and optimize
                loss = reconst_loss + kl_div
                vae_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                vae_optimizer.step()

                x_reconst_supervised = x_reconst[int(x_reconst.size()[0]*0.8):x_reconst.size()[0]].detach() # 10 percent for supervised training
                #ypred = model(x_reconst_supervised.reshape(size_original)) #Supervised learning
                ypred = model(b_x[int(b_x.size()[0]*0.8):b_x.size()[0]].reshape(size_original))
                #ypred = model(b_x.reshape(size_original))
                added_loss = loss_func(ypred, b_y)
                optimizer.zero_grad()
                added_loss.backward()
                optimizer.step()
                #if iter_id % INIT_SYNC_FREQ == 10:
                #    print('iter id: ' + str(iter_id))
            '''
            x, y, u = Variable(x), Variable(y), Variable(u)
            x, y = x.cuda(), y.cuda()
            u = u.cuda()
            #logging('y: ' + str(y))

            L = -elbo(x, y)
            U = -elbo(u)

            # Add auxiliary classification loss q(y|x)
            logits = model.classify(x)
            classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

            J_alpha = L - alpha * classication_loss + U

            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += J_alpha.data
            labelled_loss += L.data
            unlabelled_loss += U.data

            _, pred_idx = torch.max(logits, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((pred_idx.data == lab_idx.data).float())


            iter_id += 1

            #logging('\niter_id: ' + str(iter_id))

                #print('current_transferred_client: ' + str(as_manager.current_transferred_client))


            # sync(): synchronization function
            #if as_manager.sync(model, iter_id):
            if as_manager.sync_within_mediator(model, iter_id):
            #if as_manager.rank == 0 or as_manager.rank == 1:
                '''
                accuracy = test(test_loader, model)
                #print('mokaiwei')
                if epoch_id != as_manager.last_test_epoch_id and epoch_id != 0:
                    logging('\n - test - accuracy:' + str(accuracy) + ';Reconst Loss:' + str(reconst_loss.item()) + '; round_id:' + str(as_manager.round_id) + '; epoch_id:' + str(epoch_id) + '; iter_id:' + str(iter_id) + '; sync_frequency:' + str(as_manager.sync_frequency) + '; time: ' + str((as_manager.round_id + 0.05*iter_id)/3600.0))
                   # print('loss: ' + str(loss))
                as_manager.last_test_epoch_id = epoch_id
                '''
                if as_manager.last_test_epoch_id != epoch_id:
                    model.eval()
                    m = len(unlabelled)
                    logging("Epoch: {}".format(epoch_id))
                    logging('round_id: ' + str(as_manager.round_id))
                    logging("[Train]\t\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.2f}".format(total_loss / m,
                                                                                              labelled_loss / m,
                                                                                              unlabelled_loss / m,
                                                                                              accuracy / m))

                    total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
                    for x, y in validation:
                        x, y = Variable(x), Variable(y)

                        x, y = x.cuda(), y.cuda()

                        L = -elbo(x, y)
                        U = -elbo(x)

                        logits = model.classify(x)
                        classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                        J_alpha = L + alpha * classication_loss + U

                        total_loss += J_alpha.data
                        labelled_loss += L.data
                        unlabelled_loss += U.data

                        _, pred_idx = torch.max(logits, 1)
                        _, lab_idx = torch.max(y, 1)
                        accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

                    m = len(validation)
                    logging("within_mediator_Validation\t J_a: {:.3f}, L: {:.3f}, U: {:.3f}, accuracy: {:.3f}".format(total_loss / m,
                                                                                                  labelled_loss / m,
                                                                                                  unlabelled_loss / m,
                                                                                                  accuracy / m))
                    if as_manager.if_stable == 0:
                        if accuracy / m > best:
                            best = accuracy / m
                            as_manager.last_best_epoch_id = epoch_id
                        else:
                            if epoch_id - as_manager.last_best_epoch_id > as_manager.stable_window_size:
                                as_manager.if_stable = torch.tensor(1)
                    as_manager.last_test_epoch_id = epoch_id

            if as_manager.sync(model, iter_id):
                for (i, p) in enumerate(model.parameters()):
                    dist.broadcast(p.data, as_manager.rank_id[0], group=as_manager.within_mediator_group[as_manager.mediator_id-1])
                if as_manager.last_mediator_test_epoch_id != epoch_id:
                    model.eval()
                    m = len(unlabelled)
                    logging("Epoch: {}".format(epoch_id))
                    logging('mediators_round_id: ' + str(as_manager.mediators_round_id))

                    total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
                    for x, y in validation:
                        x, y = Variable(x), Variable(y)

                        x, y = x.cuda(), y.cuda()

                        L = -elbo(x, y)
                        U = -elbo(x)

                        logits = model.classify(x)
                        classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                        J_alpha = L + alpha * classication_loss + U

                        total_loss += J_alpha.data
                        labelled_loss += L.data
                        unlabelled_loss += U.data

                        _, pred_idx = torch.max(logits, 1)
                        _, lab_idx = torch.max(y, 1)
                        accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

                    m = len(validation)
                    logging("mediators_Validation\t J_a: {:.3f}, L: {:.3f}, U: {:.3f}, accuracy: {:.3f}".format(total_loss / m,
                                                                                                  labelled_loss / m,
                                                                                                  unlabelled_loss / m,
                                                                                                  accuracy / m))
                    as_manager.last_mediator_test_epoch_id = epoch_id


        epoch_id += 1
        as_manager.epoch_id = epoch_id



# load data for each client
class ML_Dataset():
    # Initialization
    def __init__(self, dataset_name, world_size, rank, batch_size, labeled_ratio, d_alpha=100.0, is_independent=True):
        self.dataset_name = dataset_name
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.labeled_ratio = labeled_ratio
        self.d_alpha = d_alpha
        self.is_independent = is_independent
        self.train_data, self.test_data, self.class_num, self.mediator_length = self.get_datasets(self.dataset_name)
        self.local_size = len(self.train_data) / self.mediator_length
        self.idxs = self.get_idxs()

    # Use Dirichlet distribution to partition dataset
    def get_composition(self):
        tp_list = []
        for i in range(self.mediator_length):
            tp_list.append(self.d_alpha)
        tp_list[self.rank % self.mediator_length] = 100.0
        tp_list = tuple(tp_list)
        composition_ratio = np.random.dirichlet(tp_list)
        return (composition_ratio*self.local_size).astype(int)

    # Partition dataset for varied clients
    def get_idxs(self):
        local_idxs = []
        self.set_seed(0)
        if self.is_independent == True: # for large scale exp: samples on each client is independently sampled from respective class pools
            labels = np.array(self.train_data.targets)
            sorted_idxs = np.argsort(labels)
            composition = self.get_composition()
            print('local dataset composition: ' + str(composition))
            class_pool_size = len(self.train_data) / self.mediator_length
            for i in range(len(composition)):
                temp = random.sample(list(sorted_idxs[int(class_pool_size)*i : int(class_pool_size)*(i+1)]),composition[i])
                #temp = np.argsort(np.array(temp))
                for j in range(composition[i]):
                    #sample_index = sorted_idxs[int(class_pool_size*random.random()) + int(class_pool_size)*i] # randomly sampling

                    # sample_index = sorted_idxs[(class_pool_size/self.world_size*self.rank+j) % class_pool_size + class_pool_size*i]
                    local_idxs.append(temp[j])
            logging('local idxs: ' + str(len(local_idxs)))
        else:
            labels = np.random.rand(len(self.train_data)) if self.d_alpha >= 1 else np.array(self.train_data.targets) # alpha>1:IID; alpha<1:non-IID
            #print('labels: ' + str(labels))
            sorted_idxs = np.argsort(labels)
            #print('sorted_ids: ' + str(sorted_idxs))
            if self.rank%self.mediator_length != self.mediator_length-1:
                local_idxs = sorted_idxs[int(self.local_size*(self.rank%self.mediator_length)) : int(self.local_size*((self.rank+1)%self.mediator_length))]
            else:
                local_idxs = sorted_idxs[int(self.local_size*(self.rank%self.mediator_length)) : ]
            logging('local idxs: ' + str(len(local_idxs)))
        return local_idxs

    def get_datasets(self, dataset_name):
        # Load dataset based on dataset name
        if dataset_name == 'Mnist':
            train_dataset = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
            class_num = 10
            mediator_length = 6
        return train_dataset, test_dataset, class_num, mediator_length




    # Allocate clients for mediator after pre-train, add in as_manager.py
    def set_mediator(self):
        clients_no_allocate = self.world_size
        allocate_flag = np.ones(self.world_size)
        num_mediator = 1
        #print('gathered mu mu: ' + str(self.gathered_mu_mu[0]))
        #print('gathered mu mu: ' + str(self.gathered_mu_sigma[0]))
        #print('gathered mu mu: ' + str(self.gathered_sigma_mu[0]))
        #print('gathered mu mu: ' + str(self.gathered_sigma_sigma[0]))


        while clients_no_allocate > 0:
            # xxx_in_mediator record the information(mu,sigma,and num_sample) used in the current mediator
            num_samples_in_mediator = []
            mu_mu_in_mediator = []
            mu_sigma_in_mediator = []
            sigma_mu_in_mediator = []
            sigma_sigma_in_mediator = []

            # Select a client for a mediator
            for i in range(self.world_size):
                if allocate_flag[i] > 0:
                    allocate_flag[i] = 0
                    clients_no_allocate = clients_no_allocate - 1
                    self.mediators_allocate[i] = num_mediator
                    num_samples_in_mediator.append(self.gathered_num_samples[0][i])
                    mu_mu_in_mediator.append(self.gathered_mu_mu[0][i])
                    mu_sigma_in_mediator.append(self.gathered_mu_sigma[0][i])
                    sigma_mu_in_mediator.append(self.gathered_sigma_mu[0][i])
                    sigma_sigma_in_mediator.append(self.gathered_sigma_sigma[0][i])
                    break

            # The first last kl divergence is one client with the global
            #logging('self.gathered_mu_mu[0][i]: ' + str(self.gathered_mu_mu[0][i]) + ' ;self.global_mu_mu: ' + str(self.global_mu_mu) + ' ;self.gathered_mu_sigma[0][i]: ' + str(self.gathered_mu_sigma[0][i]) + ' ;self.global_mu_sigma: ' + str(self.global_mu_sigma))
            #logging('self.gathered_sigma_mu[0][i]: ' + str(self.gathered_sigma_mu[0][i]) + ' ;self.global_sigma_mu: ' + str(self.global_sigma_mu) + ' ;self.gathered_sigma_sigma[0][i]: ' + str(self.gathered_sigma_sigma[0][i]) + ' ;self.global_sigma_sigma: ' + str(self.global_sigma_sigma))
            last_mu_kl_div = self.kl_divergence(self.gathered_mu_mu[0][i], self.global_mu_mu, self.gathered_mu_sigma[0][i], self.global_mu_sigma)
            last_sigma_kl_div = self.kl_divergence(self.gathered_sigma_mu[0][i], self.global_sigma_mu, self.gathered_sigma_sigma[0][i], self.global_sigma_sigma)
            #print('last mu kl div: ' + str(last_mu_kl_div))
            #print('last sigma kl div: ' + str(last_sigma_kl_div))
            last_kl_div = self.alpha * last_mu_kl_div + (1 - self.alpha) * last_sigma_kl_div

            # Calculate the most suitable clients for current mediator
            while True:
                if clients_no_allocate < 1:
                    break

                # Calculate minimum kl divergence
                min_kl_div = torch.tensor(99999.0)
                client_index = -1
                for i in range(self.world_size):
                    if allocate_flag[i] > 0:
                        current_mu_mu, current_mu_sigma =  self.calculate_mu_and_sigma(mu_mu_in_mediator, self.gathered_mu_mu[0][i], mu_sigma_in_mediator, self.gathered_mu_sigma[0][i], num_samples_in_mediator, self.gathered_num_samples[0][i])
                        current_mu_kl_div = self.kl_divergence(current_mu_mu, self.global_mu_mu, current_mu_sigma, self.global_mu_sigma)
                        current_sigma_mu, current_sigma_sigma =  self.calculate_mu_and_sigma(sigma_mu_in_mediator, self.gathered_sigma_mu[0][i], sigma_sigma_in_mediator, self.gathered_sigma_sigma[0][i], num_samples_in_mediator, self.gathered_num_samples[0][i])
                        current_sigma_kl_div = self.kl_divergence(current_sigma_mu, self.global_sigma_mu, current_sigma_sigma, self.global_sigma_sigma)
                        current_kl_div = self.alpha * current_mu_kl_div + (1 - self.alpha) * current_sigma_kl_div

                        #print("current_kl_div type: " + str(type(current_kl_div)))
                        #print("min_kl_div type: " + str(type(min_kl_div)))

                        if current_kl_div < min_kl_div:
                            min_kl_div = current_kl_div
                            client_index = i

                current_kl_div = min_kl_div
                logging("last kl div: " + str(last_kl_div))
                logging("current kl div: " + str(current_kl_div))

                # allocate the client for the current mediator
                if current_kl_div < last_kl_div:
                    #print('mokaiwei')
                    last_kl_div = current_kl_div
                    self.mediators_allocate[client_index] = num_mediator
                    allocate_flag[client_index] = 0
                    num_samples_in_mediator.append(self.gathered_num_samples[0][client_index])
                    mu_mu_in_mediator.append(self.gathered_mu_mu[0][client_index])
                    mu_sigma_in_mediator.append(self.gathered_mu_sigma[0][client_index])
                    sigma_mu_in_mediator.append(self.gathered_sigma_mu[0][client_index])
                    sigma_sigma_in_mediator.append(self.gathered_sigma_sigma[0][client_index])
                    clients_no_allocate = clients_no_allocate - 1
                    if clients_no_allocate < 1:
                        break
                else:
                    #print('abb')
                    num_mediator = num_mediator + 1
                    break         # end and save current mediator

        self.mediators_allocate = np.array([1,2,3,1,2,3,1,2,3,4,5,6,4,5,6,4,5,6,7,8,8,7,7,8])
        unique_id = np.unique(self.mediators_allocate)
        mediators_client_temp = []
        for j in unique_id:
            mediators_client_temp.append(int((np.arange(self.world_size)[self.mediators_allocate == j])[0]))
        self.mediators_num = len(mediators_client_temp)
        mediators_client_temp = np.array(mediators_client_temp)
        for i in range(len(mediators_client_temp)):
            self.mediators_client[i] = mediators_client_temp[i]


        logging("mediator allocate: " + str(self.mediators_allocate))
        logging("finish setting mediators.")


    # synchronization
    def sync(self, model, iter_id):
        if self.if_global_stable == 0:
            if self.rank == 0:
                sign_list = [torch.zeros_like(self.if_stable) for _ in range(self.mediators_num)]
            else:
                sign_list = []
            dist.gather(self.if_stable, gather_list=sign_list, dst = 0, group=self.group_to)
            #logging('sign_list: ' + str(sign_list))
            for i in range(len(sign_list)):
                if sign_list[i] == 0:
                    break
                if i == len(sign_list)-1 and sign_list[i] == 1:
                    self.next_sync_within_mediators = iter_id
                    self.if_global_stable = torch.tensor(1)
            dist.broadcast(self.if_global_stable, 0, group=self.group)
            next_sync_within_mediators_transfered = torch.tensor(self.next_sync_within_mediators)
            dist.broadcast(next_sync_within_mediators_transfered, 0, group=self.group)
            self.next_sync_within_mediators = int(next_sync_within_mediators_transfered)
            #logging('if_global_stable: ' + str(self.if_global_stable))
        if iter_id == self.next_sync_within_mediators and self.if_global_stable == 1:
            if self.rank in self.mediators_client:
                if self.mediators_num > 1:
                    if CUDA:
                        model.cpu()
                        #model_added.cpu()
                    # Aggregation


                    for (i, p) in enumerate(model.parameters()):
                        if self.rank == 0:
                            grad_list = [torch.zeros_like(p.data) for _ in range(self.mediators_num)]
                        else:
                            grad_list = []
                        dist.gather(p.data, gather_list=grad_list, dst = 0, group=self.group_to)
                        if self.rank == 0:
                            p.data = sum(grad_list) / self.mediators_num # reduce to average
                        dist.broadcast(p.data, 0, group=self.group_to)




                    if CUDA:
                        model.cuda()
                        #model_added.cuda()

            self.next_sync_within_mediators = iter_id + self.sync_frequency*self.sync_mediators_freq
            self.mediators_round_id += 1
            #self.last_model = copy.deepcopy(model)
            #self.last_model_added = copy.deepcopy(model_added)

            return True
        return False


class AuxiliaryDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        Auxiliary Deep Generative Models [Maaløe 2016]
        code replication. The ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions.

        :param dims: dimensions of x, y, z, a and hidden layers.
        """
        [x_dim, y_dim, z_dim, a_dim, h_dim] = dims
        super(AuxiliaryDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim, h_dim])

        self.aux_encoder = Encoder([x_dim, h_dim, a_dim])
        self.aux_decoder = Encoder([x_dim + z_dim + y_dim, list(reversed(h_dim)), a_dim])

        self.classifier = Classifier([x_dim + a_dim, h_dim[0], y_dim])

        self.encoder = Encoder([a_dim + y_dim + x_dim, h_dim, z_dim])
        self.decoder = Decoder([y_dim + z_dim, list(reversed(h_dim)), x_dim])

    def classify(self, x):
        # Auxiliary inference q(a|x)
        a, a_mu, a_log_var = self.aux_encoder(x)

        # Classification q(y|a,x)
        logits = self.classifier(torch.cat([x, a], dim=1))
        return logits

    def forward(self, x, y):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """
        # Auxiliary inference q(a|x)
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x)

        # Latent inference q(z|a,y,x)
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y, q_a], dim=1))

        # Generative p(x|z,y)
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        # Generative p(a|z,y,x)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([x, y, z], dim=1))

        a_kl = self._kld(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        z_kl = self._kld(z, (z_mu, z_log_var))

        self.kl_divergence = a_kl + z_kl

        return x_mu


