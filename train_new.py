from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from networks_stylegan_apnagan import StyleGenerator, StyleDiscriminator
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss
from torch.autograd import Variable
from torch.nn.functional import interpolate, upsample
from models import networks
import torch
import torch.optim as optim
import torch.nn as nn
from util.image_pool import ImagePool
import time
import util.util as util
from PIL import Image
import numpy as np

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = StyleGenerator()
# generator = nn.DataParallel(generator)
generator.to(device)

generator.load_state_dict(torch.load("./checkpoints/trained_gen_2_10000.pth"))

discriminator_PP = networks.define_D(opt.P_input_nc+opt.P_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, opt.no_lsgan, opt.init_type, opt.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)
discriminator_PB = networks.define_D(opt.P_input_nc+18, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, opt.no_lsgan, opt.init_type, opt.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
disc_pp_optimizer = optim.Adam(discriminator_PP.parameters(), lr=0.0002)
disc_pb_optimizer = optim.Adam(discriminator_PB.parameters(), lr=0.0002)

Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor

# gen_GAN_loss_criteria = nn.BCEWithLogitsLoss().to(device)
gen_GAN_loss_criteria = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=Tensor)
gen_combL1_criteria = L1_plus_perceptualLoss(lambda_L1=0.5, lambda_perceptual=0.5, perceptual_layers=3, gpu_ids=[0], percep_is_l1=1)
disc_GAN_loss_criteria = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=Tensor)


def backward_D_basic(netD, real, fake):
  pred_real = netD(real)
  loss_D_real = disc_GAN_loss_criteria(pred_real, True) * opt.lambda_GAN
  pred_fake = netD(fake.detach())
  loss_D_fake = disc_GAN_loss_criteria(pred_fake, False) * opt.lambda_GAN
  loss_D = (loss_D_real + loss_D_fake) * 0.5
  loss_D.backward()
  return loss_D

total_steps = 0
percep_loss_history = []
GAN_loss_history = []
disc_loss_history = []
# for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
for epoch in range(2,101):
    epoch_start_time = time.time()
    epoch_iter = 0
    percep_epoch_loss = []
    gan_epoch_loss = []
    disc_epoch_loss = []

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        gen_optimizer.zero_grad()

        input_P1_set = torch.FloatTensor(opt.batchSize, opt.P_input_nc, opt.fineSize, opt.fineSize)
        input_BP1_set =torch.FloatTensor(opt.batchSize, opt.BP_input_nc, opt.fineSize, opt.fineSize)
        # input_BP1_set_fake =torch.FloatTensor(opt.batchSize, opt.BP_input_nc, opt.fineSize, opt.fineSize)
        input_P2_set = torch.FloatTensor(opt.batchSize, opt.P_input_nc, opt.fineSize, opt.fineSize)
        input_BP2_set =torch.FloatTensor(opt.batchSize, opt.BP_input_nc, opt.fineSize, opt.fineSize)
        # input_BP2_set_fake =torch.FloatTensor(opt.batchSize, opt.BP_input_nc, opt.fineSize, opt.fineSize)
        
        input_P1, input_BP1 = data['P1'], data['BP1']
        input_P2, input_BP2 = data['P2'], data['BP2']
        # print("NNNNNNNNN")
        # print(input_P1.shape)
        input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        input_BP1_set.resize_(input_BP1.size()).copy_(input_BP1)
        # input_BP1_set_fake.resize_(input_BP1.size()).copy_(input_BP1)
        input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        input_BP2_set.resize_(input_BP2.size()).copy_(input_BP2)
        # input_BP2_set_fake.resize_(input_BP2.size()).copy_(input_BP2)
        image_paths = data['P1_path'][0] + '___' + data['P2_path'][0]

        input_P1 = Variable(input_P1_set).to(device)
        input_BP1 = Variable(input_BP1_set).to(device)
        # input_BP1_fake = Variable(input_BP1_set_fake)

        input_P2 = Variable(input_P2_set).to(device)
        input_BP2 = Variable(input_BP2_set).to(device)
        # input_BP2_fake = Variable(input_BP2_set_fake)

        # G_input = [input_P1,
        #            torch.cat((input_BP1, input_BP2), 1)]
        fake_p2 = generator(input_P1,torch.cat((input_BP1, input_BP2), 1))
        fake_p2 = interpolate(fake_p2, size=(256,176), mode="bilinear")




        # print(input_BP2.size(), " ", fake_p2.size())
        # interpolate(input_BP1_fake, size=(opt.batchSize, 18, 1024, 1024))
        # interpolate(input_BP2_fake, size=(opt.batchSize, 18, 1024, 1024))
        pred_fake_PB = discriminator_PB(torch.cat((fake_p2, input_BP2), 1))
        loss_G_GAN_PB = gen_GAN_loss_criteria(pred_fake_PB, True)
        pred_fake_PP = discriminator_PP(torch.cat((fake_p2, input_P1), 1))
        loss_G_GAN_PP = gen_GAN_loss_criteria(pred_fake_PP, True)
        losses = gen_combL1_criteria(fake_p2, input_P2)
        loss_G_L1 = losses[0]
        loss_originL1 = losses[1].item()
        loss_perceptual = losses[2].item()
        pair_L1loss = loss_G_L1
        pair_GANloss = loss_G_GAN_PB * opt.lambda_GAN
        pair_GANloss += loss_G_GAN_PP * opt.lambda_GAN
        pair_GANloss = pair_GANloss / 2     
        pair_loss = pair_L1loss + pair_GANloss
        pair_loss.backward()
        pair_L1loss = pair_L1loss.item()
        pair_GANloss = pair_GANloss.item()
        gen_optimizer.step()

        fake_PP_pool = ImagePool(opt.pool_size)
        fake_PB_pool = ImagePool(opt.pool_size)
        disc_pp_optimizer.zero_grad()
        real_PP = torch.cat((input_P2, input_P1), 1)
        fake_PP = fake_PP_pool.query( torch.cat((fake_p2, input_P1), 1).data)
        loss_D_PP = backward_D_basic(discriminator_PP, real_PP, fake_PP)
        disc_pp_optimizer.step()
        loss_D_PP = loss_D_PP.item()

        disc_pb_optimizer.zero_grad()
        real_PB = torch.cat((input_P2, input_BP2), 1)
        fake_PB = fake_PB_pool.query( torch.cat((fake_p2, input_BP2), 1).data)
        loss_D_PB = backward_D_basic(discriminator_PB, real_PB, fake_PB)
        disc_pb_optimizer.step()
        loss_D_PB = loss_D_PB.item()

        # if total_steps % opt.print_freq == 0:
        #     errors = model.get_current_errors()
        #     t = (time.time() - iter_start_time) / opt.batchSize
        #     visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        #     if opt.display_id > 0:
        #         visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        percep_epoch_loss.append(loss_perceptual)
        gan_epoch_loss.append(pair_GANloss)
        disc_epoch_loss.append(loss_D_PP+loss_D_PB)
        
        if total_steps % opt.print_freq == 0:
            print("Current Iteration == (epoch %d, total_steps %d)" % (epoch, total_steps))
            print("Perceptual Loss = %.4f  ; Gan Adv Loss = %.4f  ;  Disc Adv Loss = %.4f" % (loss_perceptual, pair_GANloss, (loss_D_PP+loss_D_PB)))
            fake_p2 = util.tensor2im(fake_p2.data)
            img = Image.fromarray(fake_p2)
            img.save("fashion_data/outputs/out_{}_{}.png".format(epoch,i))

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            torch.save(generator.state_dict(), "./checkpoints/trained_gen_{}_{}.pth".format(epoch,total_steps))

    percep_epoch_loss = np.mean(np.array(percep_epoch_loss))
    gan_epoch_loss = np.mean(np.array(gan_epoch_loss))
    disc_epoch_loss = np.mean(np.array(disc_epoch_loss))

    percep_loss_history.append(percep_epoch_loss)
    GAN_loss_history.append(gan_epoch_loss)
    disc_loss_history.append(disc_epoch_loss)

    np.save('percep_loss_history.npy',np.array(percep_loss_history))
    np.save('GAN_loss_history.npy',np.array(GAN_loss_history))
    np.save('disc_loss_history.npy',np.array(disc_loss_history))
    # with open('percep_loss_history.npy', 'wb') as f:
    #     np.save(f, np.array(percep_epoch_loss))

    # with open('GAN_loss_history.npy', 'wb') as f:
    #     np.save(f, np.array(gan_epoch_loss))

    # with open('disc_loss_history.npy', 'wb') as f:
    #     np.save(f, np.array(disc_epoch_loss))

    
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        torch.save(generator.state_dict(), "./checkpoints/trained_gen_{}_{}.pth".format(epoch,total_steps))

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # generator.update_learning_rate()

  